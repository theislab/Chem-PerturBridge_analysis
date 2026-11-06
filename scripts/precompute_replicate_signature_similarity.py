#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from peer_baselines import (
    DEFAULT_PEER_SAMPLING_SEED,
    PreparedSpearmanRows,
    direction_agreement_against_peers,
    exact_mean_excluding_row_mask,
    finite_column_totals,
    prepare_spearman_rows,
    select_peer_indices,
    spearman_against_peers,
    spearman_against_prepared_peers,
    summarize_peer_scores,
)


REPO_ROOT = SCRIPT_DIR.parent
CLUSTER_DATA_ROOT = Path("/lustre/groups/ml01/workspace/olga.novitskaia/data_updated")
REPO_DATA_ROOT = REPO_ROOT / "data" / "theislab_temp"


def _default_source_data_root() -> Path:
    """Prefer an explicit override, then the in-repo data directory, then the cluster path."""
    override = os.environ.get("CPB_SOURCE_DATA_ROOT", "").strip()
    if override:
        return Path(override)
    if REPO_DATA_ROOT.is_dir():
        return REPO_DATA_ROOT
    return CLUSTER_DATA_ROOT


SOURCE_DATA_ROOT = _default_source_data_root()


def sep_rep_dataset_dir(dataset_name: str, filter_min_cells: int) -> Path:
    """Resolve a dataset's separate-replicate DEG directory across both known layouts.

    The published release stages files flat, as `<dataset>/sep_rep/`; the cluster keeps the
    full pipeline path. Prefer whichever exists so the same code runs in both places, and
    fall back to the cluster shape when neither is present, which keeps the error message
    pointing at the canonical location.
    """
    flat = SOURCE_DATA_ROOT / dataset_name / "sep_rep"
    if flat.is_dir():
        return flat
    return (
        SOURCE_DATA_ROOT
        / dataset_name
        / "deg_data"
        / "sep_rep"
        / "full"
        / "qc_false"
        / f"filter_min_cells_{int(filter_min_cells)}"
        / "results"
    )
DEFAULT_SOURCE_DATASET_DIRS = {
    "sciplex": sep_rep_dataset_dir("sciplex", 10),
    "tahoe": sep_rep_dataset_dir("tahoe", 50),
    "op3": sep_rep_dataset_dir("op3", 10),
    "cigs_mce": sep_rep_dataset_dir("cigs_mce", 0),
    "novartis_batch_1000": sep_rep_dataset_dir("novartis_batch_1000", 0),
    "vcpi_0001": sep_rep_dataset_dir("vcpi_0001", 0),
    "cigs_tcm": sep_rep_dataset_dir("cigs_tcm", 0),
    "vcpi_0002": sep_rep_dataset_dir("vcpi_0002", 0),
    "gdpx2": sep_rep_dataset_dir("gdpx2", 0),
    "dilimap_train_val": sep_rep_dataset_dir("dilimap_train_val", 0),
    "l1000_phase1": sep_rep_dataset_dir("l1000_phase1", 0),
    "l1000_phase2": sep_rep_dataset_dir("l1000_phase2", 0),
}
DEFAULT_PROCESSED_DATA_ROOT = Path(
    os.environ.get("CPB_PROCESSED_DATA_ROOT", str(SOURCE_DATA_ROOT))
)
L1000_DATASETS = {"l1000_phase1", "l1000_phase2"}
PRETTY_DATASET_LABELS = {
    "sciplex": "sci-Plex",
    "tahoe": "Tahoe-100M",
    "op3": "OP3",
    "cigs_mce": "CIGS-MCE",
    "novartis_batch_1000": "Novartis DRUG-seq",
    "vcpi_0001": "VCPI-0001",
    "cigs_tcm": "CIGS-TCM",
    "vcpi_0002": "VCPI-0002",
    "gdpx2": "GDPx2",
    "dilimap_train_val": "DILImap",
    "l1000_phase1": "L1000 Phase I",
    "l1000_phase2": "L1000 Phase II",
}
DEG_P_THRESHOLD = 0.05
DEG_ABS_LOGFC_THRESHOLD = 0.2
DE_OVERLAP_K_VALUES = (50, 100, 200)
DEFAULT_MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME = 10
# Per-peer ("single-signature") baselines score every same-context different-compound peer
# separately instead of averaging them into a centroid first. Replicate peer sets can hold
# thousands of rows, so --max-baseline-peers caps how many are scored; the cap is applied
# by select_peer_indices with a deterministic per-condition seed, and both the total and
# scored peer counts are recorded so a capped run is never mistaken for a full one.
MAX_BASELINE_PEERS: Optional[int] = None
PEER_SAMPLING_SEED = DEFAULT_PEER_SAMPLING_SEED
# Tables 10 and 11 report the adj.P.Value < 0.05 DEG-restricted metrics.
PEER_BASELINE_DEG_METRICS = ("deg_lfc_spearman_sym", "direction_agreement")
DEG_DEFINITION_CONFIG = {
    "p05": {
        "display": "adj.P.Value < 0.05",
        "require_abs_logfc": False,
    },
    "p05_lfc02": {
        "display": "adj.P.Value < 0.05 and |logFC| > 0.2",
        "require_abs_logfc": True,
    },
}
ADJ_PVALUE_LAYER_PREFERENCES = (
    "adj.P.Value.within_one_contrast",
    "adj.P.Value.across_all_contrasts",
)
METADATA_OBS_COLUMNS = [
    "id",
    "cell_type",
    "pubchem_cid",
    "perturbagen",
    "perturbagen_name",
    "perturbation_label",
    "pert_time_h",
    "pert_dose_uM",
    "is_control",
]

TASK_MANIFEST_FILE_NAME = "task_manifest.tsv"
TASK_INPUT_DIR_NAME = "task_inputs"
TASK_OUTPUT_DIR_NAME = "task_outputs"
LINE_GLOBAL_SHARED_GENE_KEYS_FILE_NAME = "line_global_shared_gene_keys.tsv"
TASK_CONFIG_FILE_NAME = "task_config.json"
DATASET_METADATA_CACHE_DIR_NAME = "dataset_metadata_cache"


@dataclass
class GeneInfo:
    path: Path
    unique_gene_keys: np.ndarray
    unique_gene_positions: np.ndarray
    gene_to_var_pos: dict[str, int]


@dataclass
class PrepareResult:
    output_dir: Path
    task_manifest_path: Path
    task_output_dir: Path


@dataclass
class ReshardResult:
    output_dir: Path
    task_manifest_path: Path
    task_output_dir: Path


@dataclass
class BaselineContextVectors:
    """All reusable vectors for one dataset / line / time / dose / gene-set context."""

    rows: pd.DataFrame
    local_logfc: np.ndarray
    local_t: np.ndarray
    global_logfc: np.ndarray
    global_t: np.ndarray
    finite_totals: dict[str, tuple[np.ndarray, np.ndarray]]
    prepared_local_logfc: dict[bytes, PreparedSpearmanRows]


GENE_INFO_CACHE: dict[str, GeneInfo] = {}
SHARED_GENE_KEY_CACHE: dict[tuple[str, ...], np.ndarray] = {}
GENE_POSITION_CACHE: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
DATASET_LINE_GENE_KEY_CACHE: dict[tuple[str, str], np.ndarray] = {}
ADJ_PVALUE_LAYER_CACHE: dict[str, str] = {}


def add_common_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/replicate_signature_similarity_sep_rep"),
        help="Directory where TSV outputs will be written.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names to include, or 'all'.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="K for signed overlap@k on t-statistics.",
    )
    parser.add_argument(
        "--min-replicates-per-condition",
        type=int,
        default=2,
        help="Minimum number of replicate DE rows required for a condition.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N scored tasks.",
    )
    parser.add_argument(
        "--test-one-line-per-dataset",
        action="store_true",
        help=(
            "Test mode: after normal filtering, keep only one cell line per dataset "
            "(the line with the most retained conditions)."
        ),
    )
    parser.add_argument(
        "--test-max-conditions-per-dataset",
        type=int,
        default=0,
        help=(
            "Optional extra test-mode cap on retained conditions per dataset after line selection. "
            "Use 0 to keep all retained conditions for the selected test line."
        ),
    )
    parser.add_argument(
        "--compute-baseline-metrics",
        action="store_true",
        help=(
            "Also compute within-dataset baseline metrics using the mean signature across "
            "same line / time / dose but other drugs."
        ),
    )
    parser.add_argument(
        "--compute-deg-metrics",
        action="store_true",
        help=(
            "Also compute DEG-focused replicate metrics using adjusted p-value thresholds "
            "and baseline comparisons on the same line / time / dose other-drug baseline."
        ),
    )
    parser.add_argument(
        "--compute-retrieval-metrics",
        action="store_true",
        help=(
            "Also compute within-dataset replicate retrieval metrics in line-time strata, "
            "including the injected same-dose other-drug baseline candidate."
        ),
    )
    parser.add_argument(
        "--min-retrieval-compounds-per-line-time",
        type=int,
        default=DEFAULT_MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME,
        help=(
            "Minimum number of unique compounds required in a line-time stratum to score "
            "replicate retrieval."
        ),
    )
    parser.add_argument(
        "--max-baseline-peers",
        type=int,
        default=512,
        help=(
            "Cap how many same line / time / dose other-drug peers are scored individually "
            "for the per-peer baselines. Omit to score every peer. Capped runs record both "
            "the total and scored peer counts. The exact centroid always uses every "
            "eligible peer. Default: 512."
        ),
    )
    parser.add_argument(
        "--peer-sampling-seed",
        type=int,
        default=DEFAULT_PEER_SAMPLING_SEED,
        help="Seed for deterministic capped peer sampling. Default: 20260505.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Precompute within-dataset replicate signature similarity metrics from sep_rep h5ads."
    )
    subparsers = parser.add_subparsers(dest="command")

    run_all_parser = subparsers.add_parser(
        "run-all",
        help="Run prepare, all tasks, and merge sequentially.",
    )
    add_common_run_args(run_all_parser)
    run_all_parser.add_argument(
        "--conditions-per-task",
        type=int,
        default=250,
        help="Maximum number of retained conditions to score in one task shard.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Scan metadata, write retained-condition inventory, and create task shards.",
    )
    add_common_run_args(prepare_parser)
    prepare_parser.add_argument(
        "--conditions-per-task",
        type=int,
        default=250,
        help="Maximum number of retained conditions to score in one task shard.",
    )

    reshard_parser = subparsers.add_parser(
        "reshard",
        help="Reuse cached prepare outputs to rebuild task shards with a different shard size.",
    )
    reshard_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/replicate_signature_similarity_sep_rep"),
        help="Base output directory that contains prepare outputs.",
    )
    reshard_parser.add_argument(
        "--conditions-per-task",
        type=int,
        required=True,
        help="Maximum number of retained conditions to score in one task shard.",
    )

    run_task_parser = subparsers.add_parser(
        "run-task",
        help="Score one prepared task shard.",
    )
    run_task_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/replicate_signature_similarity_sep_rep"),
        help="Base output directory that contains task inputs and line-global gene keys.",
    )
    run_task_parser.add_argument(
        "--task-file",
        type=Path,
        required=True,
        help="Task manifest written by the prepare step.",
    )
    run_task_parser.add_argument(
        "--task-id",
        type=int,
        required=True,
        help="1-based task id from the task manifest.",
    )
    run_task_parser.add_argument(
        "--task-output-dir",
        type=Path,
        default=None,
        help="Directory where this task should write its outputs. Defaults to <output-dir>/task_outputs.",
    )
    run_task_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="K for signed overlap@k on t-statistics.",
    )
    run_task_parser.add_argument(
        "--compute-baseline-metrics",
        action="store_true",
        help=(
            "Also compute within-dataset baseline metrics using the mean signature across "
            "same line / time / dose but other drugs."
        ),
    )
    run_task_parser.add_argument(
        "--compute-deg-metrics",
        action="store_true",
        help=(
            "Also compute DEG-focused replicate metrics using adjusted p-value thresholds "
            "and baseline comparisons on the same line / time / dose other-drug baseline."
        ),
    )
    run_task_parser.add_argument(
        "--compute-retrieval-metrics",
        action="store_true",
        help="Also compute within-dataset strict matched-condition retrieval summaries.",
    )
    run_task_parser.add_argument(
        "--min-retrieval-compounds-per-line-time",
        type=int,
        default=DEFAULT_MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME,
        help="Minimum unique compounds required in a line-time stratum for retrieval scoring.",
    )
    run_task_parser.add_argument(
        "--max-baseline-peers",
        type=int,
        default=512,
        help=(
            "Cap how many same line / time / dose other-drug peers are scored individually "
            "for the per-peer baselines. Use a value greater than the available peer "
            "count to score every peer. The centroid is never capped."
        ),
    )
    run_task_parser.add_argument(
        "--peer-sampling-seed",
        type=int,
        default=DEFAULT_PEER_SAMPLING_SEED,
        help="Seed for deterministic capped peer sampling. Default: 20260505.",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge prepared task outputs into the final summary TSVs.",
    )
    merge_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/replicate_signature_similarity_sep_rep"),
        help="Base output directory that contains prepare outputs.",
    )
    merge_parser.add_argument(
        "--task-file",
        type=Path,
        required=True,
        help="Task manifest written by the prepare step.",
    )
    merge_parser.add_argument(
        "--task-output-dir",
        type=Path,
        default=None,
        help="Directory with per-task outputs. Defaults to <output-dir>/task_outputs.",
    )
    merge_parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="K for signed overlap@k on t-statistics.",
    )
    merge_parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail if any expected task output is missing.",
    )
    merge_parser.add_argument(
        "--existing-results-dir",
        type=Path,
        default=None,
        help=(
            "Optional previous output directory whose condition-level summaries should be "
            "combined with the current task outputs before final summaries are rebuilt."
        ),
    )

    return parser


def parse_args() -> argparse.Namespace:
    argv = list(sys.argv[1:])
    known_commands = {"run-all", "prepare", "reshard", "run-task", "merge"}
    if not argv or argv[0] not in known_commands:
        argv = ["run-all"] + argv
    return build_parser().parse_args(argv)


def pretty_label(dataset_name: str) -> str:
    return PRETTY_DATASET_LABELS.get(str(dataset_name), str(dataset_name))


def format_numeric(value: object) -> str:
    if pd.isna(value):
        return ""
    value = float(value)
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.12g}"


def normalize_pubchem_cid_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return format_numeric(float(numeric))
    return text


def coerce_control_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    lowered = series.astype("string").fillna("").astype(str).str.strip().str.lower()
    return lowered.isin({"1", "true", "t", "yes", "y", "dmso", "control"})


def first_non_empty(*series_or_values: object) -> str:
    for value in series_or_values:
        if isinstance(value, pd.Series):
            for item in value.astype("string").fillna("").astype(str):
                item = item.strip()
                if item and item.lower() != "nan":
                    return item
        else:
            item = str(value).strip()
            if item and item.lower() != "nan":
                return item
    return ""


def mean_available(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def strict_mean_if_all_defined(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        return float("nan")
    return float(values.mean())


def median_available(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def safe_column_spearman(frame: pd.DataFrame, x_col: str, y_col: str) -> float:
    if x_col not in frame.columns or y_col not in frame.columns:
        return float("nan")
    subset = frame[[x_col, y_col]].copy()
    subset[x_col] = pd.to_numeric(subset[x_col], errors="coerce")
    subset[y_col] = pd.to_numeric(subset[y_col], errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan).dropna()
    if len(subset) < 2:
        return float("nan")
    if subset[x_col].nunique() < 2 or subset[y_col].nunique() < 2:
        return float("nan")
    return float(subset[x_col].corr(subset[y_col], method="spearman"))


def sampled_replicate_pair_indices(n_replicates: int) -> list[tuple[int, int]]:
    n_replicates = int(n_replicates)
    if n_replicates < 2:
        return []
    if n_replicates == 2:
        return [(0, 1)]
    return [(idx, (idx + 1) % n_replicates) for idx in range(n_replicates)]


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return np.asarray([], dtype=np.float64)
    tri = np.triu_indices(matrix.shape[0], k=1)
    return np.asarray(matrix[tri], dtype=np.float64)


def pair_values_from_similarity_matrix(
    matrix: np.ndarray,
    pair_indices: list[tuple[int, int]],
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not pair_indices:
        return np.asarray([], dtype=np.float64)
    return np.asarray([matrix[left_idx, right_idx] for left_idx, right_idx in pair_indices], dtype=np.float64)


def row_spearman_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape={matrix.shape}")
    if matrix.shape[0] == 0:
        return np.empty((0, 0), dtype=np.float64)
    if matrix.shape[0] == 1:
        return np.ones((1, 1), dtype=np.float64)
    ranked = np.apply_along_axis(stats.rankdata, 1, matrix)
    with np.errstate(invalid="ignore"):
        similarity = np.corrcoef(ranked)
    return np.asarray(similarity, dtype=np.float64)


def signed_overlap_at_k(left_values: np.ndarray, right_values: np.ndarray, *, k: int) -> float:
    left_values = np.asarray(left_values, dtype=np.float64)
    right_values = np.asarray(right_values, dtype=np.float64)
    finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
    if finite_mask.sum() < 2:
        return float("nan")
    left_values = left_values[finite_mask]
    right_values = right_values[finite_mask]
    k_eff = int(min(k, left_values.size, right_values.size))
    if k_eff < 1:
        return float("nan")
    left_order = np.argsort(left_values)
    right_order = np.argsort(right_values)
    left_down = set(left_order[:k_eff].tolist())
    right_down = set(right_order[:k_eff].tolist())
    left_up = set(left_order[-k_eff:].tolist())
    right_up = set(right_order[-k_eff:].tolist())
    return float(0.5 * ((len(left_up & right_up) / k_eff) + (len(left_down & right_down) / k_eff)))


def vector_spearman_similarity(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_values = np.asarray(left_values, dtype=np.float64)
    right_values = np.asarray(right_values, dtype=np.float64)
    finite_mask = np.isfinite(left_values) & np.isfinite(right_values)
    if int(finite_mask.sum()) < 2:
        return float("nan")
    left_ranked = stats.rankdata(left_values[finite_mask])
    right_ranked = stats.rankdata(right_values[finite_mask])
    with np.errstate(invalid="ignore"):
        similarity = np.corrcoef(left_ranked, right_ranked)[0, 1]
    return float(similarity)


def negative_l2_similarity_matrix(query_matrix: np.ndarray, candidate_matrix: np.ndarray) -> np.ndarray:
    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float64)
    diff = query_matrix[:, None, :] - candidate_matrix[None, :, :]
    with np.errstate(invalid="ignore"):
        distances = np.sqrt(np.sum(diff * diff, axis=2))
    return -distances


def normalized_best_positive_rank(scores: np.ndarray, positive_mask: np.ndarray) -> tuple[float, float]:
    positive_mask = np.asarray(positive_mask, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size < 2 or int(positive_mask.sum()) == 0:
        return float("nan"), float("nan")
    ranks = stats.rankdata(-scores, method="min")
    best_rank = float(np.min(ranks[positive_mask]))
    normalized_rank = 1.0 - ((best_rank - 1.0) / (len(scores) - 1.0))
    return float(best_rank), float(normalized_rank)


def difference_if_both_defined(observed: float, baseline: float) -> float:
    if not np.isfinite(observed) or not np.isfinite(baseline):
        return float("nan")
    return float(observed - baseline)


def read_h5ad_safely(path: str | Path, *, backed: str = "r") -> ad.AnnData:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Observation names are not unique.*",
            category=UserWarning,
        )
        return ad.read_h5ad(path, backed=backed)


def ensure_open_adata(source_path: str, open_adatas: dict[str, ad.AnnData]) -> ad.AnnData:
    if source_path not in open_adatas:
        open_adatas[source_path] = read_h5ad_safely(source_path, backed="r")
    return open_adatas[source_path]


def load_vectors_for_rows(
    rows: pd.DataFrame,
    *,
    gene_keys: np.ndarray,
    open_adatas: dict[str, ad.AnnData],
) -> tuple[list[Optional[np.ndarray]], list[Optional[np.ndarray]]]:
    indexed_rows = rows.reset_index(drop=True)
    logfc_vectors: list[Optional[np.ndarray]] = [None] * len(indexed_rows)
    t_vectors: list[Optional[np.ndarray]] = [None] * len(indexed_rows)
    if indexed_rows.empty or gene_keys.size < 2:
        return logfc_vectors, t_vectors

    for source_path, block in indexed_rows.groupby("source_path", sort=False):
        source_index = block.index.to_numpy(dtype=np.int64)
        row_positions = block["source_row_pos"].astype(int).to_numpy(dtype=np.int64)
        gene_positions = gene_positions_for_source(source_path, gene_keys)
        if gene_positions.size < 2:
            continue
        adata = ensure_open_adata(str(source_path), open_adatas)
        logfc = np.asarray(adata.layers["logFC"][row_positions][:, gene_positions], dtype=np.float32)
        t_stat = np.asarray(adata.layers["t"][row_positions][:, gene_positions], dtype=np.float32)
        if logfc.ndim == 1:
            logfc = logfc[np.newaxis, :]
        if t_stat.ndim == 1:
            t_stat = t_stat[np.newaxis, :]
        for idx_value, vector in zip(source_index, logfc):
            logfc_vectors[int(idx_value)] = np.asarray(vector, dtype=np.float64)
        for idx_value, vector in zip(source_index, t_stat):
            t_vectors[int(idx_value)] = np.asarray(vector, dtype=np.float64)
    return logfc_vectors, t_vectors


def adjusted_pvalue_layer_for_source_path(
    source_path: str,
    open_adatas: dict[str, ad.AnnData],
) -> str:
    if source_path not in ADJ_PVALUE_LAYER_CACHE:
        adata = ensure_open_adata(source_path, open_adatas)
        available_layers = set(adata.layers.keys())
        for candidate in ADJ_PVALUE_LAYER_PREFERENCES:
            if candidate in available_layers:
                ADJ_PVALUE_LAYER_CACHE[source_path] = candidate
                break
        else:
            raise KeyError(
                f"None of {ADJ_PVALUE_LAYER_PREFERENCES!r} are available in {source_path}; "
                f"available layers: {sorted(available_layers)!r}"
            )
    return ADJ_PVALUE_LAYER_CACHE[source_path]


def load_adjusted_pvalue_vectors_for_rows(
    rows: pd.DataFrame,
    *,
    gene_keys: np.ndarray,
    open_adatas: dict[str, ad.AnnData],
) -> list[Optional[np.ndarray]]:
    indexed_rows = rows.reset_index(drop=True)
    adj_p_vectors: list[Optional[np.ndarray]] = [None] * len(indexed_rows)
    if indexed_rows.empty or gene_keys.size < 2:
        return adj_p_vectors

    for source_path, block in indexed_rows.groupby("source_path", sort=False):
        source_index = block.index.to_numpy(dtype=np.int64)
        row_positions = block["source_row_pos"].astype(int).to_numpy(dtype=np.int64)
        gene_positions = gene_positions_for_source(source_path, gene_keys)
        if gene_positions.size < 2:
            continue
        adata = ensure_open_adata(str(source_path), open_adatas)
        adj_layer_name = adjusted_pvalue_layer_for_source_path(str(source_path), open_adatas)
        adj_p = np.asarray(adata.layers[adj_layer_name][row_positions][:, gene_positions], dtype=np.float32)
        if adj_p.ndim == 1:
            adj_p = adj_p[np.newaxis, :]
        for idx_value, vector in zip(source_index, adj_p):
            adj_p_vectors[int(idx_value)] = np.asarray(vector, dtype=np.float64)
    return adj_p_vectors


def build_baseline_vector(vectors: list[Optional[np.ndarray]]) -> Optional[np.ndarray]:
    present_vectors = [np.asarray(vector, dtype=np.float64) for vector in vectors if vector is not None]
    if not present_vectors:
        return None
    matrix = np.vstack(present_vectors).astype(np.float64)
    finite_mask = np.isfinite(matrix)
    counts = finite_mask.sum(axis=0)
    if not np.any(counts > 0):
        return None
    sums = np.where(finite_mask, matrix, 0.0).sum(axis=0)
    baseline = np.full(matrix.shape[1], np.nan, dtype=np.float64)
    valid_columns = counts > 0
    baseline[valid_columns] = sums[valid_columns] / counts[valid_columns]
    if baseline.ndim != 1:
        return None
    return np.asarray(baseline, dtype=np.float64)


def optional_vectors_to_matrix(
    vectors: list[Optional[np.ndarray]],
    *,
    n_columns: int,
) -> np.ndarray:
    """Preserve metadata-row alignment while representing unavailable vectors as NaN."""
    matrix = np.full((len(vectors), int(n_columns)), np.nan, dtype=np.float64)
    for row_index, vector in enumerate(vectors):
        if vector is None:
            continue
        values = np.asarray(vector, dtype=np.float64).reshape(-1)
        if values.size == int(n_columns):
            matrix[row_index] = values
    return matrix


def get_baseline_context_vectors(
    *,
    context_rows: pd.DataFrame,
    context_key: tuple[str, str, str],
    local_gene_keys: np.ndarray,
    global_gene_keys: np.ndarray,
    open_adatas: dict[str, ad.AnnData],
    cache: dict[tuple[object, ...], BaselineContextVectors],
) -> BaselineContextVectors:
    """Load a context once per local/global gene-set combination within a task."""
    cache_key = (
        *context_key,
        tuple(map(str, np.asarray(local_gene_keys, dtype=object).tolist())),
        tuple(map(str, np.asarray(global_gene_keys, dtype=object).tolist())),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    rows = normalize_source_metadata_frame(context_rows).reset_index(drop=True)
    local_logfc_vectors, local_t_vectors = load_vectors_for_rows(
        rows,
        gene_keys=local_gene_keys,
        open_adatas=open_adatas,
    )
    if np.array_equal(local_gene_keys, global_gene_keys):
        global_logfc_vectors = local_logfc_vectors
        global_t_vectors = local_t_vectors
    else:
        global_logfc_vectors, global_t_vectors = load_vectors_for_rows(
            rows,
            gene_keys=global_gene_keys,
            open_adatas=open_adatas,
        )
    matrices = {
        "local_logfc": optional_vectors_to_matrix(
            local_logfc_vectors,
            n_columns=int(local_gene_keys.size),
        ),
        "local_t": optional_vectors_to_matrix(
            local_t_vectors,
            n_columns=int(local_gene_keys.size),
        ),
        "global_logfc": optional_vectors_to_matrix(
            global_logfc_vectors,
            n_columns=int(global_gene_keys.size),
        ),
        "global_t": optional_vectors_to_matrix(
            global_t_vectors,
            n_columns=int(global_gene_keys.size),
        ),
    }
    loaded = BaselineContextVectors(
        rows=rows,
        local_logfc=matrices["local_logfc"],
        local_t=matrices["local_t"],
        global_logfc=matrices["global_logfc"],
        global_t=matrices["global_t"],
        finite_totals={
            matrix_name: finite_column_totals(matrix)
            for matrix_name, matrix in matrices.items()
        },
        prepared_local_logfc={},
    )
    cache[cache_key] = loaded
    return loaded


def get_prepared_local_context_spearman(
    context_vectors: BaselineContextVectors,
    column_mask: Optional[np.ndarray],
) -> PreparedSpearmanRows:
    """Reuse context row ranks across compounds sharing the same evaluation genes."""
    if column_mask is None:
        cache_key = b"all"
        matrix = context_vectors.local_logfc
    else:
        normalized_mask = np.asarray(column_mask, dtype=bool).reshape(-1)
        cache_key = normalized_mask.tobytes()
        matrix = context_vectors.local_logfc[:, normalized_mask]
    cached = context_vectors.prepared_local_logfc.get(cache_key)
    if cached is not None:
        return cached
    # Bound rank-cache memory when rare condition-specific finite masks differ.
    if len(context_vectors.prepared_local_logfc) >= 2:
        context_vectors.prepared_local_logfc.pop(
            next(iter(context_vectors.prepared_local_logfc))
        )
    prepared = prepare_spearman_rows(matrix)
    context_vectors.prepared_local_logfc[cache_key] = prepared
    return prepared


def deg_metric_names() -> list[str]:
    return [
        "deg_lfc_spearman_sym",
        "de_overlap_refn_sym",
        "direction_agreement",
        *[f"de_overlap_k{k}" for k in DE_OVERLAP_K_VALUES],
    ]


def empty_deg_metric_dict() -> dict[str, float]:
    return {
        metric_name: float("nan")
        for metric_name in deg_metric_names()
    }


def deg_mask(
    logfc_values: np.ndarray,
    adj_p_values: np.ndarray,
    definition_key: str,
) -> np.ndarray:
    mask = np.isfinite(logfc_values) & np.isfinite(adj_p_values)
    mask &= np.asarray(adj_p_values, dtype=np.float64) < DEG_P_THRESHOLD
    if DEG_DEFINITION_CONFIG[definition_key]["require_abs_logfc"]:
        mask &= np.abs(np.asarray(logfc_values, dtype=np.float64)) > DEG_ABS_LOGFC_THRESHOLD
    return mask


def ranked_gene_keys_from_mask(
    gene_keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return np.empty(0, dtype=object)
    order = np.argsort(-np.abs(np.asarray(values, dtype=np.float64)[idx]), kind="stable")
    return np.asarray(gene_keys, dtype=object)[idx[order]]


def ranked_gene_keys_by_abs_values(
    gene_keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    finite_mask = np.isfinite(np.asarray(values, dtype=np.float64))
    idx = np.flatnonzero(finite_mask)
    if idx.size == 0:
        return np.empty(0, dtype=object)
    order = np.argsort(-np.abs(np.asarray(values, dtype=np.float64)[idx]), kind="stable")
    return np.asarray(gene_keys, dtype=object)[idx[order]]


def top_overlap_fraction(
    ranked_left: np.ndarray,
    ranked_right: np.ndarray,
    n: int,
) -> float:
    if int(n) <= 0:
        return float("nan")
    if len(ranked_left) < int(n) or len(ranked_right) < int(n):
        return float("nan")
    left_set = {str(gene_key) for gene_key in ranked_left[: int(n)].tolist()}
    right_set = {str(gene_key) for gene_key in ranked_right[: int(n)].tolist()}
    return float(len(left_set & right_set) / float(n))


def direction_agreement_with_masks(
    left_values: np.ndarray,
    right_values: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    overlap_mask = left_mask & right_mask & np.isfinite(left_values) & np.isfinite(right_values)
    if int(overlap_mask.sum()) == 0:
        return float("nan")
    left_signs = np.sign(np.asarray(left_values, dtype=np.float64)[overlap_mask])
    right_signs = np.sign(np.asarray(right_values, dtype=np.float64)[overlap_mask])
    return float(np.mean(left_signs == right_signs))


def deg_restricted_lfc_spearman(
    left_logfc: np.ndarray,
    right_logfc: np.ndarray,
    mask: np.ndarray,
) -> float:
    eval_mask = mask & np.isfinite(left_logfc) & np.isfinite(right_logfc)
    if int(eval_mask.sum()) < 2:
        return float("nan")
    return vector_spearman_similarity(
        np.asarray(left_logfc, dtype=np.float64)[eval_mask],
        np.asarray(right_logfc, dtype=np.float64)[eval_mask],
    )


def compute_observed_deg_metrics_for_pair(
    gene_keys: np.ndarray,
    left_logfc: np.ndarray,
    right_logfc: np.ndarray,
    left_adj_p: np.ndarray,
    right_adj_p: np.ndarray,
    definition_key: str,
) -> dict[str, float]:
    left_mask = deg_mask(left_logfc, left_adj_p, definition_key)
    right_mask = deg_mask(right_logfc, right_adj_p, definition_key)
    ranked_left = ranked_gene_keys_from_mask(gene_keys, left_logfc, left_mask)
    ranked_right = ranked_gene_keys_from_mask(gene_keys, right_logfc, right_mask)

    left_ref_spearman = deg_restricted_lfc_spearman(left_logfc, right_logfc, left_mask)
    right_ref_spearman = deg_restricted_lfc_spearman(left_logfc, right_logfc, right_mask)
    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())

    result = {
        "deg_lfc_spearman_sym": strict_mean_if_all_defined(
            np.asarray([left_ref_spearman, right_ref_spearman], dtype=np.float64)
        ),
        "de_overlap_refn_sym": strict_mean_if_all_defined(
            np.asarray(
                [
                    top_overlap_fraction(ranked_left, ranked_right, n_left),
                    top_overlap_fraction(ranked_left, ranked_right, n_right),
                ],
                dtype=np.float64,
            )
        ),
        "direction_agreement": direction_agreement_with_masks(
            left_logfc,
            right_logfc,
            left_mask,
            right_mask,
        ),
    }
    for k in DE_OVERLAP_K_VALUES:
        result[f"de_overlap_k{k}"] = top_overlap_fraction(ranked_left, ranked_right, k)
    return result


def compute_sample_baseline_deg_metrics(
    gene_keys: np.ndarray,
    sample_logfc: np.ndarray,
    sample_adj_p: np.ndarray,
    baseline_logfc: Optional[np.ndarray],
    definition_key: str,
) -> dict[str, float]:
    if baseline_logfc is None:
        return empty_deg_metric_dict()

    sample_mask = deg_mask(sample_logfc, sample_adj_p, definition_key)
    ranked_sample = ranked_gene_keys_from_mask(gene_keys, sample_logfc, sample_mask)
    ranked_baseline = ranked_gene_keys_by_abs_values(gene_keys, baseline_logfc)
    n_sample = int(sample_mask.sum())

    result = {
        "deg_lfc_spearman_sym": deg_restricted_lfc_spearman(
            sample_logfc,
            baseline_logfc,
            sample_mask,
        ),
        "de_overlap_refn_sym": top_overlap_fraction(
            ranked_sample,
            ranked_baseline,
            n_sample,
        ),
        "direction_agreement": direction_agreement_with_masks(
            sample_logfc,
            baseline_logfc,
            sample_mask,
            np.isfinite(baseline_logfc),
        ),
    }
    for k in DE_OVERLAP_K_VALUES:
        result[f"de_overlap_k{k}"] = top_overlap_fraction(
            ranked_sample,
            ranked_baseline,
            k,
        )
    return result


def pairwise_replicate_baseline_values(
    matrix: np.ndarray,
    baseline_vector: np.ndarray,
    *,
    pair_indices: list[tuple[int, int]],
    scorer,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    baseline_vector = np.asarray(baseline_vector, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not pair_indices:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    replicate_baseline_values: list[float] = []
    baseline_value = scorer(baseline_vector, baseline_vector)
    baseline_values: list[float] = []
    for left_idx, right_idx in pair_indices:
        replicate_baseline_values.append(
            mean_available(
                np.asarray(
                    [
                        scorer(matrix[left_idx], baseline_vector),
                        scorer(matrix[right_idx], baseline_vector),
                    ],
                    dtype=np.float64,
                )
            )
        )
        baseline_values.append(float(baseline_value))
    return (
        np.asarray(replicate_baseline_values, dtype=np.float64),
        np.asarray(baseline_values, dtype=np.float64),
    )


def build_peer_matrix(
    vectors: list[Optional[np.ndarray]],
    *,
    finite_mask: Optional[np.ndarray],
    n_expected_columns: int,
    seed_key: str,
    n_total_peers_override: Optional[int] = None,
) -> tuple[Optional[np.ndarray], int, int]:
    """Stack the peer vectors that `build_baseline_vector` would have averaged.

    Returns the matrix aligned to the evaluation gene set together with the total and
    scored peer counts, or `(None, n_total, 0)` when it cannot be aligned. Rows are
    subsampled by `select_peer_indices` when `MAX_BASELINE_PEERS` is set.
    """
    present_vectors = [
        np.asarray(vector, dtype=np.float64) for vector in vectors if vector is not None
    ]
    n_loaded_peers = int(len(present_vectors))
    if n_loaded_peers == 0:
        return None, int(n_total_peers_override or 0), 0

    # When the caller already applied the cap before loading, `n_total_peers_override`
    # carries the true peer-set size so the recorded counts stay honest; subsampling again
    # here would be a no-op at best and a different draw at worst.
    if n_total_peers_override is None:
        n_total_peers = n_loaded_peers
        selected = select_peer_indices(
            n_total_peers,
            MAX_BASELINE_PEERS,
            seed_key,
            sampling_seed=PEER_SAMPLING_SEED,
        )
    else:
        n_total_peers = int(n_total_peers_override)
        selected = np.arange(n_loaded_peers, dtype=np.int64)
    matrix = np.vstack([present_vectors[int(index)] for index in selected]).astype(np.float64)
    if matrix.shape[1] == int(n_expected_columns):
        return matrix, n_total_peers, int(matrix.shape[0])
    if finite_mask is not None and matrix.shape[1] == int(finite_mask.shape[0]):
        reduced = matrix[:, finite_mask]
        if reduced.shape[1] == int(n_expected_columns):
            return reduced, n_total_peers, int(reduced.shape[0])
    return None, n_total_peers, 0


def replicate_pair_peer_summary(
    left_peer_scores: np.ndarray,
    right_peer_scores: np.ndarray,
    observed_value: float,
    prefix: str,
) -> dict[str, float | int]:
    """Summarize one replicate pair's per-peer baseline distribution.

    Each peer contributes the mean of its score against the two replicates, matching how
    `pairwise_replicate_baseline_values` averages the two centroid scores.
    """
    stacked = np.vstack(
        [
            np.asarray(left_peer_scores, dtype=np.float64),
            np.asarray(right_peer_scores, dtype=np.float64),
        ]
    )
    # Mean over whichever sides are defined, matching mean_available in the centroid path.
    finite = np.isfinite(stacked)
    counts = finite.sum(axis=0)
    sums = np.where(finite, stacked, 0.0).sum(axis=0)
    pair_peer_scores = np.full(stacked.shape[1], np.nan, dtype=np.float64)
    valid = counts > 0
    pair_peer_scores[valid] = sums[valid] / counts[valid]
    return summarize_peer_scores(observed_value, pair_peer_scores, prefix)


def build_sampled_replicate_domain_rows(context_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    query_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    for condition_key, block in context_frame.groupby("condition_key", sort=False):
        block = normalize_source_metadata_frame(block)
        block = block.sort_values(["source_path", "source_row_pos"]).reset_index(drop=True)
        pair_indices = sampled_replicate_pair_indices(len(block))
        for pair_idx, (query_idx, target_idx) in enumerate(pair_indices):
            query_row = block.iloc[query_idx].to_dict()
            target_row = block.iloc[target_idx].to_dict()
            pair_label = f"{condition_key}::pair{pair_idx}"
            query_row["sampled_pair_id"] = pair_label
            target_row["sampled_pair_id"] = pair_label
            query_rows.append(query_row)
            target_rows.append(target_row)
    if not query_rows:
        empty = context_frame.iloc[0:0].copy()
        empty["sampled_pair_id"] = pd.Series(dtype=object)
        return empty, empty.copy()
    return pd.DataFrame(query_rows).reset_index(drop=True), pd.DataFrame(target_rows).reset_index(drop=True)


def compute_retrieval_condition_task_summary(
    *,
    dataset_name: str,
    task_conditions: pd.DataFrame,
    dataset_metadata_frame: pd.DataFrame,
    line_global_shared_gene_keys: dict[str, np.ndarray],
    min_retrieval_compounds_per_line_time: int,
    open_adatas: dict[str, ad.AnnData],
) -> pd.DataFrame:
    task_conditions = task_conditions.copy()
    dataset_metadata_frame = normalize_source_metadata_frame(dataset_metadata_frame)
    task_condition_keys = set(task_conditions["condition_key"].astype(str).tolist())
    task_contexts = (
        task_conditions[["cell_type", "time_key"]]
        .drop_duplicates()
        .sort_values(["cell_type", "time_key"])
        .itertuples(index=False)
    )
    condition_records: list[dict[str, object]] = []

    for context in task_contexts:
        cell_type = str(context.cell_type)
        time_key = str(context.time_key)
        context_frame = dataset_metadata_frame.loc[
            (dataset_metadata_frame["cell_type"].astype(str) == cell_type)
            & (dataset_metadata_frame["time_key"].astype(str) == time_key)
        ].copy()
        if context_frame.empty:
            continue
        if int(context_frame["pubchem_cid"].astype(str).nunique()) < int(min_retrieval_compounds_per_line_time):
            continue

        query_rows, target_rows = build_sampled_replicate_domain_rows(context_frame)
        if query_rows.empty or target_rows.empty:
            continue
        query_rows = normalize_source_metadata_frame(query_rows)
        target_rows = normalize_source_metadata_frame(target_rows)

        query_rows_for_task = query_rows.loc[query_rows["condition_key"].astype(str).isin(task_condition_keys)].copy()
        if query_rows_for_task.empty:
            continue
        if int(target_rows["pubchem_cid"].astype(str).nunique()) < int(min_retrieval_compounds_per_line_time):
            continue

        source_paths = sorted(
            set(query_rows["source_path"].astype(str).tolist()) | set(target_rows["source_path"].astype(str).tolist())
        )
        local_gene_keys = shared_gene_keys_for_paths(source_paths)
        local_gene_keys = np.asarray(local_gene_keys, dtype=object)
        if local_gene_keys.size < 2:
            continue

        query_logfc_vectors, _ = load_vectors_for_rows(
            query_rows,
            gene_keys=local_gene_keys,
            open_adatas=open_adatas,
        )
        target_logfc_vectors, _ = load_vectors_for_rows(
            target_rows,
            gene_keys=local_gene_keys,
            open_adatas=open_adatas,
        )

        if not all(vector is not None for vector in query_logfc_vectors):
            continue
        if not all(vector is not None for vector in target_logfc_vectors):
            continue

        query_logfc = np.vstack(query_logfc_vectors).astype(np.float64)
        target_logfc = np.vstack(target_logfc_vectors).astype(np.float64)

        finite_mask = (
            np.isfinite(query_logfc).all(axis=0)
            & np.isfinite(target_logfc).all(axis=0)
        )
        if int(finite_mask.sum()) < 2:
            continue
        query_logfc = query_logfc[:, finite_mask]
        target_logfc = target_logfc[:, finite_mask]
        representation_matrices = {
            "logFC": (query_logfc, target_logfc),
        }
        target_condition_keys = target_rows["condition_key"].astype(str).to_numpy(dtype=object)

        query_row_indices = query_rows_for_task.index.to_numpy(dtype=np.int64)
        query_condition_keys = query_rows["condition_key"].astype(str).to_numpy(dtype=object)
        query_dose_keys = query_rows["dose_key"].astype(str).to_numpy(dtype=object)
        query_compounds = query_rows["pubchem_cid"].astype(str).to_numpy(dtype=object)
        query_perturbagen_names = query_rows.get("perturbagen_name", pd.Series("", index=query_rows.index)).astype(str).to_numpy(dtype=object)
        query_perturbagens = query_rows.get("perturbagen", pd.Series("", index=query_rows.index)).astype(str).to_numpy(dtype=object)
        query_perturbation_labels = query_rows.get("perturbation_label", pd.Series("", index=query_rows.index)).astype(str).to_numpy(dtype=object)

        for representation_name, (query_matrix, target_matrix) in representation_matrices.items():
            similarity = negative_l2_similarity_matrix(query_matrix, target_matrix)
            per_query_records: list[dict[str, object]] = []
            for query_idx in query_row_indices:
                positive_mask = target_condition_keys == str(query_condition_keys[query_idx])
                if int(positive_mask.sum()) == 0 or len(target_condition_keys) < 2:
                    continue
                _, observed_normalized_rank = normalized_best_positive_rank(similarity[query_idx], positive_mask)

                same_dose_other_mask = (
                    (query_dose_keys == str(query_dose_keys[query_idx]))
                    & (query_compounds != str(query_compounds[query_idx]))
                )
                baseline_normalized_rank = float("nan")
                if int(same_dose_other_mask.sum()) > 0:
                    baseline_candidate = np.nanmean(query_matrix[same_dose_other_mask], axis=0)
                    if np.isfinite(baseline_candidate).all():
                        baseline_similarity = float(
                            negative_l2_similarity_matrix(
                                query_matrix[query_idx][np.newaxis, :],
                                baseline_candidate[np.newaxis, :],
                            )[0, 0]
                        )
                        augmented_scores = np.concatenate(
                            [similarity[query_idx], np.asarray([baseline_similarity], dtype=np.float64)]
                        )
                        baseline_positive_mask = np.zeros(len(augmented_scores), dtype=bool)
                        baseline_positive_mask[-1] = True
                        _, baseline_normalized_rank = normalized_best_positive_rank(
                            augmented_scores,
                            baseline_positive_mask,
                        )

                per_query_records.append(
                    {
                        "dataset_name": dataset_name,
                        "cell_type": cell_type,
                        "time_key": time_key,
                        "representation": representation_name,
                        "condition_key": str(query_condition_keys[query_idx]),
                        "pubchem_cid": str(query_compounds[query_idx]),
                        "dose_key": str(query_dose_keys[query_idx]),
                        "perturbagen_display": first_non_empty(
                            pd.Series([query_perturbagen_names[query_idx]]),
                            pd.Series([query_perturbagens[query_idx]]),
                            pd.Series([query_perturbation_labels[query_idx]]),
                            query_compounds[query_idx],
                        ),
                        "n_target_conditions": int(target_matrix.shape[0]),
                        "n_unique_compounds": int(context_frame["pubchem_cid"].astype(str).nunique()),
                        "observed_normalized_best_positive_rank": observed_normalized_rank,
                        "baseline_normalized_best_positive_rank": baseline_normalized_rank,
                        "delta_vs_baseline_normalized_best_positive_rank": difference_if_both_defined(
                            observed_normalized_rank,
                            baseline_normalized_rank,
                        ),
                    }
                )

            if not per_query_records:
                continue
            per_query_frame = pd.DataFrame(per_query_records)
            per_condition_frame = (
                per_query_frame.groupby(
                    [
                        "dataset_name",
                        "cell_type",
                        "time_key",
                        "representation",
                        "condition_key",
                        "pubchem_cid",
                        "dose_key",
                        "perturbagen_display",
                    ],
                    as_index=False,
                )
                .agg(
                    n_queries=("condition_key", "size"),
                    n_queries_with_baseline=(
                        "baseline_normalized_best_positive_rank",
                        lambda values: int(pd.Series(values).notna().sum()),
                    ),
                    n_target_conditions=("n_target_conditions", "max"),
                    n_unique_compounds=("n_unique_compounds", "max"),
                    mean_observed_normalized_best_positive_rank=("observed_normalized_best_positive_rank", "mean"),
                    mean_baseline_normalized_best_positive_rank=("baseline_normalized_best_positive_rank", "mean"),
                    mean_delta_vs_baseline_normalized_best_positive_rank=(
                        "delta_vs_baseline_normalized_best_positive_rank",
                        "mean",
                    ),
                )
            )
            condition_records.extend(per_condition_frame.to_dict(orient="records"))

    if not condition_records:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "time_key",
                "representation",
                "condition_key",
                "pubchem_cid",
                "dose_key",
                "perturbagen_display",
                "n_queries",
                "n_queries_with_baseline",
                "n_target_conditions",
                "n_unique_compounds",
                "mean_observed_normalized_best_positive_rank",
                "mean_baseline_normalized_best_positive_rank",
                "mean_delta_vs_baseline_normalized_best_positive_rank",
            ]
        )
    return pd.DataFrame(condition_records).sort_values(
        ["dataset_name", "cell_type", "time_key", "representation", "condition_key"]
    ).reset_index(drop=True)


def resolve_dataset_file_paths(dataset_dir: Path) -> list[Path]:
    paths = sorted(dataset_dir.glob("*.h5ad"))
    if "novartis_batch_1000" not in str(dataset_dir):
        return paths

    selected_by_cell_type: dict[str, Path] = {}
    for path in paths:
        hint = filename_cell_type_hint(path)
        cell_type = hint
        if "_de" in cell_type:
            cell_type = cell_type.replace("_de", "")
        if "_" in cell_type:
            maybe_cell_type, suffix = cell_type.rsplit("_", 1)
            try:
                float(suffix)
            except ValueError:
                pass
            else:
                cell_type = maybe_cell_type
        if cell_type not in selected_by_cell_type:
            selected_by_cell_type[cell_type] = path
    return sorted(selected_by_cell_type.values())


def filename_cell_type_hint(path: Path) -> str:
    name = path.name
    if name.endswith("_de.h5ad"):
        return name[:-8]
    if name.endswith(".h5ad"):
        return name[:-5]
    return name


def path_matches_allowed_cell_types(path: Path, allowed_cell_types: set[str]) -> bool:
    hint = filename_cell_type_hint(path)
    if hint in allowed_cell_types:
        return True
    return any(hint.startswith(f"{cell_type}_") for cell_type in allowed_cell_types)


def resolve_processed_sep_rep_h5ad(dataset_name: str, data_root: Path = DEFAULT_PROCESSED_DATA_ROOT) -> Path:
    sep_rep_dir = data_root / dataset_name / "pseudobulk_processed" / "sep_rep"
    matches = sorted(sep_rep_dir.glob("*.h5ad"))
    if not matches:
        raise FileNotFoundError(f"No processed sep_rep .h5ad file found in {sep_rep_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Expected one processed sep_rep .h5ad file in {sep_rep_dir}, found {len(matches)}")
    return matches[0]


def load_processed_sep_rep_metadata(dataset_name: str, data_root: Path = DEFAULT_PROCESSED_DATA_ROOT) -> pd.DataFrame:
    path = resolve_processed_sep_rep_h5ad(dataset_name, data_root=data_root)
    adata = read_h5ad_safely(path, backed="r")
    try:
        available_columns = [column_name for column_name in METADATA_OBS_COLUMNS if column_name in adata.obs.columns]
        obs = adata.obs[available_columns].copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    for column_name in [
        "cell_type",
        "pubchem_cid",
        "perturbagen",
        "perturbagen_name",
        "perturbation_label",
    ]:
        if column_name not in obs.columns:
            obs[column_name] = ""
        obs[column_name] = obs[column_name].astype("string").fillna("").astype(str).str.strip()
    obs["pubchem_cid"] = obs["pubchem_cid"].map(normalize_pubchem_cid_value)

    obs["pert_time_h"] = pd.to_numeric(obs.get("pert_time_h", np.nan), errors="coerce")
    obs["pert_dose_uM"] = pd.to_numeric(obs.get("pert_dose_uM", np.nan), errors="coerce")
    if "is_control" in obs.columns:
        control_mask = coerce_control_mask(obs["is_control"]).to_numpy(dtype=bool)
    else:
        control_mask = np.zeros(len(obs), dtype=bool)

    valid_mask = (
        ~control_mask
        & (obs["cell_type"] != "").to_numpy(dtype=bool)
        & (obs["pubchem_cid"] != "").to_numpy(dtype=bool)
        & np.isfinite(obs["pert_time_h"].to_numpy(dtype=float))
        & np.isfinite(obs["pert_dose_uM"].to_numpy(dtype=float))
        & (obs["pert_dose_uM"].to_numpy(dtype=float) > 0.0)
    )
    metadata = obs.loc[valid_mask, ["cell_type", "pubchem_cid", "pert_time_h", "pert_dose_uM"]].copy()
    metadata["dataset_name"] = dataset_name
    metadata["time_key"] = metadata["pert_time_h"].map(format_numeric)
    metadata["dose_key"] = metadata["pert_dose_uM"].map(format_numeric)
    metadata["condition_key"] = (
        metadata["cell_type"].astype(str)
        + "|"
        + metadata["pubchem_cid"].astype(str)
        + "|"
        + metadata["time_key"].astype(str)
        + "|"
        + metadata["dose_key"].astype(str)
    )
    return metadata.drop_duplicates(
        subset=["dataset_name", "cell_type", "pubchem_cid", "time_key", "dose_key", "condition_key"]
    ).reset_index(drop=True)


def load_source_metadata(path: Path) -> pd.DataFrame:
    adata = read_h5ad_safely(path, backed="r")
    try:
        available_columns = [column_name for column_name in METADATA_OBS_COLUMNS if column_name in adata.obs.columns]
        obs = adata.obs[available_columns].copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    obs["source_row_pos"] = np.arange(len(obs), dtype=np.int64)
    obs["source_path"] = str(path)
    obs["source_file_name"] = path.name

    for column_name in [
        "id",
        "cell_type",
        "perturbagen",
        "perturbagen_name",
        "perturbation_label",
        "pubchem_cid",
    ]:
        if column_name not in obs.columns:
            obs[column_name] = ""
        obs[column_name] = obs[column_name].astype("string").fillna("").astype(str).str.strip()
    obs["pubchem_cid"] = obs["pubchem_cid"].map(normalize_pubchem_cid_value)

    obs["pert_time_h"] = pd.to_numeric(obs.get("pert_time_h", np.nan), errors="coerce")
    obs["pert_dose_uM"] = pd.to_numeric(obs.get("pert_dose_uM", np.nan), errors="coerce")
    if "is_control" in obs.columns:
        control_mask = coerce_control_mask(obs["is_control"]).to_numpy(dtype=bool)
    else:
        control_mask = np.zeros(len(obs), dtype=bool)
    valid_mask = (
        ~control_mask
        & (obs["cell_type"] != "").to_numpy(dtype=bool)
        & (obs["pubchem_cid"] != "").to_numpy(dtype=bool)
        & np.isfinite(obs["pert_time_h"].to_numpy(dtype=float))
        & np.isfinite(obs["pert_dose_uM"].to_numpy(dtype=float))
        & (obs["pert_dose_uM"].to_numpy(dtype=float) > 0.0)
    )
    obs = obs.loc[
        valid_mask,
        [
            "source_row_pos",
            "source_path",
            "source_file_name",
            "id",
            "cell_type",
            "pubchem_cid",
            "perturbagen",
            "perturbagen_name",
            "perturbation_label",
            "pert_time_h",
            "pert_dose_uM",
        ],
    ].copy()
    obs["time_key"] = obs["pert_time_h"].map(format_numeric)
    obs["dose_key"] = obs["pert_dose_uM"].map(format_numeric)
    obs["condition_key"] = (
        obs["cell_type"].astype(str)
        + "|"
        + obs["pubchem_cid"].astype(str)
        + "|"
        + obs["time_key"].astype(str)
        + "|"
        + obs["dose_key"].astype(str)
    )
    obs["replicate_id"] = obs["source_path"].astype(str) + "::" + obs["source_row_pos"].astype(str)
    return obs.reset_index(drop=True)


def load_gene_info(path: str | Path) -> GeneInfo:
    path_str = str(Path(path))
    if path_str in GENE_INFO_CACHE:
        return GENE_INFO_CACHE[path_str]

    adata = read_h5ad_safely(path_str, backed="r")
    try:
        var = adata.var.copy()
    finally:
        if getattr(adata, "file", None) is not None:
            adata.file.close()

    if "symbol" in var.columns:
        gene_key_series = pd.Series(
            var["symbol"].astype("string").fillna("").astype(str).str.strip().to_numpy(),
            index=np.arange(len(var), dtype=np.int64),
        )
    else:
        gene_key_series = pd.Series(
            pd.Index(var.index.astype(str)).astype(str).str.strip().to_numpy(),
            index=np.arange(len(var), dtype=np.int64),
        )

    keep_mask = (gene_key_series != "") & ~gene_key_series.duplicated(keep="first")
    unique_gene_positions = gene_key_series.index[keep_mask].to_numpy(dtype=np.int64)
    unique_gene_keys = gene_key_series.loc[keep_mask].to_numpy(dtype=object)
    gene_info = GeneInfo(
        path=Path(path_str),
        unique_gene_keys=unique_gene_keys,
        unique_gene_positions=unique_gene_positions,
        gene_to_var_pos={
            str(gene_key): int(var_pos)
            for gene_key, var_pos in zip(unique_gene_keys.tolist(), unique_gene_positions.tolist())
        },
    )
    GENE_INFO_CACHE[path_str] = gene_info
    return gene_info


def build_dataset_index(
    dataset_name: str,
    dataset_dir: Path,
    *,
    min_replicates_per_condition: int,
    allowed_cell_types: Optional[set[str]] = None,
    allowed_line_compounds: Optional[set[tuple[str, str]]] = None,
    allowed_condition_keys: Optional[set[str]] = None,
) -> dict[str, object]:
    frames: list[pd.DataFrame] = []
    line_to_files: dict[str, set[str]] = defaultdict(set)
    all_file_paths = resolve_dataset_file_paths(dataset_dir)

    if allowed_cell_types is not None:
        all_file_paths = [path for path in all_file_paths if path_matches_allowed_cell_types(path, allowed_cell_types)]

    n_scanned_files = len(all_file_paths)
    n_kept_files = 0

    for path in all_file_paths:
        frame = load_source_metadata(path)
        if frame.empty:
            continue
        if allowed_cell_types is not None:
            frame = frame.loc[frame["cell_type"].isin(allowed_cell_types)].copy()
        if allowed_line_compounds is not None and not frame.empty:
            keep_mask = [
                (cell_type, pubchem_cid) in allowed_line_compounds
                for cell_type, pubchem_cid in zip(frame["cell_type"].astype(str), frame["pubchem_cid"].astype(str))
            ]
            frame = frame.loc[keep_mask].copy()
        if allowed_condition_keys is not None and not frame.empty:
            frame = frame.loc[frame["condition_key"].isin(allowed_condition_keys)].copy()
        if frame.empty:
            continue
        n_kept_files += 1
        frame["dataset_name"] = dataset_name
        frames.append(frame)
        for cell_type in frame["cell_type"].dropna().astype(str).unique().tolist():
            if cell_type:
                line_to_files[cell_type].add(str(path))

    if not frames:
        empty_columns = [
            "dataset_name",
            "source_path",
            "source_file_name",
            "source_row_pos",
            "replicate_id",
            "condition_key",
            "id",
            "cell_type",
            "pubchem_cid",
            "perturbagen",
            "perturbagen_name",
            "perturbation_label",
            "pert_time_h",
            "pert_dose_uM",
            "time_key",
            "dose_key",
        ]
        return {
            "frame": pd.DataFrame(columns=empty_columns),
            "condition_groups": {},
            "condition_rows_by_key": {},
            "line_to_files": {},
            "n_scanned_files": n_scanned_files,
            "n_kept_files": 0,
        }

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.sort_values(
        ["cell_type", "pubchem_cid", "pert_time_h", "pert_dose_uM", "source_path", "source_row_pos"]
    ).reset_index(drop=True)
    condition_groups = frame.groupby(["cell_type", "pubchem_cid", "time_key", "dose_key"], sort=False).groups
    counts = frame.groupby(["cell_type", "pubchem_cid", "time_key", "dose_key"]).size()
    qualifying_keys = {
        tuple(key)
        for key, count in counts.items()
        if int(count) >= int(min_replicates_per_condition)
    }
    qualifying_groups = {
        key: row_indexes
        for key, row_indexes in condition_groups.items()
        if tuple(key) in qualifying_keys
    }
    condition_rows_by_key = {
        f"{cell_type}|{pubchem_cid}|{time_key}|{dose_key}": list(map(int, row_indexes))
        for (cell_type, pubchem_cid, time_key, dose_key), row_indexes in qualifying_groups.items()
    }
    return {
        "frame": frame,
        "condition_groups": qualifying_groups,
        "condition_rows_by_key": condition_rows_by_key,
        "line_to_files": {cell_type: sorted(paths) for cell_type, paths in line_to_files.items()},
        "n_scanned_files": n_scanned_files,
        "n_kept_files": n_kept_files,
    }


def build_dataset_index_with_fallback(
    dataset_name: str,
    dataset_dir: Path,
    *,
    min_replicates_per_condition: int,
    allowed_cell_types: Optional[set[str]] = None,
    allowed_line_compounds: Optional[set[tuple[str, str]]] = None,
    allowed_condition_keys: Optional[set[str]] = None,
    expect_nonempty: bool = False,
) -> dict[str, object]:
    index = build_dataset_index(
        dataset_name,
        dataset_dir,
        min_replicates_per_condition=min_replicates_per_condition,
        allowed_cell_types=allowed_cell_types,
        allowed_line_compounds=allowed_line_compounds,
        allowed_condition_keys=allowed_condition_keys,
    )
    index["filter_attempt"] = "strict_filtered"
    index["fallback_used"] = False

    if expect_nonempty and index["frame"].empty:
        candidate_line_count = 0 if allowed_cell_types is None else len(allowed_cell_types)
        candidate_compound_count = 0 if allowed_line_compounds is None else len(allowed_line_compounds)
        candidate_condition_count = 0 if allowed_condition_keys is None else len(allowed_condition_keys)
        raise ValueError(
            "Strict filtered sep_rep scan returned no rows for "
            f"{dataset_name}. "
            f"dataset_dir={dataset_dir} "
            f"candidate_lines={candidate_line_count} "
            f"candidate_line_compounds={candidate_compound_count} "
            f"candidate_condition_keys={candidate_condition_count}"
        )

    return index


def build_condition_inventory(dataset_indices: dict[str, dict[str, object]], active_datasets: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name in active_datasets:
        frame = dataset_indices[dataset_name]["frame"]
        condition_groups = dataset_indices[dataset_name]["condition_groups"]
        for (cell_type, pubchem_cid, time_key, dose_key), row_indexes in condition_groups.items():
            row_indexes = list(row_indexes)
            n_replicates = int(len(row_indexes))
            subset = frame.iloc[row_indexes]
            source_paths = sorted(set(subset["source_path"].astype(str).tolist()))
            records.append(
                {
                    "dataset_name": dataset_name,
                    "cell_type": str(cell_type),
                    "pubchem_cid": str(pubchem_cid),
                    "time_key": str(time_key),
                    "dose_key": str(dose_key),
                    "perturbagen_display": first_non_empty(
                        subset.get("perturbagen_name", pd.Series(dtype=object)),
                        subset.get("perturbagen", pd.Series(dtype=object)),
                        subset.get("perturbation_label", pd.Series(dtype=object)),
                        pubchem_cid,
                    ),
                    "n_replicates": n_replicates,
                    "n_replicate_pairs": int(len(sampled_replicate_pair_indices(n_replicates))),
                    "n_total_possible_replicate_pairs": int(n_replicates * (n_replicates - 1) // 2),
                    "n_source_files": int(len(source_paths)),
                    "primary_source_path": source_paths[0],
                    "source_path_key": "||".join(source_paths),
                    "condition_key": f"{cell_type}|{pubchem_cid}|{time_key}|{dose_key}",
                }
            )
    if not records:
        return pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "pubchem_cid",
                "time_key",
                "dose_key",
                "perturbagen_display",
                "n_replicates",
                "n_replicate_pairs",
                "n_total_possible_replicate_pairs",
                "n_source_files",
                "primary_source_path",
                "source_path_key",
                "condition_key",
            ]
        )
    return pd.DataFrame(records).sort_values(
        ["dataset_name", "cell_type", "primary_source_path", "pubchem_cid", "time_key", "dose_key"]
    ).reset_index(drop=True)


def dataset_line_gene_keys(dataset_name: str, cell_type: str, dataset_indices: dict[str, dict[str, object]]) -> np.ndarray:
    cache_key = (str(dataset_name), str(cell_type))
    if cache_key in DATASET_LINE_GENE_KEY_CACHE:
        return DATASET_LINE_GENE_KEY_CACHE[cache_key]
    file_paths = dataset_indices[dataset_name]["line_to_files"].get(cell_type, [])
    if not file_paths:
        DATASET_LINE_GENE_KEY_CACHE[cache_key] = np.asarray([], dtype=object)
        return DATASET_LINE_GENE_KEY_CACHE[cache_key]
    shared_gene_keys: Optional[set[str]] = None
    for path_str in file_paths:
        gene_info = load_gene_info(path_str)
        gene_keys = set(map(str, gene_info.unique_gene_keys.tolist()))
        shared_gene_keys = gene_keys if shared_gene_keys is None else (shared_gene_keys & gene_keys)
    DATASET_LINE_GENE_KEY_CACHE[cache_key] = np.asarray(sorted(shared_gene_keys or set()), dtype=object)
    return DATASET_LINE_GENE_KEY_CACHE[cache_key]


def set_line_global_shared_gene_keys(
    retained_lines: dict[str, list[str]],
    active_datasets: list[str],
    dataset_indices: dict[str, dict[str, object]],
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for cell_type in sorted({cell_type for lines in retained_lines.values() for cell_type in lines}):
        datasets_with_line = [
            dataset_name for dataset_name in active_datasets if cell_type in retained_lines.get(dataset_name, [])
        ]
        shared_gene_keys: Optional[set[str]] = None
        for dataset_name in datasets_with_line:
            dataset_gene_keys = set(map(str, dataset_line_gene_keys(dataset_name, cell_type, dataset_indices).tolist()))
            shared_gene_keys = dataset_gene_keys if shared_gene_keys is None else (shared_gene_keys & dataset_gene_keys)
        result[cell_type] = np.asarray(sorted(shared_gene_keys or set()), dtype=object)
    return result


def shared_gene_keys_for_paths(source_paths: list[str]) -> np.ndarray:
    key = tuple(sorted(set(map(str, source_paths))))
    if key in SHARED_GENE_KEY_CACHE:
        return SHARED_GENE_KEY_CACHE[key]
    if not key:
        SHARED_GENE_KEY_CACHE[key] = np.asarray([], dtype=object)
        return SHARED_GENE_KEY_CACHE[key]
    shared_gene_keys: Optional[set[str]] = None
    for path_str in key:
        gene_info = load_gene_info(path_str)
        gene_keys = set(map(str, gene_info.unique_gene_keys.tolist()))
        shared_gene_keys = gene_keys if shared_gene_keys is None else (shared_gene_keys & gene_keys)
    SHARED_GENE_KEY_CACHE[key] = np.asarray(sorted(shared_gene_keys or set()), dtype=object)
    return SHARED_GENE_KEY_CACHE[key]


def gene_positions_for_source(source_path: str, gene_keys: np.ndarray) -> np.ndarray:
    gene_key_tuple = tuple(map(str, np.asarray(gene_keys, dtype=object).tolist()))
    cache_key = (str(source_path), gene_key_tuple)
    if cache_key in GENE_POSITION_CACHE:
        return GENE_POSITION_CACHE[cache_key]
    gene_info = load_gene_info(source_path)
    positions = [gene_info.gene_to_var_pos[gene_key] for gene_key in gene_key_tuple if gene_key in gene_info.gene_to_var_pos]
    GENE_POSITION_CACHE[cache_key] = np.asarray(positions, dtype=np.int64)
    return GENE_POSITION_CACHE[cache_key]


def task_input_dir(output_dir: Path) -> Path:
    return output_dir / TASK_INPUT_DIR_NAME


def task_output_dir(output_dir: Path) -> Path:
    return output_dir / TASK_OUTPUT_DIR_NAME


def task_manifest_path(output_dir: Path) -> Path:
    return output_dir / TASK_MANIFEST_FILE_NAME


def line_global_gene_keys_path(output_dir: Path) -> Path:
    return task_input_dir(output_dir) / LINE_GLOBAL_SHARED_GENE_KEYS_FILE_NAME


def task_config_path(output_dir: Path) -> Path:
    return task_input_dir(output_dir) / TASK_CONFIG_FILE_NAME


def dataset_metadata_cache_dir(output_dir: Path) -> Path:
    return task_input_dir(output_dir) / DATASET_METADATA_CACHE_DIR_NAME


def dataset_metadata_cache_path(output_dir: Path, dataset_name: str) -> Path:
    return dataset_metadata_cache_dir(output_dir) / f"{dataset_name}_eligible_rows.tsv"


def task_conditions_path(output_dir: Path, task_id: int) -> Path:
    return task_input_dir(output_dir) / f"task_{task_id:06d}_conditions.tsv"


def task_replicates_path(output_dir: Path, task_id: int) -> Path:
    return task_input_dir(output_dir) / f"task_{task_id:06d}_replicates.tsv"


def task_metrics_path(base_task_output_dir: Path, task_id: int) -> Path:
    return base_task_output_dir / f"task_{task_id:06d}_condition_metric_summary.tsv"


def task_errors_path(base_task_output_dir: Path, task_id: int) -> Path:
    return base_task_output_dir / f"task_{task_id:06d}_condition_scoring_errors.tsv"


def task_condition_retrieval_path(base_task_output_dir: Path, task_id: int) -> Path:
    return base_task_output_dir / f"task_{task_id:06d}_condition_retrieval_summary.tsv"


def normalize_source_metadata_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    string_columns = [
        "dataset_name",
        "source_path",
        "source_file_name",
        "id",
        "cell_type",
        "pubchem_cid",
        "perturbagen",
        "perturbagen_name",
        "perturbation_label",
        "time_key",
        "dose_key",
        "condition_key",
        "replicate_id",
    ]
    for column_name in string_columns:
        if column_name in normalized.columns:
            normalized[column_name] = normalized[column_name].astype("string").fillna("").astype(str)
    if "source_row_pos" in normalized.columns:
        normalized["source_row_pos"] = pd.to_numeric(normalized["source_row_pos"], errors="coerce").astype(np.int64)
    if "pert_time_h" in normalized.columns:
        normalized["pert_time_h"] = pd.to_numeric(normalized["pert_time_h"], errors="coerce")
    if "pert_dose_uM" in normalized.columns:
        normalized["pert_dose_uM"] = pd.to_numeric(normalized["pert_dose_uM"], errors="coerce")
    return normalized


def load_task_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Task manifest not found: {path}")
    manifest = pd.read_csv(path, sep="\t", keep_default_na=False)
    if manifest.empty:
        return manifest
    int_columns = ["task_id", "n_conditions", "n_replicate_rows", "n_source_files"]
    for column_name in int_columns:
        if column_name in manifest.columns:
            manifest[column_name] = pd.to_numeric(manifest[column_name], errors="coerce").astype(np.int64)
    return manifest


def write_task_config(path: Path, config: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def load_task_config(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Task config not found: {path}")
    return json.loads(path.read_text())


def config_fingerprint(config: dict[str, object]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def write_tsv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a complete TSV before atomically publishing it at the final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        frame.to_csv(temporary_path, sep="\t", index=False)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_line_global_gene_keys(path: Path, mapping: dict[str, np.ndarray]) -> None:
    records = [
        {
            "cell_type": str(cell_type),
            "gene_keys_json": json.dumps(list(map(str, np.asarray(gene_keys, dtype=object).tolist()))),
            "n_gene_keys": int(np.asarray(gene_keys, dtype=object).size),
        }
        for cell_type, gene_keys in sorted(mapping.items())
    ]
    pd.DataFrame(records).to_csv(path, sep="\t", index=False)


def load_line_global_gene_keys(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Line-global gene key file not found: {path}")
    frame = pd.read_csv(path, sep="\t", keep_default_na=False)
    result: dict[str, np.ndarray] = {}
    for _, row in frame.iterrows():
        cell_type = str(row["cell_type"])
        gene_keys = json.loads(str(row["gene_keys_json"])) if str(row["gene_keys_json"]) else []
        result[cell_type] = np.asarray(gene_keys, dtype=object)
    return result


def compute_condition_metric_record_from_rows(
    condition_row: pd.Series,
    condition_rows: pd.DataFrame,
    *,
    line_global_shared_gene_keys: dict[str, np.ndarray],
    top_k: int,
    compute_deg_metrics: bool = False,
    compute_baseline_metrics: bool = False,
    baseline_source_frame: Optional[pd.DataFrame] = None,
    baseline_context_row_indexes: Optional[dict[tuple[str, str, str], np.ndarray]] = None,
    baseline_context_vector_cache: Optional[
        dict[tuple[object, ...], BaselineContextVectors]
    ] = None,
    open_adatas: Optional[dict[str, ad.AnnData]] = None,
) -> Optional[dict[str, object]]:
    condition_rows = normalize_source_metadata_frame(condition_rows)
    dataset_name = str(condition_row["dataset_name"])
    cell_type = str(condition_row["cell_type"])
    pubchem_cid = str(condition_row["pubchem_cid"])
    time_key = str(condition_row["time_key"])
    dose_key = str(condition_row["dose_key"])
    n_replicates = int(len(condition_rows))
    pair_indices = sampled_replicate_pair_indices(n_replicates)

    if n_replicates < 2 or not pair_indices:
        return None

    source_paths = sorted(set(condition_rows["source_path"].astype(str).tolist()))
    local_gene_keys = shared_gene_keys_for_paths(source_paths)
    global_gene_keys = line_global_shared_gene_keys.get(cell_type, np.asarray([], dtype=object))
    if open_adatas is None:
        open_adatas = {}
    if baseline_context_vector_cache is None:
        baseline_context_vector_cache = {}

    indexed_rows = condition_rows.reset_index(drop=True)
    local_logfc_vectors, local_t_vectors = load_vectors_for_rows(
        indexed_rows,
        gene_keys=local_gene_keys,
        open_adatas=open_adatas,
    )
    global_logfc_vectors, global_t_vectors = load_vectors_for_rows(
        indexed_rows,
        gene_keys=global_gene_keys,
        open_adatas=open_adatas,
    )

    record = {
        "dataset_name": dataset_name,
        "cell_type": cell_type,
        "pubchem_cid": pubchem_cid,
        "time_key": time_key,
        "dose_key": dose_key,
        "condition_key": str(condition_row["condition_key"]),
        "perturbagen_display": str(condition_row["perturbagen_display"]),
        "n_replicates": n_replicates,
        "n_replicate_pairs": int(len(pair_indices)),
        "n_total_possible_replicate_pairs": int(n_replicates * (n_replicates - 1) // 2),
        "n_source_files": int(indexed_rows["source_path"].nunique()),
        "n_local_shared_genes": 0,
        "n_global_shared_genes": 0,
        "mean_abs_t": float("nan"),
        "mean_abs_t_global": float("nan"),
        "n_baseline_peer_rows": 0,
        "n_baseline_peer_compounds": 0,
        "n_baseline_peer_rows_loaded": 0,
        "n_peer_rows_total": 0,
        "n_peer_rows_available": 0,
        "n_peer_rows_scored": 0,
        "mean_replicate_spearman_logfc": float("nan"),
        "median_replicate_spearman_logfc": float("nan"),
        "mean_replicate_spearman_t": float("nan"),
        "median_replicate_spearman_t": float("nan"),
        f"mean_replicate_signed_overlap_t_top{top_k}": float("nan"),
        f"median_replicate_signed_overlap_t_top{top_k}": float("nan"),
        "mean_replicate_spearman_logfc_global": float("nan"),
        "median_replicate_spearman_logfc_global": float("nan"),
        "mean_replicate_spearman_t_global": float("nan"),
        "median_replicate_spearman_t_global": float("nan"),
        f"mean_replicate_signed_overlap_t_top{top_k}_global": float("nan"),
        f"median_replicate_signed_overlap_t_top{top_k}_global": float("nan"),
        "mean_baseline_spearman_logfc": float("nan"),
        "median_baseline_spearman_logfc": float("nan"),
        "mean_baseline_spearman_t": float("nan"),
        "median_baseline_spearman_t": float("nan"),
        f"mean_baseline_signed_overlap_t_top{top_k}": float("nan"),
        f"median_baseline_signed_overlap_t_top{top_k}": float("nan"),
        "mean_baseline_spearman_logfc_global": float("nan"),
        "median_baseline_spearman_logfc_global": float("nan"),
        "mean_baseline_spearman_t_global": float("nan"),
        "median_baseline_spearman_t_global": float("nan"),
        f"mean_baseline_signed_overlap_t_top{top_k}_global": float("nan"),
        f"median_baseline_signed_overlap_t_top{top_k}_global": float("nan"),
        "mean_replicate_baseline_spearman_logfc": float("nan"),
        "median_replicate_baseline_spearman_logfc": float("nan"),
        "mean_replicate_baseline_spearman_t": float("nan"),
        "median_replicate_baseline_spearman_t": float("nan"),
        f"mean_replicate_baseline_signed_overlap_t_top{top_k}": float("nan"),
        f"median_replicate_baseline_signed_overlap_t_top{top_k}": float("nan"),
        "mean_replicate_baseline_spearman_logfc_global": float("nan"),
        "median_replicate_baseline_spearman_logfc_global": float("nan"),
        "mean_replicate_baseline_spearman_t_global": float("nan"),
        "median_replicate_baseline_spearman_t_global": float("nan"),
        f"mean_replicate_baseline_signed_overlap_t_top{top_k}_global": float("nan"),
        f"median_replicate_baseline_signed_overlap_t_top{top_k}_global": float("nan"),
        "mean_replicate_minus_baseline_spearman_logfc": float("nan"),
        "median_replicate_minus_baseline_spearman_logfc": float("nan"),
        "mean_replicate_minus_baseline_spearman_t": float("nan"),
        "median_replicate_minus_baseline_spearman_t": float("nan"),
        f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}": float("nan"),
        f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}": float("nan"),
        "mean_replicate_minus_baseline_spearman_logfc_global": float("nan"),
        "median_replicate_minus_baseline_spearman_logfc_global": float("nan"),
        "mean_replicate_minus_baseline_spearman_t_global": float("nan"),
        "median_replicate_minus_baseline_spearman_t_global": float("nan"),
        f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}_global": float("nan"),
        f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}_global": float("nan"),
        "n_valid_logfc_pairs": 0,
        "n_valid_t_pairs": 0,
        f"n_valid_signed_overlap_t_top{top_k}_pairs": 0,
        "n_valid_logfc_pairs_global": 0,
        "n_valid_t_pairs_global": 0,
        f"n_valid_signed_overlap_t_top{top_k}_pairs_global": 0,
        "n_valid_baseline_logfc_pairs": 0,
        "n_valid_baseline_t_pairs": 0,
        f"n_valid_baseline_signed_overlap_t_top{top_k}_pairs": 0,
        "n_valid_baseline_logfc_pairs_global": 0,
        "n_valid_baseline_t_pairs_global": 0,
        f"n_valid_baseline_signed_overlap_t_top{top_k}_pairs_global": 0,
        "n_valid_replicate_baseline_logfc_pairs": 0,
        "n_valid_replicate_baseline_t_pairs": 0,
        f"n_valid_replicate_baseline_signed_overlap_t_top{top_k}_pairs": 0,
        "n_valid_replicate_baseline_logfc_pairs_global": 0,
        "n_valid_replicate_baseline_t_pairs_global": 0,
        f"n_valid_replicate_baseline_signed_overlap_t_top{top_k}_pairs_global": 0,
        "n_valid_replicate_minus_baseline_logfc_pairs": 0,
        "n_valid_replicate_minus_baseline_t_pairs": 0,
        f"n_valid_replicate_minus_baseline_signed_overlap_t_top{top_k}_pairs": 0,
        "n_valid_replicate_minus_baseline_logfc_pairs_global": 0,
        "n_valid_replicate_minus_baseline_t_pairs_global": 0,
        f"n_valid_replicate_minus_baseline_signed_overlap_t_top{top_k}_pairs_global": 0,
    }
    if compute_deg_metrics:
        for definition_key in DEG_DEFINITION_CONFIG:
            for metric_name in deg_metric_names():
                for prefix in [
                    "mean_replicate",
                    "median_replicate",
                    "mean_baseline_pair",
                    "median_baseline_pair",
                    "mean_delta_vs_baseline_pair",
                    "median_delta_vs_baseline_pair",
                ]:
                    record[f"{prefix}_{metric_name}_{definition_key}"] = float("nan")

    local_logfc_matrix: Optional[np.ndarray] = None
    local_t_matrix: Optional[np.ndarray] = None
    local_adj_p_matrix: Optional[np.ndarray] = None
    local_gene_keys_eval = np.asarray([], dtype=object)
    logfc_values = np.asarray([], dtype=np.float64)
    t_values = np.asarray([], dtype=np.float64)
    overlap_values = np.asarray([], dtype=np.float64)
    local_adj_p_vectors: list[Optional[np.ndarray]] = []
    if compute_deg_metrics:
        local_adj_p_vectors = load_adjusted_pvalue_vectors_for_rows(
            indexed_rows,
            gene_keys=local_gene_keys,
            open_adatas=open_adatas,
        )
    if all(vector is not None for vector in local_logfc_vectors) and all(vector is not None for vector in local_t_vectors):
        local_logfc_matrix = np.vstack(local_logfc_vectors).astype(np.float64)
        local_t_matrix = np.vstack(local_t_vectors).astype(np.float64)
        finite_local_mask = np.isfinite(local_logfc_matrix).all(axis=0) & np.isfinite(local_t_matrix).all(axis=0)
        if compute_deg_metrics and local_adj_p_vectors and all(vector is not None for vector in local_adj_p_vectors):
            local_adj_p_matrix = np.vstack(local_adj_p_vectors).astype(np.float64)
            finite_local_mask &= np.isfinite(local_adj_p_matrix).all(axis=0)
        else:
            local_adj_p_matrix = None
        local_gene_keys_eval = np.asarray(local_gene_keys, dtype=object)[finite_local_mask]
        local_logfc_matrix = local_logfc_matrix[:, finite_local_mask]
        local_t_matrix = local_t_matrix[:, finite_local_mask]
        if local_adj_p_matrix is not None:
            local_adj_p_matrix = local_adj_p_matrix[:, finite_local_mask]
        if local_logfc_matrix.shape[1] >= 2:
            record["n_local_shared_genes"] = int(local_logfc_matrix.shape[1])
            record["mean_abs_t"] = float(np.nanmean(np.abs(local_t_matrix)))
            logfc_similarity = row_spearman_similarity_matrix(local_logfc_matrix)
            t_similarity = row_spearman_similarity_matrix(local_t_matrix)
            logfc_values = pair_values_from_similarity_matrix(logfc_similarity, pair_indices)
            t_values = pair_values_from_similarity_matrix(t_similarity, pair_indices)
            overlap_values = np.asarray(
                [
                    signed_overlap_at_k(local_t_matrix[i], local_t_matrix[j], k=top_k)
                    for i, j in pair_indices
                ],
                dtype=np.float64,
            )
            record["mean_replicate_spearman_logfc"] = mean_available(logfc_values)
            record["median_replicate_spearman_logfc"] = median_available(logfc_values)
            record["mean_replicate_spearman_t"] = mean_available(t_values)
            record["median_replicate_spearman_t"] = median_available(t_values)
            record[f"mean_replicate_signed_overlap_t_top{top_k}"] = mean_available(overlap_values)
            record[f"median_replicate_signed_overlap_t_top{top_k}"] = median_available(overlap_values)
            record["n_valid_logfc_pairs"] = int(np.isfinite(logfc_values).sum())
            record["n_valid_t_pairs"] = int(np.isfinite(t_values).sum())
            record[f"n_valid_signed_overlap_t_top{top_k}_pairs"] = int(np.isfinite(overlap_values).sum())

    global_logfc_matrix: Optional[np.ndarray] = None
    global_t_matrix: Optional[np.ndarray] = None
    logfc_values_global = np.asarray([], dtype=np.float64)
    t_values_global = np.asarray([], dtype=np.float64)
    overlap_values_global = np.asarray([], dtype=np.float64)
    if all(vector is not None for vector in global_logfc_vectors) and all(vector is not None for vector in global_t_vectors):
        global_logfc_matrix = np.vstack(global_logfc_vectors).astype(np.float64)
        global_t_matrix = np.vstack(global_t_vectors).astype(np.float64)
        finite_global_mask = np.isfinite(global_logfc_matrix).all(axis=0) & np.isfinite(global_t_matrix).all(axis=0)
        global_logfc_matrix = global_logfc_matrix[:, finite_global_mask]
        global_t_matrix = global_t_matrix[:, finite_global_mask]
        if global_logfc_matrix.shape[1] >= 2:
            record["n_global_shared_genes"] = int(global_logfc_matrix.shape[1])
            record["mean_abs_t_global"] = float(np.nanmean(np.abs(global_t_matrix)))
            logfc_similarity_global = row_spearman_similarity_matrix(global_logfc_matrix)
            t_similarity_global = row_spearman_similarity_matrix(global_t_matrix)
            logfc_values_global = pair_values_from_similarity_matrix(logfc_similarity_global, pair_indices)
            t_values_global = pair_values_from_similarity_matrix(t_similarity_global, pair_indices)
            overlap_values_global = np.asarray(
                [
                    signed_overlap_at_k(global_t_matrix[i], global_t_matrix[j], k=top_k)
                    for i, j in pair_indices
                ],
                dtype=np.float64,
            )
            record["mean_replicate_spearman_logfc_global"] = mean_available(logfc_values_global)
            record["median_replicate_spearman_logfc_global"] = median_available(logfc_values_global)
            record["mean_replicate_spearman_t_global"] = mean_available(t_values_global)
            record["median_replicate_spearman_t_global"] = median_available(t_values_global)
            record[f"mean_replicate_signed_overlap_t_top{top_k}_global"] = mean_available(overlap_values_global)
            record[f"median_replicate_signed_overlap_t_top{top_k}_global"] = median_available(overlap_values_global)
            record["n_valid_logfc_pairs_global"] = int(np.isfinite(logfc_values_global).sum())
            record["n_valid_t_pairs_global"] = int(np.isfinite(t_values_global).sum())
            record[f"n_valid_signed_overlap_t_top{top_k}_pairs_global"] = int(np.isfinite(overlap_values_global).sum())

    local_baseline_logfc: Optional[np.ndarray] = None
    global_baseline_logfc: Optional[np.ndarray] = None
    local_peer_logfc_matrix: Optional[np.ndarray] = None
    local_peer_logfc_prepared = None
    peer_seed_key = "|".join([dataset_name, cell_type, pubchem_cid, time_key, dose_key])

    if compute_baseline_metrics and baseline_source_frame is not None and baseline_context_row_indexes is not None:
        context_key = (cell_type, time_key, dose_key)
        context_indexes = baseline_context_row_indexes.get(context_key)
        if context_indexes is not None and len(context_indexes) > 0:
            raw_context_frame = baseline_source_frame.iloc[
                np.asarray(context_indexes, dtype=np.int64)
            ].copy()
            context_vectors = get_baseline_context_vectors(
                context_rows=raw_context_frame,
                context_key=context_key,
                local_gene_keys=local_gene_keys,
                global_gene_keys=global_gene_keys,
                open_adatas=open_adatas,
                cache=baseline_context_vector_cache,
            )
            peer_row_mask = (
                context_vectors.rows["pubchem_cid"].astype(str).to_numpy() != pubchem_cid
            )
            if peer_row_mask.any():
                same_compound_mask = ~peer_row_mask
                baseline_rows = context_vectors.rows.loc[peer_row_mask].reset_index(drop=True)
                record["n_baseline_peer_rows"] = int(peer_row_mask.sum())
                record["n_baseline_peer_compounds"] = int(
                    baseline_rows["pubchem_cid"].astype(str).nunique()
                )
                # The published centroid remains exact and therefore always uses every
                # eligible peer row. The cap applies only to the single-peer distribution.
                local_baseline_logfc = exact_mean_excluding_row_mask(
                    context_vectors.local_logfc,
                    same_compound_mask,
                    total_sums=context_vectors.finite_totals["local_logfc"][0],
                    total_counts=context_vectors.finite_totals["local_logfc"][1],
                )
                local_baseline_t = exact_mean_excluding_row_mask(
                    context_vectors.local_t,
                    same_compound_mask,
                    total_sums=context_vectors.finite_totals["local_t"][0],
                    total_counts=context_vectors.finite_totals["local_t"][1],
                )
                global_baseline_logfc = exact_mean_excluding_row_mask(
                    context_vectors.global_logfc,
                    same_compound_mask,
                    total_sums=context_vectors.finite_totals["global_logfc"][0],
                    total_counts=context_vectors.finite_totals["global_logfc"][1],
                )
                global_baseline_t = exact_mean_excluding_row_mask(
                    context_vectors.global_t,
                    same_compound_mask,
                    total_sums=context_vectors.finite_totals["global_t"][0],
                    total_counts=context_vectors.finite_totals["global_t"][1],
                )
                record["n_baseline_peer_rows_loaded"] = int(peer_row_mask.sum())

                if local_baseline_logfc is not None and local_logfc_matrix is not None:
                    if local_adj_p_matrix is not None:
                        if local_baseline_logfc.shape[0] == finite_local_mask.shape[0]:
                            local_baseline_logfc = local_baseline_logfc[finite_local_mask]
                        else:
                            local_baseline_logfc = None
                    elif local_baseline_logfc.shape[0] != local_logfc_matrix.shape[1]:
                        local_baseline_logfc = None
                if local_baseline_t is not None and local_t_matrix is not None:
                    if local_adj_p_matrix is not None:
                        if local_baseline_t.shape[0] == finite_local_mask.shape[0]:
                            local_baseline_t = local_baseline_t[finite_local_mask]
                        else:
                            local_baseline_t = None
                    elif local_baseline_t.shape[0] != local_t_matrix.shape[1]:
                        local_baseline_t = None
                if global_baseline_logfc is not None and global_logfc_matrix is not None:
                    if global_baseline_logfc.shape[0] == finite_global_mask.shape[0]:
                        global_baseline_logfc = global_baseline_logfc[finite_global_mask]
                    elif global_baseline_logfc.shape[0] != global_logfc_matrix.shape[1]:
                        global_baseline_logfc = None
                if global_baseline_t is not None and global_t_matrix is not None:
                    if global_baseline_t.shape[0] == finite_global_mask.shape[0]:
                        global_baseline_t = global_baseline_t[finite_global_mask]
                    elif global_baseline_t.shape[0] != global_t_matrix.shape[1]:
                        global_baseline_t = None

                if (
                    local_logfc_matrix is not None
                    and local_t_matrix is not None
                    and local_baseline_logfc is not None
                    and local_baseline_t is not None
                ):
                    local_replicate_baseline_logfc, local_baseline_logfc_values = pairwise_replicate_baseline_values(
                        local_logfc_matrix,
                        local_baseline_logfc,
                        pair_indices=pair_indices,
                        scorer=vector_spearman_similarity,
                    )
                    local_replicate_baseline_t, local_baseline_t_values = pairwise_replicate_baseline_values(
                        local_t_matrix,
                        local_baseline_t,
                        pair_indices=pair_indices,
                        scorer=vector_spearman_similarity,
                    )
                    local_replicate_baseline_overlap, local_baseline_overlap_values = pairwise_replicate_baseline_values(
                        local_t_matrix,
                        local_baseline_t,
                        pair_indices=pair_indices,
                        scorer=lambda left, right: signed_overlap_at_k(left, right, k=top_k),
                    )
                    record["mean_baseline_spearman_logfc"] = mean_available(local_baseline_logfc_values)
                    record["median_baseline_spearman_logfc"] = median_available(local_baseline_logfc_values)
                    record["mean_baseline_spearman_t"] = mean_available(local_baseline_t_values)
                    record["median_baseline_spearman_t"] = median_available(local_baseline_t_values)
                    record[f"mean_baseline_signed_overlap_t_top{top_k}"] = mean_available(local_baseline_overlap_values)
                    record[f"median_baseline_signed_overlap_t_top{top_k}"] = median_available(local_baseline_overlap_values)
                    record["mean_replicate_baseline_spearman_logfc"] = mean_available(local_replicate_baseline_logfc)
                    record["median_replicate_baseline_spearman_logfc"] = median_available(local_replicate_baseline_logfc)
                    record["mean_replicate_baseline_spearman_t"] = mean_available(local_replicate_baseline_t)
                    record["median_replicate_baseline_spearman_t"] = median_available(local_replicate_baseline_t)
                    record[f"mean_replicate_baseline_signed_overlap_t_top{top_k}"] = mean_available(local_replicate_baseline_overlap)
                    record[f"median_replicate_baseline_signed_overlap_t_top{top_k}"] = median_available(local_replicate_baseline_overlap)
                    replicate_minus_baseline_logfc = logfc_values - local_replicate_baseline_logfc
                    replicate_minus_baseline_t = t_values - local_replicate_baseline_t
                    replicate_minus_baseline_overlap = overlap_values - local_replicate_baseline_overlap
                    record["mean_replicate_minus_baseline_spearman_logfc"] = mean_available(replicate_minus_baseline_logfc)
                    record["median_replicate_minus_baseline_spearman_logfc"] = median_available(replicate_minus_baseline_logfc)
                    record["mean_replicate_minus_baseline_spearman_t"] = mean_available(replicate_minus_baseline_t)
                    record["median_replicate_minus_baseline_spearman_t"] = median_available(replicate_minus_baseline_t)
                    record[f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}"] = mean_available(replicate_minus_baseline_overlap)
                    record[f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}"] = median_available(replicate_minus_baseline_overlap)
                    record["n_valid_baseline_logfc_pairs"] = int(np.isfinite(local_baseline_logfc_values).sum())
                    record["n_valid_baseline_t_pairs"] = int(np.isfinite(local_baseline_t_values).sum())
                    record[f"n_valid_baseline_signed_overlap_t_top{top_k}_pairs"] = int(np.isfinite(local_baseline_overlap_values).sum())
                    record["n_valid_replicate_baseline_logfc_pairs"] = int(np.isfinite(local_replicate_baseline_logfc).sum())
                    record["n_valid_replicate_baseline_t_pairs"] = int(np.isfinite(local_replicate_baseline_t).sum())
                    record[f"n_valid_replicate_baseline_signed_overlap_t_top{top_k}_pairs"] = int(np.isfinite(local_replicate_baseline_overlap).sum())
                    record["n_valid_replicate_minus_baseline_logfc_pairs"] = int(np.isfinite(replicate_minus_baseline_logfc).sum())
                    record["n_valid_replicate_minus_baseline_t_pairs"] = int(np.isfinite(replicate_minus_baseline_t).sum())
                    record[f"n_valid_replicate_minus_baseline_signed_overlap_t_top{top_k}_pairs"] = int(np.isfinite(replicate_minus_baseline_overlap).sum())

                # Per-peer baseline for the all-gene logFC Spearman summary: score each
                # same line / time / dose other-drug peer separately instead of averaging
                # the peers into a centroid first.
                if local_logfc_matrix is not None:
                    peer_context_positions = np.flatnonzero(
                        peer_row_mask
                        & np.isfinite(context_vectors.local_logfc).any(axis=1)
                    )
                    selected_peer_positions = select_peer_indices(
                        int(peer_context_positions.size),
                        MAX_BASELINE_PEERS,
                        peer_seed_key,
                        sampling_seed=PEER_SAMPLING_SEED,
                    )
                    scored_context_positions = peer_context_positions[
                        selected_peer_positions
                    ]
                    local_peer_logfc_matrix = context_vectors.local_logfc[
                        scored_context_positions
                    ]
                    if local_adj_p_matrix is not None:
                        local_peer_logfc_matrix = local_peer_logfc_matrix[
                            :, finite_local_mask
                        ]
                    elif local_peer_logfc_matrix.shape[1] != local_logfc_matrix.shape[1]:
                        local_peer_logfc_matrix = None
                    record["n_peer_rows_total"] = int(peer_row_mask.sum())
                    record["n_peer_rows_available"] = int(peer_context_positions.size)
                    record["n_peer_rows_scored"] = (
                        int(local_peer_logfc_matrix.shape[0])
                        if local_peer_logfc_matrix is not None
                        else 0
                    )
                    record["peer_sampling_seed"] = int(PEER_SAMPLING_SEED)
                    if local_peer_logfc_matrix is not None and local_peer_logfc_matrix.shape[0] > 0:
                        local_peer_logfc_prepared = get_prepared_local_context_spearman(
                            context_vectors,
                            finite_local_mask
                            if local_adj_p_matrix is not None
                            else None,
                        )
                        peer_summary_fields: dict[str, list[float]] = defaultdict(list)
                        peer_delta_values: list[float] = []
                        for pair_position, (left_idx, right_idx) in enumerate(pair_indices):
                            left_peer_scores = spearman_against_prepared_peers(
                                local_logfc_matrix[left_idx],
                                local_peer_logfc_prepared,
                                row_indices=scored_context_positions,
                            )
                            right_peer_scores = spearman_against_prepared_peers(
                                local_logfc_matrix[right_idx],
                                local_peer_logfc_prepared,
                                row_indices=scored_context_positions,
                            )
                            observed_value = (
                                float(logfc_values[pair_position])
                                if pair_position < len(logfc_values)
                                else float("nan")
                            )
                            pair_peer_summary = replicate_pair_peer_summary(
                                left_peer_scores,
                                right_peer_scores,
                                observed_value,
                                "peer",
                            )
                            for field_name, field_value in pair_peer_summary.items():
                                peer_summary_fields[field_name].append(float(field_value))
                            peer_delta_values.append(
                                difference_if_both_defined(
                                    observed_value,
                                    float(pair_peer_summary["peer_mean_score"]),
                                )
                            )
                        record["mean_peer_baseline_spearman_logfc"] = mean_available(
                            np.asarray(peer_summary_fields["peer_mean_score"], dtype=np.float64)
                        )
                        record["mean_peer_baseline_sd_spearman_logfc"] = mean_available(
                            np.asarray(peer_summary_fields["peer_sd_score"], dtype=np.float64)
                        )
                        record["mean_peer_baseline_fraction_below_observed_spearman_logfc"] = mean_available(
                            np.asarray(
                                peer_summary_fields["peer_fraction_below_observed"],
                                dtype=np.float64,
                            )
                        )
                        record["mean_peer_baseline_corrected_percentile_spearman_logfc"] = mean_available(
                            np.asarray(
                                peer_summary_fields["peer_corrected_percentile"],
                                dtype=np.float64,
                            )
                        )
                        record["mean_replicate_minus_peer_baseline_spearman_logfc"] = mean_available(
                            np.asarray(peer_delta_values, dtype=np.float64)
                        )
                        record["n_valid_peer_baseline_logfc_pairs"] = int(
                            np.isfinite(
                                np.asarray(peer_summary_fields["peer_mean_score"], dtype=np.float64)
                            ).sum()
                        )

                if (
                    global_logfc_matrix is not None
                    and global_t_matrix is not None
                    and global_baseline_logfc is not None
                    and global_baseline_t is not None
                ):
                    global_replicate_baseline_logfc, global_baseline_logfc_values = pairwise_replicate_baseline_values(
                        global_logfc_matrix,
                        global_baseline_logfc,
                        pair_indices=pair_indices,
                        scorer=vector_spearman_similarity,
                    )
                    global_replicate_baseline_t, global_baseline_t_values = pairwise_replicate_baseline_values(
                        global_t_matrix,
                        global_baseline_t,
                        pair_indices=pair_indices,
                        scorer=vector_spearman_similarity,
                    )
                    global_replicate_baseline_overlap, global_baseline_overlap_values = pairwise_replicate_baseline_values(
                        global_t_matrix,
                        global_baseline_t,
                        pair_indices=pair_indices,
                        scorer=lambda left, right: signed_overlap_at_k(left, right, k=top_k),
                    )
                    record["mean_baseline_spearman_logfc_global"] = mean_available(global_baseline_logfc_values)
                    record["median_baseline_spearman_logfc_global"] = median_available(global_baseline_logfc_values)
                    record["mean_baseline_spearman_t_global"] = mean_available(global_baseline_t_values)
                    record["median_baseline_spearman_t_global"] = median_available(global_baseline_t_values)
                    record[f"mean_baseline_signed_overlap_t_top{top_k}_global"] = mean_available(global_baseline_overlap_values)
                    record[f"median_baseline_signed_overlap_t_top{top_k}_global"] = median_available(global_baseline_overlap_values)
                    record["mean_replicate_baseline_spearman_logfc_global"] = mean_available(global_replicate_baseline_logfc)
                    record["median_replicate_baseline_spearman_logfc_global"] = median_available(global_replicate_baseline_logfc)
                    record["mean_replicate_baseline_spearman_t_global"] = mean_available(global_replicate_baseline_t)
                    record["median_replicate_baseline_spearman_t_global"] = median_available(global_replicate_baseline_t)
                    record[f"mean_replicate_baseline_signed_overlap_t_top{top_k}_global"] = mean_available(global_replicate_baseline_overlap)
                    record[f"median_replicate_baseline_signed_overlap_t_top{top_k}_global"] = median_available(global_replicate_baseline_overlap)
                    replicate_minus_baseline_logfc_global = logfc_values_global - global_replicate_baseline_logfc
                    replicate_minus_baseline_t_global = t_values_global - global_replicate_baseline_t
                    replicate_minus_baseline_overlap_global = overlap_values_global - global_replicate_baseline_overlap
                    record["mean_replicate_minus_baseline_spearman_logfc_global"] = mean_available(replicate_minus_baseline_logfc_global)
                    record["median_replicate_minus_baseline_spearman_logfc_global"] = median_available(replicate_minus_baseline_logfc_global)
                    record["mean_replicate_minus_baseline_spearman_t_global"] = mean_available(replicate_minus_baseline_t_global)
                    record["median_replicate_minus_baseline_spearman_t_global"] = median_available(replicate_minus_baseline_t_global)
                    record[f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}_global"] = mean_available(replicate_minus_baseline_overlap_global)
                    record[f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}_global"] = median_available(replicate_minus_baseline_overlap_global)
                    record["n_valid_baseline_logfc_pairs_global"] = int(np.isfinite(global_baseline_logfc_values).sum())
                    record["n_valid_baseline_t_pairs_global"] = int(np.isfinite(global_baseline_t_values).sum())
                    record[f"n_valid_baseline_signed_overlap_t_top{top_k}_pairs_global"] = int(np.isfinite(global_baseline_overlap_values).sum())
                    record["n_valid_replicate_baseline_logfc_pairs_global"] = int(np.isfinite(global_replicate_baseline_logfc).sum())
                    record["n_valid_replicate_baseline_t_pairs_global"] = int(np.isfinite(global_replicate_baseline_t).sum())
                    record[f"n_valid_replicate_baseline_signed_overlap_t_top{top_k}_pairs_global"] = int(np.isfinite(global_replicate_baseline_overlap).sum())
                    record["n_valid_replicate_minus_baseline_logfc_pairs_global"] = int(np.isfinite(replicate_minus_baseline_logfc_global).sum())
                    record["n_valid_replicate_minus_baseline_t_pairs_global"] = int(np.isfinite(replicate_minus_baseline_t_global).sum())
                    record[f"n_valid_replicate_minus_baseline_signed_overlap_t_top{top_k}_pairs_global"] = int(np.isfinite(replicate_minus_baseline_overlap_global).sum())

    if (
        compute_deg_metrics
        and local_logfc_matrix is not None
        and local_adj_p_matrix is not None
        and local_logfc_matrix.shape[1] >= 2
        and local_adj_p_matrix.shape[1] >= 2
    ):
        for definition_key in DEG_DEFINITION_CONFIG:
            observed_values_by_metric = {
                metric_name: []
                for metric_name in deg_metric_names()
            }
            baseline_pair_values_by_metric = {
                metric_name: []
                for metric_name in deg_metric_names()
            }
            delta_values_by_metric = {
                metric_name: []
                for metric_name in deg_metric_names()
            }
            peer_values_by_metric: dict[str, dict[str, list[float]]] = {
                metric_name: defaultdict(list)
                for metric_name in PEER_BASELINE_DEG_METRICS
            }

            for left_idx, right_idx in pair_indices:
                observed_metrics = compute_observed_deg_metrics_for_pair(
                    local_gene_keys_eval,
                    local_logfc_matrix[left_idx],
                    local_logfc_matrix[right_idx],
                    local_adj_p_matrix[left_idx],
                    local_adj_p_matrix[right_idx],
                    definition_key,
                )
                for metric_name, metric_value in observed_metrics.items():
                    observed_values_by_metric[metric_name].append(metric_value)

                if local_baseline_logfc is not None:
                    left_baseline_metrics = compute_sample_baseline_deg_metrics(
                        local_gene_keys_eval,
                        local_logfc_matrix[left_idx],
                        local_adj_p_matrix[left_idx],
                        local_baseline_logfc,
                        definition_key,
                    )
                    right_baseline_metrics = compute_sample_baseline_deg_metrics(
                        local_gene_keys_eval,
                        local_logfc_matrix[right_idx],
                        local_adj_p_matrix[right_idx],
                        local_baseline_logfc,
                        definition_key,
                    )
                    for metric_name in deg_metric_names():
                        baseline_pair_value = mean_available(
                            np.asarray(
                                [
                                    left_baseline_metrics[metric_name],
                                    right_baseline_metrics[metric_name],
                                ],
                                dtype=np.float64,
                            )
                        )
                        baseline_pair_values_by_metric[metric_name].append(baseline_pair_value)
                        observed_value = observed_metrics[metric_name]
                        if np.isfinite(observed_value) and np.isfinite(baseline_pair_value):
                            delta_values_by_metric[metric_name].append(
                                float(observed_value - baseline_pair_value)
                            )
                        else:
                            delta_values_by_metric[metric_name].append(float("nan"))
                else:
                    for metric_name in deg_metric_names():
                        baseline_pair_values_by_metric[metric_name].append(float("nan"))
                        delta_values_by_metric[metric_name].append(float("nan"))

                # Per-peer DEG baselines, on the sample-referenced convention: each
                # replicate's own DEG mask defines the evaluation genes, exactly as the
                # centroid baseline does in compute_sample_baseline_deg_metrics.
                if local_peer_logfc_matrix is not None and local_peer_logfc_matrix.shape[0] > 0:
                    left_deg_mask = deg_mask(
                        local_logfc_matrix[left_idx],
                        local_adj_p_matrix[left_idx],
                        definition_key,
                    )
                    right_deg_mask = deg_mask(
                        local_logfc_matrix[right_idx],
                        local_adj_p_matrix[right_idx],
                        definition_key,
                    )
                    for metric_name in PEER_BASELINE_DEG_METRICS:
                        if metric_name == "deg_lfc_spearman_sym":
                            left_peer_scores = spearman_against_peers(
                                local_logfc_matrix[left_idx],
                                local_peer_logfc_matrix,
                                left_deg_mask,
                            )
                            right_peer_scores = spearman_against_peers(
                                local_logfc_matrix[right_idx],
                                local_peer_logfc_matrix,
                                right_deg_mask,
                            )
                        else:
                            left_peer_scores = direction_agreement_against_peers(
                                local_logfc_matrix[left_idx],
                                local_peer_logfc_matrix,
                                left_deg_mask,
                            )
                            right_peer_scores = direction_agreement_against_peers(
                                local_logfc_matrix[right_idx],
                                local_peer_logfc_matrix,
                                right_deg_mask,
                            )
                        observed_value = float(observed_metrics[metric_name])
                        pair_peer_summary = replicate_pair_peer_summary(
                            left_peer_scores,
                            right_peer_scores,
                            observed_value,
                            "peer",
                        )
                        for field_name, field_value in pair_peer_summary.items():
                            peer_values_by_metric[metric_name][field_name].append(float(field_value))
                        peer_values_by_metric[metric_name]["delta"].append(
                            difference_if_both_defined(
                                observed_value,
                                float(pair_peer_summary["peer_mean_score"]),
                            )
                        )

            for metric_name in PEER_BASELINE_DEG_METRICS:
                field_values = peer_values_by_metric[metric_name]
                if not field_values:
                    continue
                for record_suffix, field_name in (
                    ("", "peer_mean_score"),
                    ("_sd", "peer_sd_score"),
                    ("_fraction_below_observed", "peer_fraction_below_observed"),
                    ("_corrected_percentile", "peer_corrected_percentile"),
                ):
                    record[f"mean_peer_baseline_{metric_name}{record_suffix}_{definition_key}"] = mean_available(
                        np.asarray(field_values.get(field_name, []), dtype=np.float64)
                    )
                record[f"mean_delta_vs_peer_baseline_{metric_name}_{definition_key}"] = mean_available(
                    np.asarray(field_values.get("delta", []), dtype=np.float64)
                )

            for metric_name in deg_metric_names():
                observed_array = np.asarray(observed_values_by_metric[metric_name], dtype=np.float64)
                baseline_array = np.asarray(baseline_pair_values_by_metric[metric_name], dtype=np.float64)
                delta_array = np.asarray(delta_values_by_metric[metric_name], dtype=np.float64)
                record[f"mean_replicate_{metric_name}_{definition_key}"] = mean_available(observed_array)
                record[f"median_replicate_{metric_name}_{definition_key}"] = median_available(observed_array)
                record[f"mean_baseline_pair_{metric_name}_{definition_key}"] = mean_available(baseline_array)
                record[f"median_baseline_pair_{metric_name}_{definition_key}"] = median_available(baseline_array)
                record[f"mean_delta_vs_baseline_pair_{metric_name}_{definition_key}"] = mean_available(delta_array)
                record[f"median_delta_vs_baseline_pair_{metric_name}_{definition_key}"] = median_available(delta_array)

    if record["n_local_shared_genes"] < 2 and record["n_global_shared_genes"] < 2:
        return None
    return record


def summarize_condition_frame(frame: pd.DataFrame, top_k: int) -> pd.Series:
    record = {
        "n_conditions": int(len(frame)),
        "n_unique_lines": int(frame["cell_type"].nunique()) if "cell_type" in frame.columns else 1,
        "n_unique_compounds": int(frame["pubchem_cid"].nunique()),
        "n_total_replicates": int(frame["n_replicates"].sum()),
        "n_total_replicate_pairs": int(frame["n_replicate_pairs"].sum()),
        "mean_n_replicates": float(frame["n_replicates"].mean()),
        "mean_n_local_shared_genes": float(frame["n_local_shared_genes"].mean()),
        "mean_n_global_shared_genes": float(frame["n_global_shared_genes"].mean()),
        "mean_mean_abs_t": float(frame["mean_abs_t"].mean()) if "mean_abs_t" in frame.columns else float("nan"),
        "mean_mean_abs_t_global": float(frame["mean_abs_t_global"].mean()) if "mean_abs_t_global" in frame.columns else float("nan"),
        "mean_n_baseline_peer_rows": float(frame["n_baseline_peer_rows"].mean()) if "n_baseline_peer_rows" in frame.columns else float("nan"),
        "mean_n_baseline_peer_compounds": float(frame["n_baseline_peer_compounds"].mean()) if "n_baseline_peer_compounds" in frame.columns else float("nan"),
        "mean_n_peer_rows_total": float(frame["n_peer_rows_total"].mean()) if "n_peer_rows_total" in frame.columns else float("nan"),
        "mean_n_peer_rows_available": float(frame["n_peer_rows_available"].mean()) if "n_peer_rows_available" in frame.columns else float("nan"),
        "mean_n_peer_rows_scored": float(frame["n_peer_rows_scored"].mean()) if "n_peer_rows_scored" in frame.columns else float("nan"),
    }
    summary_mean_columns = [
        "mean_replicate_spearman_logfc",
        "median_replicate_spearman_logfc",
        "mean_replicate_spearman_t",
        "median_replicate_spearman_t",
        f"mean_replicate_signed_overlap_t_top{top_k}",
        f"median_replicate_signed_overlap_t_top{top_k}",
        "mean_replicate_spearman_logfc_global",
        "median_replicate_spearman_logfc_global",
        "mean_replicate_spearman_t_global",
        "median_replicate_spearman_t_global",
        f"mean_replicate_signed_overlap_t_top{top_k}_global",
        f"median_replicate_signed_overlap_t_top{top_k}_global",
        "mean_baseline_spearman_logfc",
        "median_baseline_spearman_logfc",
        "mean_baseline_spearman_t",
        "median_baseline_spearman_t",
        f"mean_baseline_signed_overlap_t_top{top_k}",
        f"median_baseline_signed_overlap_t_top{top_k}",
        "mean_baseline_spearman_logfc_global",
        "median_baseline_spearman_logfc_global",
        "mean_baseline_spearman_t_global",
        "median_baseline_spearman_t_global",
        f"mean_baseline_signed_overlap_t_top{top_k}_global",
        f"median_baseline_signed_overlap_t_top{top_k}_global",
        "mean_replicate_baseline_spearman_logfc",
        "median_replicate_baseline_spearman_logfc",
        "mean_replicate_baseline_spearman_t",
        "median_replicate_baseline_spearman_t",
        f"mean_replicate_baseline_signed_overlap_t_top{top_k}",
        f"median_replicate_baseline_signed_overlap_t_top{top_k}",
        "mean_replicate_baseline_spearman_logfc_global",
        "median_replicate_baseline_spearman_logfc_global",
        "mean_replicate_baseline_spearman_t_global",
        "median_replicate_baseline_spearman_t_global",
        f"mean_replicate_baseline_signed_overlap_t_top{top_k}_global",
        f"median_replicate_baseline_signed_overlap_t_top{top_k}_global",
        "mean_replicate_minus_baseline_spearman_logfc",
        "median_replicate_minus_baseline_spearman_logfc",
        "mean_replicate_minus_baseline_spearman_t",
        "median_replicate_minus_baseline_spearman_t",
        f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}",
        f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}",
        "mean_replicate_minus_baseline_spearman_logfc_global",
        "median_replicate_minus_baseline_spearman_logfc_global",
        "mean_replicate_minus_baseline_spearman_t_global",
        "median_replicate_minus_baseline_spearman_t_global",
        f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}_global",
        f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}_global",
    ]
    summary_mean_columns.extend(
        sorted(
            column_name
            for column_name in frame.columns
            if column_name.startswith(
                (
                    "mean_replicate_deg_",
                    "median_replicate_deg_",
                    "mean_baseline_pair_deg_",
                    "median_baseline_pair_deg_",
                    "mean_delta_vs_baseline_pair_deg_",
                    "median_delta_vs_baseline_pair_deg_",
                    # Per-peer baselines: all-gene logFC plus the DEG-restricted metrics.
                    "mean_peer_baseline_",
                    "mean_delta_vs_peer_baseline_",
                    "mean_replicate_minus_peer_baseline_",
                )
            )
        )
    )
    summary_mean_columns = list(dict.fromkeys(summary_mean_columns))
    for column_name in summary_mean_columns:
        record[column_name] = float(frame[column_name].mean()) if column_name in frame.columns else float("nan")
    return pd.Series(record)


def empty_summary_record(top_k: int) -> dict[str, object]:
    record = {
        "n_conditions": 0,
        "n_unique_lines": 0,
        "n_unique_compounds": 0,
        "n_total_replicates": 0,
        "n_total_replicate_pairs": 0,
        "mean_n_replicates": float("nan"),
        "mean_n_local_shared_genes": float("nan"),
        "mean_n_global_shared_genes": float("nan"),
        "mean_mean_abs_t": float("nan"),
        "mean_mean_abs_t_global": float("nan"),
        "mean_n_baseline_peer_rows": float("nan"),
        "mean_n_baseline_peer_compounds": float("nan"),
    }
    summary_mean_columns = [
        "mean_replicate_spearman_logfc",
        "median_replicate_spearman_logfc",
        "mean_replicate_spearman_t",
        "median_replicate_spearman_t",
        f"mean_replicate_signed_overlap_t_top{top_k}",
        f"median_replicate_signed_overlap_t_top{top_k}",
        "mean_replicate_spearman_logfc_global",
        "median_replicate_spearman_logfc_global",
        "mean_replicate_spearman_t_global",
        "median_replicate_spearman_t_global",
        f"mean_replicate_signed_overlap_t_top{top_k}_global",
        f"median_replicate_signed_overlap_t_top{top_k}_global",
        "mean_baseline_spearman_logfc",
        "median_baseline_spearman_logfc",
        "mean_baseline_spearman_t",
        "median_baseline_spearman_t",
        f"mean_baseline_signed_overlap_t_top{top_k}",
        f"median_baseline_signed_overlap_t_top{top_k}",
        "mean_baseline_spearman_logfc_global",
        "median_baseline_spearman_logfc_global",
        "mean_baseline_spearman_t_global",
        "median_baseline_spearman_t_global",
        f"mean_baseline_signed_overlap_t_top{top_k}_global",
        f"median_baseline_signed_overlap_t_top{top_k}_global",
        "mean_replicate_baseline_spearman_logfc",
        "median_replicate_baseline_spearman_logfc",
        "mean_replicate_baseline_spearman_t",
        "median_replicate_baseline_spearman_t",
        f"mean_replicate_baseline_signed_overlap_t_top{top_k}",
        f"median_replicate_baseline_signed_overlap_t_top{top_k}",
        "mean_replicate_baseline_spearman_logfc_global",
        "median_replicate_baseline_spearman_logfc_global",
        "mean_replicate_baseline_spearman_t_global",
        "median_replicate_baseline_spearman_t_global",
        f"mean_replicate_baseline_signed_overlap_t_top{top_k}_global",
        f"median_replicate_baseline_signed_overlap_t_top{top_k}_global",
        "mean_replicate_minus_baseline_spearman_logfc",
        "median_replicate_minus_baseline_spearman_logfc",
        "mean_replicate_minus_baseline_spearman_t",
        "median_replicate_minus_baseline_spearman_t",
        f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}",
        f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}",
        "mean_replicate_minus_baseline_spearman_logfc_global",
        "median_replicate_minus_baseline_spearman_logfc_global",
        "mean_replicate_minus_baseline_spearman_t_global",
        "median_replicate_minus_baseline_spearman_t_global",
        f"mean_replicate_minus_baseline_signed_overlap_t_top{top_k}_global",
        f"median_replicate_minus_baseline_signed_overlap_t_top{top_k}_global",
    ]
    for column_name in summary_mean_columns:
        record[column_name] = float("nan")
    return record


def select_test_lines_per_dataset(retained_conditions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if retained_conditions.empty:
        empty = pd.DataFrame(columns=["dataset_name", "cell_type", "n_conditions", "n_replicate_pairs", "n_replicates"])
        return retained_conditions.copy(), empty

    ranking = (
        retained_conditions.groupby(["dataset_name", "cell_type"], as_index=False)
        .agg(
            n_conditions=("condition_key", "size"),
            n_replicate_pairs=("n_replicate_pairs", "sum"),
            n_replicates=("n_replicates", "sum"),
        )
        .sort_values(
            ["dataset_name", "n_conditions", "n_replicate_pairs", "n_replicates", "cell_type"],
            ascending=[True, False, False, False, True],
        )
        .reset_index(drop=True)
    )
    selected_lines = ranking.groupby("dataset_name", as_index=False, sort=False).head(1).reset_index(drop=True)
    filtered = retained_conditions.merge(
        selected_lines[["dataset_name", "cell_type"]],
        on=["dataset_name", "cell_type"],
        how="inner",
    )
    filtered = filtered.sort_values(
        ["dataset_name", "cell_type", "primary_source_path", "pubchem_cid", "time_key", "dose_key"]
    ).reset_index(drop=True)
    return filtered, selected_lines


def limit_test_conditions_per_dataset(
    retained_conditions: pd.DataFrame,
    *,
    max_conditions_per_dataset: int,
    min_retrieval_compounds_per_line_time: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if retained_conditions.empty or int(max_conditions_per_dataset) <= 0:
        empty = pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "time_key",
                "dose_key",
                "n_available_conditions",
                "n_available_unique_compounds",
                "n_kept_conditions",
                "n_kept_unique_compounds",
            ]
        )
        return retained_conditions.copy(), empty

    limited_frames: list[pd.DataFrame] = []
    selection_records: list[dict[str, object]] = []

    for dataset_name, dataset_frame in retained_conditions.groupby("dataset_name", sort=False):
        dataset_frame = dataset_frame.copy()
        context_ranking = (
            dataset_frame.groupby(["cell_type", "time_key", "dose_key"], as_index=False)
            .agg(
                n_available_conditions=("condition_key", "size"),
                n_available_unique_compounds=("pubchem_cid", "nunique"),
                total_replicate_pairs=("n_replicate_pairs", "sum"),
                total_replicates=("n_replicates", "sum"),
            )
            .sort_values(
                [
                    "n_available_unique_compounds",
                    "n_available_conditions",
                    "total_replicate_pairs",
                    "total_replicates",
                    "cell_type",
                    "time_key",
                    "dose_key",
                ],
                ascending=[False, False, False, False, True, True, True],
            )
            .reset_index(drop=True)
        )
        if context_ranking.empty:
            continue

        eligible_contexts = context_ranking.loc[
            context_ranking["n_available_unique_compounds"].astype(int)
            >= int(min_retrieval_compounds_per_line_time)
        ].copy()

        best_context = eligible_contexts.iloc[0] if not eligible_contexts.empty else context_ranking.iloc[0]
        cell_type = str(best_context["cell_type"])
        time_key = str(best_context["time_key"])
        dose_key = str(best_context["dose_key"])
        context_frame = dataset_frame.loc[
            (dataset_frame["cell_type"].astype(str) == cell_type)
            & (dataset_frame["time_key"].astype(str) == time_key)
            & (dataset_frame["dose_key"].astype(str) == dose_key)
        ].copy()
        context_frame = context_frame.sort_values(
            [
                "n_replicate_pairs",
                "n_replicates",
                "pubchem_cid",
                "dose_key",
            ],
            ascending=[False, False, True, True],
        ).reset_index(drop=True)
        keep_frame = context_frame.head(int(max_conditions_per_dataset)).copy()

        if keep_frame.empty:
            continue

        selection_records.append(
            {
                "dataset_name": str(dataset_name),
                "cell_type": cell_type,
                "time_key": time_key,
                "dose_key": dose_key,
                "n_available_conditions": int(best_context["n_available_conditions"]),
                "n_available_unique_compounds": int(best_context["n_available_unique_compounds"]),
                "n_kept_conditions": int(len(keep_frame)),
                "n_kept_unique_compounds": int(keep_frame["pubchem_cid"].astype(str).nunique()),
                "meets_retrieval_threshold": bool(
                    int(best_context["n_available_unique_compounds"])
                    >= int(min_retrieval_compounds_per_line_time)
                ),
            }
        )
        limited_frames.append(keep_frame)

    if not limited_frames:
        empty = pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "time_key",
                "dose_key",
                "n_available_conditions",
                "n_available_unique_compounds",
                "n_kept_conditions",
                "n_kept_unique_compounds",
                "meets_retrieval_threshold",
            ]
        )
        return retained_conditions.iloc[0:0].copy(), empty

    limited = pd.concat(limited_frames, ignore_index=True)
    limited = limited.sort_values(
        ["dataset_name", "cell_type", "time_key", "dose_key", "pubchem_cid"]
    ).reset_index(drop=True)
    selection = pd.DataFrame(selection_records).sort_values(["dataset_name"]).reset_index(drop=True)
    return limited, selection


def choose_test_cell_types_from_processed_metadata(
    grouped_candidate_frames: dict[str, pd.DataFrame],
    *,
    dataset_names: list[str],
    min_replicates_per_condition: int,
    allowed_cell_types_by_dataset: Optional[dict[str, set[str]]] = None,
    allowed_line_compounds_by_dataset: Optional[dict[str, set[tuple[str, str]]]] = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        frame = grouped_candidate_frames.get(dataset_name, pd.DataFrame()).copy()
        if frame.empty:
            continue
        allowed_cell_types = None if allowed_cell_types_by_dataset is None else allowed_cell_types_by_dataset.get(dataset_name)
        if allowed_cell_types is not None:
            frame = frame.loc[frame["cell_type"].astype(str).isin(allowed_cell_types)].copy()
        allowed_line_compounds = (
            None if allowed_line_compounds_by_dataset is None else allowed_line_compounds_by_dataset.get(dataset_name)
        )
        if allowed_line_compounds is not None:
            keep_mask = [
                (str(cell_type), str(pubchem_cid)) in allowed_line_compounds
                for cell_type, pubchem_cid in zip(frame["cell_type"].astype(str), frame["pubchem_cid"].astype(str))
            ]
            frame = frame.loc[keep_mask].copy()
        if frame.empty:
            continue
        counts = (
            frame.groupby(["cell_type", "condition_key"], as_index=False)
            .size()
            .rename(columns={"size": "n_rows_for_condition"})
        )
        ranking = (
            counts.groupby("cell_type", as_index=False)
            .agg(
                n_candidate_conditions=("condition_key", "size"),
                n_candidate_conditions_ge_min=("n_rows_for_condition", lambda values: int(np.sum(np.asarray(values) >= int(min_replicates_per_condition)))),
                max_rows_per_condition=("n_rows_for_condition", "max"),
            )
            .merge(
                frame.groupby("cell_type", as_index=False).agg(
                    n_candidate_rows=("condition_key", "size"),
                    n_candidate_compounds=("pubchem_cid", "nunique"),
                ),
                on="cell_type",
                how="left",
            )
            .sort_values(
                [
                    "n_candidate_conditions_ge_min",
                    "n_candidate_conditions",
                    "n_candidate_rows",
                    "n_candidate_compounds",
                    "cell_type",
                ],
                ascending=[False, False, False, False, True],
            )
            .reset_index(drop=True)
        )
        if ranking.empty:
            continue
        top = ranking.iloc[0]
        records.append(
            {
                "dataset_name": dataset_name,
                "cell_type": str(top["cell_type"]),
                "n_candidate_conditions": int(top["n_candidate_conditions"]),
                "n_candidate_conditions_ge_min": int(top["n_candidate_conditions_ge_min"]),
                "n_candidate_rows": int(top["n_candidate_rows"]),
                "n_candidate_compounds": int(top["n_candidate_compounds"]),
                "max_rows_per_condition": int(top["max_rows_per_condition"]),
            }
        )
    return pd.DataFrame(records).sort_values("dataset_name").reset_index(drop=True) if records else pd.DataFrame(
        columns=[
            "dataset_name",
            "cell_type",
            "n_candidate_conditions",
            "n_candidate_conditions_ge_min",
            "n_candidate_rows",
            "n_candidate_compounds",
            "max_rows_per_condition",
        ]
    )


def resolve_dataset_order(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(DEFAULT_SOURCE_DATASET_DIRS)
    dataset_order = [dataset_name.strip() for dataset_name in dataset_arg.split(",") if dataset_name.strip()]
    unknown = [dataset_name for dataset_name in dataset_order if dataset_name not in DEFAULT_SOURCE_DATASET_DIRS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}")
    return dataset_order


def write_prepare_outputs(
    *,
    output_dir: Path,
    dataset_order: list[str],
    dataset_indices: dict[str, dict[str, object]],
    active_datasets: list[str],
    condition_inventory: pd.DataFrame,
    retained_conditions: pd.DataFrame,
    line_global_shared_gene_keys: dict[str, np.ndarray],
    test_line_selection: Optional[pd.DataFrame] = None,
    task_config: Optional[dict[str, object]] = None,
) -> None:
    index_file_summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "n_scanned_files": int(dataset_indices[dataset_name]["n_scanned_files"]),
                "n_nonempty_metadata_files": int(dataset_indices[dataset_name]["n_kept_files"]),
                "n_active_lines": int(len(dataset_indices[dataset_name]["line_to_files"])),
                "scan_mode": str(dataset_indices[dataset_name].get("filter_attempt", "filtered")),
                "fallback_used": bool(dataset_indices[dataset_name].get("fallback_used", False)),
            }
            for dataset_name in active_datasets
        ]
    )

    if condition_inventory.empty:
        condition_summary = pd.DataFrame(columns=[
            "dataset_name",
            "n_conditions_with_replicates",
            "n_retained_conditions",
            "n_line_compounds",
            "max_replicates",
            "total_replicate_pairs",
        ])
    else:
        condition_summary = condition_inventory.groupby("dataset_name", as_index=False).agg(
            n_conditions_with_replicates=("condition_key", "size"),
            n_retained_conditions=("retain_for_eval", "sum"),
            n_line_compounds=(
                "pubchem_cid",
                lambda values: int(len(set(zip(condition_inventory.loc[values.index, "cell_type"], values)))),
            ),
            max_replicates=("n_replicates", "max"),
            total_replicate_pairs=("n_replicate_pairs", "sum"),
        )
    selection_summary = (
        pd.DataFrame({"dataset_name": dataset_order})
        .merge(condition_summary, on="dataset_name", how="left")
        .merge(index_file_summary, on="dataset_name", how="left")
        .sort_values("dataset_name")
        .reset_index(drop=True)
    )
    for column_name in [
        "n_conditions_with_replicates",
        "n_retained_conditions",
        "n_line_compounds",
        "max_replicates",
        "total_replicate_pairs",
        "n_scanned_files",
        "n_nonempty_metadata_files",
        "n_active_lines",
    ]:
        selection_summary[column_name] = selection_summary[column_name].fillna(0).astype(int)
    if "scan_mode" in selection_summary.columns:
        selection_summary["scan_mode"] = selection_summary["scan_mode"].fillna("not_scanned").astype(str)
    if "fallback_used" in selection_summary.columns:
        selection_summary["fallback_used"] = selection_summary["fallback_used"].fillna(False).astype(bool)
    selection_summary_path = output_dir / "dataset_selection_summary.tsv"
    selection_summary.to_csv(selection_summary_path, sep="\t", index=False)
    print(f"Saved dataset selection summary to {selection_summary_path}")

    retained_conditions_path = output_dir / "retained_replicate_conditions.tsv"
    retained_conditions.to_csv(retained_conditions_path, sep="\t", index=False)
    print(f"Saved retained condition inventory to {retained_conditions_path}")

    retained_lines = {
        dataset_name: sorted(
            retained_conditions.loc[retained_conditions["dataset_name"] == dataset_name, "cell_type"].unique().tolist()
        )
        for dataset_name in active_datasets
    }
    line_global_gene_counts = pd.DataFrame(
        {
            "cell_type": list(line_global_shared_gene_keys.keys()),
            "n_global_shared_genes": [int(gene_keys.size) for gene_keys in line_global_shared_gene_keys.values()],
            "datasets": [
                ", ".join(
                    pretty_label(dataset_name)
                    for dataset_name in active_datasets
                    if cell_type in retained_lines.get(dataset_name, [])
                )
                for cell_type in line_global_shared_gene_keys
            ],
        }
    ).sort_values(["n_global_shared_genes", "cell_type"], ascending=[False, True]).reset_index(drop=True)
    line_global_gene_counts_path = output_dir / "line_global_gene_counts.tsv"
    line_global_gene_counts.to_csv(line_global_gene_counts_path, sep="\t", index=False)
    print(f"Saved line-global gene counts to {line_global_gene_counts_path}")

    line_global_gene_key_file = line_global_gene_keys_path(output_dir)
    line_global_gene_key_file.parent.mkdir(parents=True, exist_ok=True)
    write_line_global_gene_keys(line_global_gene_key_file, line_global_shared_gene_keys)
    print(f"Saved line-global gene key sets to {line_global_gene_key_file}")

    cache_dir = dataset_metadata_cache_dir(output_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for dataset_name in active_datasets:
        cache_path = dataset_metadata_cache_path(output_dir, dataset_name)
        dataset_indices[dataset_name]["frame"].to_csv(cache_path, sep="\t", index=False)
    if task_config is not None:
        config_path = task_config_path(output_dir)
        write_task_config(config_path, task_config)
        print(f"Saved task config to {config_path}")

    if test_line_selection is not None:
        test_line_selection_path = output_dir / "test_line_selection.tsv"
        test_line_selection.to_csv(test_line_selection_path, sep="\t", index=False)
        print(f"Saved test-line selection to {test_line_selection_path}")


def create_task_shards(
    *,
    output_dir: Path,
    retained_conditions: pd.DataFrame,
    dataset_indices: dict[str, dict[str, object]],
    conditions_per_task: int,
) -> pd.DataFrame:
    if int(conditions_per_task) < 1:
        raise ValueError("--conditions-per-task must be >= 1")

    input_dir = task_input_dir(output_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    output_task_dir = task_output_dir(output_dir)
    output_task_dir.mkdir(parents=True, exist_ok=True)

    task_records: list[dict[str, object]] = []
    task_id = 0
    shard_group_columns = [
        "dataset_name",
        "source_path_key",
        "cell_type",
        "time_key",
        "dose_key",
    ]
    for shard_key, batch_frame in retained_conditions.groupby(
        shard_group_columns,
        sort=False,
    ):
        dataset_name, source_path_key, cell_type, time_key, dose_key = shard_key
        batch_frame = batch_frame.reset_index(drop=True)
        condition_row_lookup = dataset_indices[dataset_name]["condition_rows_by_key"]
        dataset_frame = dataset_indices[dataset_name]["frame"]
        source_paths = [path for path in str(source_path_key).split("||") if path]

        for chunk_start in range(0, len(batch_frame), int(conditions_per_task)):
            chunk_frame = batch_frame.iloc[chunk_start : chunk_start + int(conditions_per_task)].copy().reset_index(drop=True)
            chunk_condition_keys = chunk_frame["condition_key"].astype(str).tolist()
            row_indexes: list[int] = []
            for condition_key in chunk_condition_keys:
                row_indexes.extend(condition_row_lookup[condition_key])
            unique_row_indexes = sorted(set(map(int, row_indexes)))
            chunk_replicates = dataset_frame.iloc[unique_row_indexes].copy().reset_index(drop=True)

            task_id += 1
            conditions_path = task_conditions_path(output_dir, task_id)
            replicates_path = task_replicates_path(output_dir, task_id)
            chunk_frame.to_csv(conditions_path, sep="\t", index=False)
            chunk_replicates.to_csv(replicates_path, sep="\t", index=False)

            task_records.append(
                {
                    "task_id": task_id,
                    "dataset_name": str(dataset_name),
                    "source_path_key": str(source_path_key),
                    "cell_type": str(cell_type),
                    "time_key": str(time_key),
                    "dose_key": str(dose_key),
                    "n_conditions": int(len(chunk_frame)),
                    "n_replicate_rows": int(len(chunk_replicates)),
                    "n_source_files": int(len(source_paths)),
                    "conditions_path": str(conditions_path),
                    "replicates_path": str(replicates_path),
                }
            )

    manifest = pd.DataFrame(task_records).sort_values("task_id").reset_index(drop=True)
    manifest_path = task_manifest_path(output_dir)
    manifest.to_csv(manifest_path, sep="\t", index=False)
    print(f"Saved task manifest to {manifest_path}")
    print(f"Prepared {len(manifest):,} replicate-scoring tasks")
    return manifest


def load_cached_dataset_indices(output_dir: Path, dataset_names: list[str]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for dataset_name in dataset_names:
        cache_path = dataset_metadata_cache_path(output_dir, dataset_name)
        if not cache_path.exists():
            raise FileNotFoundError(f"Cached dataset metadata not found for {dataset_name}: {cache_path}")
        frame = pd.read_csv(cache_path, sep="\t", keep_default_na=False)
        frame = normalize_source_metadata_frame(frame)
        condition_rows_by_key = {
            str(condition_key): list(map(int, indexes))
            for condition_key, indexes in frame.groupby("condition_key", sort=False).groups.items()
        }
        line_to_files: dict[str, list[str]] = {}
        if not frame.empty:
            grouped = frame.groupby("cell_type", sort=False)["source_path"].apply(lambda values: sorted(set(map(str, values))))
            line_to_files = {str(cell_type): paths for cell_type, paths in grouped.to_dict().items()}
        result[dataset_name] = {
            "frame": frame,
            "condition_rows_by_key": condition_rows_by_key,
            "line_to_files": line_to_files,
            "n_scanned_files": int(frame["source_path"].astype(str).nunique()) if not frame.empty else 0,
            "n_kept_files": int(frame["source_path"].astype(str).nunique()) if not frame.empty else 0,
            "filter_attempt": "cached_strict_filtered",
            "fallback_used": False,
        }
    return result


def clear_existing_task_shards(output_dir: Path, *, clear_task_outputs: bool) -> None:
    manifest = task_manifest_path(output_dir)
    if manifest.exists():
        manifest.unlink()
    for path in task_input_dir(output_dir).glob("task_*_conditions.tsv"):
        path.unlink()
    for path in task_input_dir(output_dir).glob("task_*_replicates.tsv"):
        path.unlink()
    if clear_task_outputs:
        for path in task_output_dir(output_dir).glob("task_*_condition_metric_summary.tsv"):
            path.unlink()
        for path in task_output_dir(output_dir).glob("task_*_condition_scoring_errors.tsv"):
            path.unlink()
        for path in task_output_dir(output_dir).glob("task_*_condition_retrieval_summary.tsv"):
            path.unlink()
        for path in task_output_dir(output_dir).glob("task_*_retrieval_stratum_summary.tsv"):
            path.unlink()


def prepare(
    output_dir: Path,
    dataset_arg: str,
    min_replicates_per_condition: int,
    conditions_per_task: int,
    *,
    test_one_line_per_dataset: bool = False,
    test_max_conditions_per_dataset: int = 0,
    compute_baseline_metrics: bool = False,
    compute_deg_metrics: bool = False,
    compute_retrieval_metrics: bool = False,
    min_retrieval_compounds_per_line_time: int = DEFAULT_MIN_RETRIEVAL_COMPOUNDS_PER_LINE_TIME,
    max_baseline_peers: Optional[int] = 512,
    peer_sampling_seed: int = DEFAULT_PEER_SAMPLING_SEED,
) -> PrepareResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_order = resolve_dataset_order(dataset_arg)
    source_dataset_dirs = {
        dataset_name: DEFAULT_SOURCE_DATASET_DIRS[dataset_name]
        for dataset_name in dataset_order
    }
    non_l1000_datasets = [dataset_name for dataset_name in dataset_order if dataset_name not in L1000_DATASETS]

    print("Scanning processed sep_rep metadata first to determine candidate lines and compounds...")
    grouped_candidate_frames = {
        dataset_name: load_processed_sep_rep_metadata(dataset_name)
        for dataset_name in dataset_order
    }
    non_l1000_grouped_frames = [
        grouped_candidate_frames[dataset_name]
        for dataset_name in non_l1000_datasets
        if not grouped_candidate_frames[dataset_name].empty
    ]
    if not non_l1000_grouped_frames:
        raise ValueError("No non-L1000 processed sep_rep non-control rows were found for the configured datasets.")

    non_l1000_grouped_inventory = pd.concat(non_l1000_grouped_frames, ignore_index=True)
    non_l1000_grouped_inventory = non_l1000_grouped_inventory.drop_duplicates(
        subset=["dataset_name", "cell_type", "pubchem_cid", "time_key", "dose_key", "condition_key"]
    ).reset_index(drop=True)
    supported_line_compounds = set(
        zip(non_l1000_grouped_inventory["cell_type"].astype(str), non_l1000_grouped_inventory["pubchem_cid"].astype(str))
    )
    supported_cell_types = {cell_type for cell_type, _ in supported_line_compounds}
    print(
        f"Grouped non-L1000 support set: {len(supported_cell_types):,} candidate cell lines and "
        f"{len(supported_line_compounds):,} candidate cell line / compound pairs"
    )

    per_dataset_grouped_cell_types = {
        dataset_name: set(grouped_candidate_frames[dataset_name]["cell_type"].astype(str).tolist())
        for dataset_name in dataset_order
    }

    if test_one_line_per_dataset:
        non_l1000_test_lines = choose_test_cell_types_from_processed_metadata(
            grouped_candidate_frames,
            dataset_names=non_l1000_datasets,
            min_replicates_per_condition=min_replicates_per_condition,
        )
        if not non_l1000_test_lines.empty:
            for row in non_l1000_test_lines.itertuples(index=False):
                per_dataset_grouped_cell_types[str(row.dataset_name)] = {str(row.cell_type)}
            print(
                "Test-mode non-L1000 lines: "
                + ", ".join(
                    f"{pretty_label(row.dataset_name)}:{row.cell_type}"
                    for row in non_l1000_test_lines.itertuples(index=False)
                )
            )

    print("Scanning non-L1000 sep_rep metadata only for candidate line files...")
    non_l1000_indices = {
        dataset_name: build_dataset_index_with_fallback(
            dataset_name,
            source_dataset_dirs[dataset_name],
            min_replicates_per_condition=min_replicates_per_condition,
            allowed_cell_types=per_dataset_grouped_cell_types[dataset_name],
            expect_nonempty=not grouped_candidate_frames[dataset_name].empty,
        )
        for dataset_name in non_l1000_datasets
    }
    non_l1000_active_datasets = [
        dataset_name
        for dataset_name in non_l1000_datasets
        if not non_l1000_indices[dataset_name]["frame"].empty
    ]
    if not non_l1000_active_datasets:
        raise ValueError("No non-L1000 sep_rep non-control rows were found for the configured datasets.")

    non_l1000_inventory = build_condition_inventory(non_l1000_indices, non_l1000_active_datasets)
    if non_l1000_inventory.empty:
        raise ValueError("No non-L1000 replicate-supported conditions were found.")

    l1000_allowed_cell_types = supported_cell_types
    if test_one_line_per_dataset:
        l1000_test_lines = choose_test_cell_types_from_processed_metadata(
            grouped_candidate_frames,
            dataset_names=[dataset_name for dataset_name in dataset_order if dataset_name in L1000_DATASETS],
            min_replicates_per_condition=min_replicates_per_condition,
            allowed_cell_types_by_dataset={
                dataset_name: supported_cell_types
                for dataset_name in dataset_order
                if dataset_name in L1000_DATASETS
            },
            allowed_line_compounds_by_dataset={
                dataset_name: supported_line_compounds
                for dataset_name in dataset_order
                if dataset_name in L1000_DATASETS
            },
        )
        if not l1000_test_lines.empty:
            print(
                "Test-mode L1000 lines: "
                + ", ".join(
                    f"{pretty_label(row.dataset_name)}:{row.cell_type}"
                    for row in l1000_test_lines.itertuples(index=False)
                )
            )
        l1000_allowed_cell_types_by_dataset = {
            dataset_name: supported_cell_types
            for dataset_name in dataset_order
            if dataset_name in L1000_DATASETS
        }
        for row in l1000_test_lines.itertuples(index=False):
            l1000_allowed_cell_types_by_dataset[str(row.dataset_name)] = {str(row.cell_type)}
        l1000_allowed_line_compounds_by_dataset = {
            dataset_name: None
            for dataset_name in dataset_order
            if dataset_name in L1000_DATASETS
        }
    else:
        l1000_allowed_cell_types_by_dataset = {
            dataset_name: l1000_allowed_cell_types
            for dataset_name in dataset_order
            if dataset_name in L1000_DATASETS
        }
        l1000_allowed_line_compounds_by_dataset = {
            dataset_name: supported_line_compounds
            for dataset_name in dataset_order
            if dataset_name in L1000_DATASETS
        }

    print("Scanning L1000 sep_rep metadata only for supported lines and compounds...")
    l1000_indices = {
        dataset_name: build_dataset_index_with_fallback(
            dataset_name,
            source_dataset_dirs[dataset_name],
            min_replicates_per_condition=min_replicates_per_condition,
            allowed_cell_types=l1000_allowed_cell_types_by_dataset[dataset_name],
            allowed_line_compounds=l1000_allowed_line_compounds_by_dataset[dataset_name],
            expect_nonempty=test_one_line_per_dataset or bool(l1000_allowed_line_compounds_by_dataset[dataset_name]),
        )
        for dataset_name in dataset_order
        if dataset_name in L1000_DATASETS
    }

    dataset_indices = {**non_l1000_indices, **l1000_indices}
    active_datasets = [
        dataset_name
        for dataset_name in dataset_order
        if dataset_name in dataset_indices and not dataset_indices[dataset_name]["frame"].empty
    ]
    if not active_datasets:
        raise ValueError("No sep_rep non-control rows were found for the configured datasets.")

    print("Datasets in scope:", ", ".join(pretty_label(dataset_name) for dataset_name in active_datasets))
    condition_inventory = build_condition_inventory(dataset_indices, active_datasets)
    if condition_inventory.empty:
        raise ValueError("No replicate-supported conditions were found.")

    non_l1000_support = (
        non_l1000_grouped_inventory.groupby(["cell_type", "pubchem_cid"], as_index=False)
        .agg(n_non_l1000_supporting_datasets=("dataset_name", "nunique"))
    )
    condition_inventory = condition_inventory.merge(
        non_l1000_support,
        on=["cell_type", "pubchem_cid"],
        how="left",
    )
    condition_inventory["n_non_l1000_supporting_datasets"] = (
        condition_inventory["n_non_l1000_supporting_datasets"].fillna(0).astype(int)
    )
    condition_inventory["retain_for_eval"] = (
        ~condition_inventory["dataset_name"].isin(L1000_DATASETS)
        | condition_inventory["n_non_l1000_supporting_datasets"].gt(0)
    )

    retained_conditions = condition_inventory.loc[condition_inventory["retain_for_eval"]].copy()
    retained_conditions = retained_conditions.sort_values(
        ["dataset_name", "cell_type", "primary_source_path", "pubchem_cid", "time_key", "dose_key"]
    ).reset_index(drop=True)
    if retained_conditions.empty:
        raise ValueError("No retained replicate-supported conditions remain after applying the L1000 restriction.")

    test_line_selection: Optional[pd.DataFrame] = None
    if test_one_line_per_dataset:
        retained_conditions, test_line_selection = select_test_lines_per_dataset(retained_conditions)
        if retained_conditions.empty:
            raise ValueError("Test mode selected no retained conditions.")
        print(
            "Test mode active: keeping one line per dataset -> "
            + ", ".join(
                f"{pretty_label(row.dataset_name)}:{row.cell_type}"
                for row in test_line_selection.itertuples(index=False)
            )
        )
        if int(test_max_conditions_per_dataset) > 0:
            retained_conditions, limited_condition_selection = limit_test_conditions_per_dataset(
                retained_conditions,
                max_conditions_per_dataset=int(test_max_conditions_per_dataset),
                min_retrieval_compounds_per_line_time=int(min_retrieval_compounds_per_line_time),
            )
            if retained_conditions.empty:
                raise ValueError("Test mode condition cap removed all retained conditions.")
            test_line_selection = (
                limited_condition_selection
                if test_line_selection is None or test_line_selection.empty
                else test_line_selection.merge(
                    limited_condition_selection,
                    on=["dataset_name", "cell_type"],
                    how="left",
                )
            )
            print(
                "Test mode active: capped retained conditions per dataset -> "
                + ", ".join(
                    (
                        f"{pretty_label(row.dataset_name)}:{row.cell_type}/time={row.time_key}"
                        f" ({row.n_kept_conditions} kept, {row.n_kept_unique_compounds} compounds)"
                    )
                    for row in limited_condition_selection.itertuples(index=False)
                )
            )

    retained_lines = {
        dataset_name: sorted(
            retained_conditions.loc[retained_conditions["dataset_name"] == dataset_name, "cell_type"].unique().tolist()
        )
        for dataset_name in active_datasets
    }
    line_global_shared_gene_keys = set_line_global_shared_gene_keys(
        retained_lines,
        active_datasets,
        dataset_indices,
    )

    write_prepare_outputs(
        output_dir=output_dir,
        dataset_order=dataset_order,
        dataset_indices=dataset_indices,
        active_datasets=active_datasets,
        condition_inventory=condition_inventory,
        retained_conditions=retained_conditions,
        line_global_shared_gene_keys=line_global_shared_gene_keys,
        test_line_selection=test_line_selection,
        task_config={
            "datasets": dataset_order,
            "conditions_per_task": int(conditions_per_task),
            "min_replicates_per_condition": int(min_replicates_per_condition),
            "test_one_line_per_dataset": bool(test_one_line_per_dataset),
            "test_max_conditions_per_dataset": int(test_max_conditions_per_dataset),
            "compute_baseline_metrics": bool(compute_baseline_metrics),
            "compute_deg_metrics": bool(compute_deg_metrics),
            "compute_retrieval_metrics": bool(compute_retrieval_metrics),
            "min_retrieval_compounds_per_line_time": int(min_retrieval_compounds_per_line_time),
            "max_baseline_peers": (
                int(max_baseline_peers) if max_baseline_peers is not None else None
            ),
            "peer_sampling_seed": int(peer_sampling_seed),
            "peer_baseline_engine_version": 2,
        },
    )
    create_task_shards(
        output_dir=output_dir,
        retained_conditions=retained_conditions,
        dataset_indices=dataset_indices,
        conditions_per_task=conditions_per_task,
    )
    return PrepareResult(
        output_dir=output_dir,
        task_manifest_path=task_manifest_path(output_dir),
        task_output_dir=task_output_dir(output_dir),
    )


def reshard(output_dir: Path, *, conditions_per_task: int) -> ReshardResult:
    retained_conditions_path = output_dir / "retained_replicate_conditions.tsv"
    if not retained_conditions_path.exists():
        raise FileNotFoundError(f"Retained condition inventory not found: {retained_conditions_path}")
    retained_conditions = pd.read_csv(retained_conditions_path, sep="\t", keep_default_na=False)
    if retained_conditions.empty:
        raise ValueError("Retained condition inventory is empty; cannot rebuild shards.")
    dataset_names = sorted(retained_conditions["dataset_name"].astype(str).unique().tolist())
    dataset_indices = load_cached_dataset_indices(output_dir, dataset_names)
    clear_existing_task_shards(output_dir, clear_task_outputs=True)
    create_task_shards(
        output_dir=output_dir,
        retained_conditions=retained_conditions,
        dataset_indices=dataset_indices,
        conditions_per_task=conditions_per_task,
    )
    config_path = task_config_path(output_dir)
    if config_path.exists():
        config = load_task_config(config_path)
    else:
        config = {}
    config["conditions_per_task"] = int(conditions_per_task)
    write_task_config(config_path, config)
    print(f"Updated task config at {config_path}")
    return ReshardResult(
        output_dir=output_dir,
        task_manifest_path=task_manifest_path(output_dir),
        task_output_dir=task_output_dir(output_dir),
    )


def run_task(
    *,
    output_dir: Path,
    task_file: Path,
    task_id: int,
    task_output_dir_path: Optional[Path],
    top_k: int,
    compute_baseline_metrics: bool,
    compute_deg_metrics: bool,
    compute_retrieval_metrics: bool,
    min_retrieval_compounds_per_line_time: int,
) -> None:
    saved_config = (
        load_task_config(task_config_path(output_dir))
        if task_config_path(output_dir).exists()
        else {}
    )
    saved_config = {
        **saved_config,
        "max_baseline_peers": (
            int(MAX_BASELINE_PEERS) if MAX_BASELINE_PEERS is not None else None
        ),
        "peer_sampling_seed": int(PEER_SAMPLING_SEED),
        "peer_baseline_engine_version": 2,
    }
    peer_config_fingerprint = config_fingerprint(saved_config)
    manifest = load_task_manifest(task_file)
    task_row = manifest.loc[manifest["task_id"] == int(task_id)]
    if task_row.empty:
        raise ValueError(f"Task id {task_id} not found in {task_file}")
    task_row = task_row.iloc[0]

    task_output_dir_path = task_output_dir_path or task_output_dir(output_dir)
    task_output_dir_path.mkdir(parents=True, exist_ok=True)

    conditions_path = Path(str(task_row["conditions_path"]))
    replicates_path = Path(str(task_row["replicates_path"]))
    conditions_frame = pd.read_csv(conditions_path, sep="\t", keep_default_na=False)
    replicates_frame = pd.read_csv(replicates_path, sep="\t", keep_default_na=False)
    conditions_frame = normalize_source_metadata_frame(conditions_frame)
    replicates_frame = normalize_source_metadata_frame(replicates_frame)

    global_gene_keys = load_line_global_gene_keys(line_global_gene_keys_path(output_dir))
    dataset_name = str(task_row["dataset_name"])
    replicate_groups = {
        str(condition_key): block.copy().reset_index(drop=True)
        for condition_key, block in replicates_frame.groupby("condition_key", sort=False)
    }
    retained_conditions_all = pd.read_csv(
        output_dir / "retained_replicate_conditions.tsv",
        sep="\t",
        keep_default_na=False,
    )
    retained_conditions_for_dataset = retained_conditions_all.loc[
        retained_conditions_all["dataset_name"].astype(str) == dataset_name
    ].copy()
    retained_condition_keys = set(retained_conditions_for_dataset["condition_key"].astype(str).tolist())

    full_dataset_source_frame: Optional[pd.DataFrame] = None
    baseline_source_frame: Optional[pd.DataFrame] = None
    baseline_context_row_indexes: Optional[dict[tuple[str, str, str], np.ndarray]] = None
    if compute_baseline_metrics or compute_retrieval_metrics:
        full_dataset_source_frame = pd.read_csv(
            dataset_metadata_cache_path(output_dir, dataset_name),
            sep="\t",
            keep_default_na=False,
        )
        full_dataset_source_frame = normalize_source_metadata_frame(full_dataset_source_frame)
        full_dataset_source_frame = full_dataset_source_frame.loc[
            full_dataset_source_frame["condition_key"].astype(str).isin(retained_condition_keys)
        ].copy()
    if compute_baseline_metrics and full_dataset_source_frame is not None:
        baseline_source_frame = full_dataset_source_frame.copy()
        task_contexts = conditions_frame[["cell_type", "time_key", "dose_key"]].drop_duplicates().copy()
        baseline_source_frame = baseline_source_frame.merge(
            task_contexts.assign(_keep=1),
            on=["cell_type", "time_key", "dose_key"],
            how="inner",
        ).drop(columns="_keep")
        baseline_context_row_indexes = {
            (str(cell_type), str(time_key), str(dose_key)): indexes.to_numpy(dtype=np.int64)
            for (cell_type, time_key, dose_key), indexes in baseline_source_frame.groupby(
                ["cell_type", "time_key", "dose_key"],
                sort=False,
            ).groups.items()
        }

    open_adatas: dict[str, ad.AnnData] = {}
    baseline_context_vector_cache: dict[
        tuple[object, ...], BaselineContextVectors
    ] = {}
    metric_records: list[dict[str, object]] = []
    error_records: list[dict[str, object]] = []
    retrieval_condition_summary = pd.DataFrame()
    try:
        source_paths_to_open = set(replicates_frame["source_path"].astype(str).unique().tolist())
        if full_dataset_source_frame is not None and not full_dataset_source_frame.empty:
            source_paths_to_open.update(full_dataset_source_frame["source_path"].astype(str).unique().tolist())
        for source_path in sorted(source_paths_to_open):
            open_adatas[source_path] = read_h5ad_safely(source_path, backed="r")

        for _, condition_row in conditions_frame.iterrows():
            condition_key = str(condition_row["condition_key"])
            condition_rows = replicate_groups.get(condition_key)
            if condition_rows is None or condition_rows.empty:
                error_records.append(
                    {
                        "dataset_name": condition_row["dataset_name"],
                        "cell_type": condition_row["cell_type"],
                        "pubchem_cid": condition_row["pubchem_cid"],
                        "time_key": condition_row["time_key"],
                        "dose_key": condition_row["dose_key"],
                        "error": "Missing replicate rows for prepared condition shard",
                    }
                )
                continue
            try:
                record = compute_condition_metric_record_from_rows(
                    condition_row,
                    condition_rows,
                    line_global_shared_gene_keys=global_gene_keys,
                    top_k=top_k,
                    compute_deg_metrics=compute_deg_metrics,
                    compute_baseline_metrics=compute_baseline_metrics,
                    baseline_source_frame=baseline_source_frame,
                    baseline_context_row_indexes=baseline_context_row_indexes,
                    baseline_context_vector_cache=baseline_context_vector_cache,
                    open_adatas=open_adatas,
                )
            except Exception as exc:
                error_records.append(
                    {
                        "dataset_name": condition_row["dataset_name"],
                        "cell_type": condition_row["cell_type"],
                        "pubchem_cid": condition_row["pubchem_cid"],
                        "time_key": condition_row["time_key"],
                        "dose_key": condition_row["dose_key"],
                        "error": repr(exc),
                    }
                )
                continue
            if record is not None:
                record["max_baseline_peers"] = (
                    int(MAX_BASELINE_PEERS)
                    if MAX_BASELINE_PEERS is not None
                    else 0
                )
                record["peer_sampling_seed"] = int(PEER_SAMPLING_SEED)
                record["peer_config_fingerprint"] = peer_config_fingerprint
                metric_records.append(record)

        if compute_retrieval_metrics and full_dataset_source_frame is not None and not full_dataset_source_frame.empty:
            retrieval_condition_summary = compute_retrieval_condition_task_summary(
                dataset_name=dataset_name,
                task_conditions=conditions_frame,
                dataset_metadata_frame=full_dataset_source_frame,
                line_global_shared_gene_keys=global_gene_keys,
                min_retrieval_compounds_per_line_time=min_retrieval_compounds_per_line_time,
                open_adatas=open_adatas,
            )
    finally:
        for adata in open_adatas.values():
            if getattr(adata, "file", None) is not None:
                adata.file.close()

    metric_path = task_metrics_path(task_output_dir_path, int(task_id))
    error_path = task_errors_path(task_output_dir_path, int(task_id))
    retrieval_path = task_condition_retrieval_path(task_output_dir_path, int(task_id))

    write_tsv_atomic(pd.DataFrame(metric_records), metric_path)
    if compute_retrieval_metrics:
        write_tsv_atomic(retrieval_condition_summary, retrieval_path)
    elif retrieval_path.exists():
        retrieval_path.unlink()
    if error_records:
        write_tsv_atomic(pd.DataFrame(error_records), error_path)
    elif error_path.exists():
        error_path.unlink()

    print(
        f"Finished task {int(task_id):,}: "
        f"{len(metric_records):,} scored conditions, {len(error_records):,} errors"
    )


def build_final_summaries(
    condition_metric_summary: pd.DataFrame,
    *,
    top_k: int,
    all_dataset_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_records = [
        {
            "dataset_name": dataset_name,
            **summarize_condition_frame(frame, top_k).to_dict(),
        }
        for dataset_name, frame in condition_metric_summary.groupby("dataset_name", sort=False)
    ]
    dataset_metric_summary = pd.DataFrame(summary_records)
    if all_dataset_names is not None:
        existing = set(dataset_metric_summary["dataset_name"].astype(str).tolist()) if not dataset_metric_summary.empty else set()
        missing = [dataset_name for dataset_name in all_dataset_names if dataset_name not in existing]
        if missing:
            if dataset_metric_summary.empty:
                empty_row_template = empty_summary_record(top_k)
            else:
                empty_row_template = {
                    column_name: float("nan")
                    for column_name in dataset_metric_summary.columns
                    if column_name != "dataset_name"
                }
                for column_name in [
                    "n_conditions",
                    "n_unique_lines",
                    "n_unique_compounds",
                    "n_total_replicates",
                    "n_total_replicate_pairs",
                ]:
                    if column_name in empty_row_template:
                        empty_row_template[column_name] = 0
            dataset_metric_summary = pd.concat(
                [
                    dataset_metric_summary,
                    pd.DataFrame(
                        [{"dataset_name": dataset_name, **empty_row_template} for dataset_name in missing]
                    ),
                ],
                ignore_index=True,
            )
    dataset_metric_summary = dataset_metric_summary.sort_values(["dataset_name"]).reset_index(drop=True)

    line_metric_summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "cell_type": cell_type,
                **summarize_condition_frame(frame, top_k).to_dict(),
            }
            for (dataset_name, cell_type), frame in condition_metric_summary.groupby(["dataset_name", "cell_type"], sort=False)
        ]
    ).sort_values(["dataset_name", "cell_type"]).reset_index(drop=True)
    return dataset_metric_summary, line_metric_summary


def build_t_strength_relationship_summaries(
    condition_metric_summary: pd.DataFrame,
    *,
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    relationship_specs = [
        ("spearman_mean_abs_t_vs_replicate_logfc", "mean_abs_t", "mean_replicate_spearman_logfc"),
        ("spearman_mean_abs_t_vs_replicate_t", "mean_abs_t", "mean_replicate_spearman_t"),
        (
            f"spearman_mean_abs_t_vs_replicate_signed_overlap_t_top{top_k}",
            "mean_abs_t",
            f"mean_replicate_signed_overlap_t_top{top_k}",
        ),
        (
            "spearman_mean_abs_t_global_vs_replicate_logfc_global",
            "mean_abs_t_global",
            "mean_replicate_spearman_logfc_global",
        ),
        (
            "spearman_mean_abs_t_global_vs_replicate_t_global",
            "mean_abs_t_global",
            "mean_replicate_spearman_t_global",
        ),
        (
            f"spearman_mean_abs_t_global_vs_replicate_signed_overlap_t_top{top_k}_global",
            "mean_abs_t_global",
            f"mean_replicate_signed_overlap_t_top{top_k}_global",
        ),
    ]

    dataset_summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "n_conditions": int(len(frame)),
                **{
                    output_column: safe_column_spearman(frame, x_col, y_col)
                    for output_column, x_col, y_col in relationship_specs
                },
            }
            for dataset_name, frame in condition_metric_summary.groupby("dataset_name", sort=False)
        ]
    ).sort_values("dataset_name").reset_index(drop=True)

    line_summary = pd.DataFrame(
        [
            {
                "dataset_name": dataset_name,
                "cell_type": cell_type,
                "n_conditions": int(len(frame)),
                **{
                    output_column: safe_column_spearman(frame, x_col, y_col)
                    for output_column, x_col, y_col in relationship_specs
                },
            }
            for (dataset_name, cell_type), frame in condition_metric_summary.groupby(
                ["dataset_name", "cell_type"],
                sort=False,
            )
        ]
    ).sort_values(["dataset_name", "cell_type"]).reset_index(drop=True)

    return dataset_summary, line_summary


def deg_metric_columns(frame: pd.DataFrame) -> list[str]:
    deg_metric_name_tokens = tuple(deg_metric_names())
    allowed_prefixes = (
        "mean_replicate_",
        "median_replicate_",
        "mean_baseline_pair_",
        "median_baseline_pair_",
        "mean_delta_vs_baseline_pair_",
        "median_delta_vs_baseline_pair_",
        "mean_peer_baseline_",
        "mean_delta_vs_peer_baseline_",
    )
    return sorted(
        column_name
        for column_name in frame.columns
        if column_name.startswith(allowed_prefixes)
        and any(metric_name in column_name for metric_name in deg_metric_name_tokens)
    )


def build_retrieval_final_summaries(
    condition_retrieval_summary: pd.DataFrame,
    *,
    all_dataset_names: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if condition_retrieval_summary.empty:
        empty_condition = pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "time_key",
                "representation",
                "condition_key",
                "pubchem_cid",
                "dose_key",
                "perturbagen_display",
                "n_queries",
                "n_queries_with_baseline",
                "n_target_conditions",
                "n_unique_compounds",
                "mean_observed_normalized_best_positive_rank",
                "mean_baseline_normalized_best_positive_rank",
                "mean_delta_vs_baseline_normalized_best_positive_rank",
            ]
        )
        empty_stratum = pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "time_key",
                "representation",
                "n_conditions",
                "n_queries",
                "n_queries_with_baseline",
                "n_target_conditions",
                "n_unique_compounds",
                "mean_observed_normalized_best_positive_rank",
                "mean_baseline_normalized_best_positive_rank",
                "mean_delta_vs_baseline_normalized_best_positive_rank",
            ]
        )
        empty_dataset = pd.DataFrame(
            columns=[
                "dataset_name",
                "representation",
                "n_line_time_strata",
                "n_conditions",
                "n_queries",
                "n_queries_with_baseline",
                "mean_target_conditions",
                "mean_unique_compounds",
                "mean_observed_normalized_best_positive_rank",
                "mean_baseline_normalized_best_positive_rank",
                "mean_delta_vs_baseline_normalized_best_positive_rank",
            ]
        )
        empty_line = pd.DataFrame(
            columns=[
                "dataset_name",
                "cell_type",
                "representation",
                "n_time_strata",
                "n_conditions",
                "n_queries",
                "n_queries_with_baseline",
                "mean_target_conditions",
                "mean_unique_compounds",
                "mean_observed_normalized_best_positive_rank",
                "mean_baseline_normalized_best_positive_rank",
                "mean_delta_vs_baseline_normalized_best_positive_rank",
            ]
        )
        return empty_condition, empty_stratum, empty_dataset, empty_line

    condition_summary = condition_retrieval_summary.copy()
    condition_summary = condition_summary.sort_values(
        ["dataset_name", "cell_type", "time_key", "representation", "condition_key"]
    ).reset_index(drop=True)

    stratum_summary = (
        condition_summary.groupby(
            ["dataset_name", "cell_type", "time_key", "representation"],
            as_index=False,
        )
        .agg(
            n_conditions=("condition_key", "size"),
            n_queries=("n_queries", "sum"),
            n_queries_with_baseline=("n_queries_with_baseline", "sum"),
            n_target_conditions=("n_target_conditions", "max"),
            n_unique_compounds=("n_unique_compounds", "max"),
            mean_observed_normalized_best_positive_rank=("mean_observed_normalized_best_positive_rank", "mean"),
            mean_baseline_normalized_best_positive_rank=("mean_baseline_normalized_best_positive_rank", "mean"),
            mean_delta_vs_baseline_normalized_best_positive_rank=("mean_delta_vs_baseline_normalized_best_positive_rank", "mean"),
        )
        .sort_values(["dataset_name", "cell_type", "time_key", "representation"])
        .reset_index(drop=True)
    )

    dataset_summary = (
        stratum_summary.groupby(["dataset_name", "representation"], as_index=False)
        .agg(
            n_line_time_strata=("time_key", "size"),
            n_conditions=("n_conditions", "sum"),
            n_queries=("n_queries", "sum"),
            n_queries_with_baseline=("n_queries_with_baseline", "sum"),
            mean_target_conditions=("n_target_conditions", "mean"),
            mean_unique_compounds=("n_unique_compounds", "mean"),
            mean_observed_normalized_best_positive_rank=("mean_observed_normalized_best_positive_rank", "mean"),
            mean_baseline_normalized_best_positive_rank=("mean_baseline_normalized_best_positive_rank", "mean"),
            mean_delta_vs_baseline_normalized_best_positive_rank=("mean_delta_vs_baseline_normalized_best_positive_rank", "mean"),
        )
        .sort_values(["dataset_name", "representation"])
        .reset_index(drop=True)
    )
    if all_dataset_names is not None:
        representations = sorted(dataset_summary["representation"].astype(str).unique().tolist())
        if not representations:
            representations = ["logFC"]
        full_index = pd.MultiIndex.from_product([all_dataset_names, representations], names=["dataset_name", "representation"]).to_frame(index=False)
        dataset_summary = full_index.merge(dataset_summary, on=["dataset_name", "representation"], how="left")
        for column_name in ["n_line_time_strata", "n_conditions", "n_queries", "n_queries_with_baseline"]:
            dataset_summary[column_name] = dataset_summary[column_name].fillna(0).astype(int)
        dataset_summary = dataset_summary.sort_values(["dataset_name", "representation"]).reset_index(drop=True)

    line_summary = (
        stratum_summary.groupby(["dataset_name", "cell_type", "representation"], as_index=False)
        .agg(
            n_time_strata=("time_key", "size"),
            n_conditions=("n_conditions", "sum"),
            n_queries=("n_queries", "sum"),
            n_queries_with_baseline=("n_queries_with_baseline", "sum"),
            mean_target_conditions=("n_target_conditions", "mean"),
            mean_unique_compounds=("n_unique_compounds", "mean"),
            mean_observed_normalized_best_positive_rank=("mean_observed_normalized_best_positive_rank", "mean"),
            mean_baseline_normalized_best_positive_rank=("mean_baseline_normalized_best_positive_rank", "mean"),
            mean_delta_vs_baseline_normalized_best_positive_rank=("mean_delta_vs_baseline_normalized_best_positive_rank", "mean"),
        )
        .sort_values(["dataset_name", "cell_type", "representation"])
        .reset_index(drop=True)
    )
    return condition_summary, stratum_summary, dataset_summary, line_summary


def read_optional_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def coerce_non_identifier_columns_to_numeric(frame: pd.DataFrame, identifier_columns: set[str]) -> pd.DataFrame:
    coerced = frame.copy()
    for column_name in coerced.columns:
        if column_name in identifier_columns:
            continue
        coerced[column_name] = pd.to_numeric(coerced[column_name], errors="coerce")
    return coerced


def combine_ordered_unique_names(*name_lists: Optional[list[str]]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for names in name_lists:
        if names is None:
            continue
        for name in names:
            name = str(name)
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
    return result


def read_dataset_names_from_selection_summary(results_dir: Path) -> Optional[list[str]]:
    selection_summary_path = results_dir / "dataset_selection_summary.tsv"
    if not selection_summary_path.exists():
        return None
    selection_summary = pd.read_csv(selection_summary_path, sep="\t", keep_default_na=False)
    if "dataset_name" not in selection_summary.columns:
        return None
    return selection_summary["dataset_name"].astype(str).tolist()


def drop_duplicate_condition_rows(frame: pd.DataFrame) -> pd.DataFrame:
    duplicate_key_options = [
        ["dataset_name", "condition_key"],
        ["dataset_name", "cell_type", "pubchem_cid", "time_key", "dose_key"],
    ]
    for key_columns in duplicate_key_options:
        if all(column_name in frame.columns for column_name in key_columns):
            return frame.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)
    return frame.reset_index(drop=True)


def drop_duplicate_retrieval_rows(frame: pd.DataFrame) -> pd.DataFrame:
    duplicate_key_options = [
        ["dataset_name", "representation", "condition_key"],
        ["dataset_name", "cell_type", "time_key", "representation", "pubchem_cid", "dose_key"],
    ]
    for key_columns in duplicate_key_options:
        if all(column_name in frame.columns for column_name in key_columns):
            return frame.drop_duplicates(subset=key_columns, keep="last").reset_index(drop=True)
    return frame.reset_index(drop=True)


def merge_task_outputs(
    *,
    output_dir: Path,
    task_file: Path,
    task_output_dir_path: Optional[Path],
    top_k: int,
    strict_missing: bool,
    existing_results_dir: Optional[Path] = None,
) -> None:
    manifest = load_task_manifest(task_file)
    task_output_dir_path = task_output_dir_path or task_output_dir(output_dir)
    config: dict[str, object] = {}
    config_path = task_config_path(output_dir)
    if config_path.exists():
        config = load_task_config(config_path)
    expect_baseline_metrics = bool(config.get("compute_baseline_metrics", False))
    expect_deg_metrics = bool(config.get("compute_deg_metrics", False))
    expect_retrieval_metrics = bool(config.get("compute_retrieval_metrics", False))
    expected_peer_config_fingerprint = config_fingerprint(config) if config else None
    existing_results_dir = existing_results_dir.resolve() if existing_results_dir is not None else None
    current_dataset_names = read_dataset_names_from_selection_summary(output_dir)
    existing_dataset_names = (
        read_dataset_names_from_selection_summary(existing_results_dir)
        if existing_results_dir is not None
        else None
    )
    all_dataset_names = combine_ordered_unique_names(existing_dataset_names, current_dataset_names) or None

    metric_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    retrieval_condition_frames: list[pd.DataFrame] = []
    missing_task_outputs: list[dict[str, object]] = []

    if existing_results_dir is not None:
        existing_condition_metric_path = existing_results_dir / "condition_metric_summary.tsv"
        if not existing_condition_metric_path.exists():
            raise FileNotFoundError(
                f"Existing condition metric summary not found: {existing_condition_metric_path}"
            )
        existing_condition_metric = read_optional_tsv(existing_condition_metric_path)
        if not existing_condition_metric.empty:
            metric_frames.append(existing_condition_metric)
            print(
                "Loaded existing condition-level metric summary from "
                f"{existing_condition_metric_path}"
            )

        existing_error_path = existing_results_dir / "condition_scoring_errors.tsv"
        existing_errors = read_optional_tsv(existing_error_path)
        if not existing_errors.empty:
            error_frames.append(existing_errors)

        existing_retrieval_path = existing_results_dir / "condition_retrieval_summary.tsv"
        existing_retrieval = read_optional_tsv(existing_retrieval_path)
        if not existing_retrieval.empty:
            retrieval_condition_frames.append(existing_retrieval)

    for _, task_row in manifest.iterrows():
        task_id = int(task_row["task_id"])
        metric_path = task_metrics_path(task_output_dir_path, task_id)
        error_path = task_errors_path(task_output_dir_path, task_id)
        retrieval_path = task_condition_retrieval_path(task_output_dir_path, task_id)

        if metric_path.exists():
            frame = read_optional_tsv(metric_path)
            if not frame.empty:
                if expect_baseline_metrics and expected_peer_config_fingerprint is not None:
                    if "peer_config_fingerprint" not in frame.columns:
                        raise ValueError(
                            f"Task {task_id} predates peer-baseline cache provenance. "
                            "Rerun stage 2 before merging."
                        )
                    observed_fingerprints = set(
                        frame["peer_config_fingerprint"].astype(str).tolist()
                    )
                    if observed_fingerprints != {expected_peer_config_fingerprint}:
                        raise ValueError(
                            f"Task {task_id} has stale peer-baseline configuration "
                            f"{sorted(observed_fingerprints)!r}; expected "
                            f"{expected_peer_config_fingerprint!r}. Rerun stage 2."
                        )
                metric_frames.append(frame)
        else:
            missing_task_outputs.append(
                {
                    "task_id": task_id,
                    "dataset_name": task_row["dataset_name"],
                    "reason": "missing_metric_output",
                    "expected_path": str(metric_path),
                }
            )

        if error_path.exists():
            frame = read_optional_tsv(error_path)
            if not frame.empty:
                error_frames.append(frame)
        if retrieval_path.exists():
            frame = read_optional_tsv(retrieval_path)
            if not frame.empty:
                retrieval_condition_frames.append(frame)

    if missing_task_outputs:
        missing_frame = pd.DataFrame(missing_task_outputs)
        missing_path = output_dir / "missing_task_outputs.tsv"
        missing_frame.to_csv(missing_path, sep="\t", index=False)
        print(f"Saved missing task output report to {missing_path}")
        if strict_missing:
            raise FileNotFoundError(
                f"{len(missing_task_outputs)} task outputs are missing. "
                f"See {missing_path}"
            )

    if not metric_frames:
        raise ValueError("No task metric outputs were found to merge.")

    condition_metric_summary = pd.concat(metric_frames, ignore_index=True)
    condition_metric_summary = drop_duplicate_condition_rows(condition_metric_summary)
    condition_metric_summary = coerce_non_identifier_columns_to_numeric(
        condition_metric_summary,
        identifier_columns={
            "dataset_name",
            "cell_type",
            "pubchem_cid",
            "time_key",
            "dose_key",
            "condition_key",
            "perturbagen_display",
            "peer_config_fingerprint",
        },
    )
    condition_metric_summary = condition_metric_summary.sort_values(
        ["dataset_name", "cell_type", "pubchem_cid", "time_key", "dose_key"]
    ).reset_index(drop=True)
    condition_metric_summary_path = output_dir / "condition_metric_summary.tsv"
    condition_metric_summary.to_csv(condition_metric_summary_path, sep="\t", index=False)
    print(f"Saved condition-level metric summary to {condition_metric_summary_path}")
    print(f"Scored {len(condition_metric_summary):,} replicate-supported conditions")
    if (
        expect_baseline_metrics
        and "n_baseline_peer_rows" in condition_metric_summary.columns
        and condition_metric_summary["n_baseline_peer_rows"].fillna(0).max() <= 0
    ):
        print(
            "WARNING: baseline metrics were requested, but no baseline peer rows were found "
            "in the merged condition summaries. This usually means stage 2 reused task shards "
            "prepared without baseline support. Rerun from stage 1."
        )

    if error_frames:
        error_summary = pd.concat(error_frames, ignore_index=True)
        error_summary = error_summary.sort_values(
            ["dataset_name", "cell_type", "pubchem_cid", "time_key", "dose_key", "error"]
        ).reset_index(drop=True)
        error_path = output_dir / "condition_scoring_errors.tsv"
        error_summary.to_csv(error_path, sep="\t", index=False)
        print(f"Saved condition scoring errors to {error_path}")

    dataset_metric_summary, line_metric_summary = build_final_summaries(
        condition_metric_summary,
        top_k=top_k,
        all_dataset_names=all_dataset_names,
    )
    dataset_metric_summary_path = output_dir / "dataset_metric_summary.tsv"
    line_metric_summary_path = output_dir / "dataset_line_metric_summary.tsv"
    dataset_metric_summary.to_csv(dataset_metric_summary_path, sep="\t", index=False)
    line_metric_summary.to_csv(line_metric_summary_path, sep="\t", index=False)
    print(f"Saved dataset metric summary to {dataset_metric_summary_path}")
    print(f"Saved dataset-line metric summary to {line_metric_summary_path}")

    condition_deg_columns = deg_metric_columns(condition_metric_summary)
    dataset_deg_columns = deg_metric_columns(dataset_metric_summary)
    line_deg_columns = deg_metric_columns(line_metric_summary)
    if expect_deg_metrics and not condition_deg_columns:
        raise ValueError(
            "DEG metrics were requested in the saved prepare config, but the merged task outputs "
            "do not contain DEG metric columns. Rerun from stage 1 so stage 2 writes DEG-enabled shards."
        )
    if condition_deg_columns:
        condition_deg_metric_summary = condition_metric_summary[
            [
                "dataset_name",
                "cell_type",
                "pubchem_cid",
                "time_key",
                "dose_key",
                "condition_key",
                "perturbagen_display",
                "n_replicates",
                "n_replicate_pairs",
                "n_source_files",
                "n_local_shared_genes",
                "n_baseline_peer_rows",
                "n_baseline_peer_compounds",
                *[
                    column_name
                    for column_name in ("n_peer_rows_total", "n_peer_rows_scored")
                    if column_name in condition_metric_summary.columns
                ],
                *condition_deg_columns,
            ]
        ].copy()
        condition_deg_metric_summary_path = output_dir / "condition_deg_metric_summary.tsv"
        condition_deg_metric_summary.to_csv(condition_deg_metric_summary_path, sep="\t", index=False)
        print(f"Saved condition-level DEG metric summary to {condition_deg_metric_summary_path}")
    else:
        for stale_path in [
            output_dir / "condition_deg_metric_summary.tsv",
            output_dir / "dataset_deg_metric_summary.tsv",
            output_dir / "dataset_line_deg_metric_summary.tsv",
        ]:
            if stale_path.exists():
                stale_path.unlink()
                print(f"Removed stale DEG summary file {stale_path}")
    if dataset_deg_columns:
        dataset_deg_metric_summary = dataset_metric_summary[
            [
                "dataset_name",
                "n_conditions",
                "n_unique_lines",
                "n_unique_compounds",
                "n_total_replicates",
                "n_total_replicate_pairs",
                *dataset_deg_columns,
            ]
        ].copy()
        dataset_deg_metric_summary_path = output_dir / "dataset_deg_metric_summary.tsv"
        dataset_deg_metric_summary.to_csv(dataset_deg_metric_summary_path, sep="\t", index=False)
        print(f"Saved dataset DEG metric summary to {dataset_deg_metric_summary_path}")
    if line_deg_columns:
        line_deg_metric_summary = line_metric_summary[
            [
                "dataset_name",
                "cell_type",
                "n_conditions",
                "n_unique_lines",
                "n_unique_compounds",
                "n_total_replicates",
                "n_total_replicate_pairs",
                *line_deg_columns,
            ]
        ].copy()
        line_deg_metric_summary_path = output_dir / "dataset_line_deg_metric_summary.tsv"
        line_deg_metric_summary.to_csv(line_deg_metric_summary_path, sep="\t", index=False)
        print(f"Saved dataset-line DEG metric summary to {line_deg_metric_summary_path}")

    dataset_t_strength_relationship_summary, line_t_strength_relationship_summary = (
        build_t_strength_relationship_summaries(
            condition_metric_summary,
            top_k=top_k,
        )
    )
    dataset_t_strength_relationship_summary_path = output_dir / "dataset_t_strength_relationship_summary.tsv"
    line_t_strength_relationship_summary_path = output_dir / "dataset_line_t_strength_relationship_summary.tsv"
    dataset_t_strength_relationship_summary.to_csv(
        dataset_t_strength_relationship_summary_path,
        sep="\t",
        index=False,
    )
    line_t_strength_relationship_summary.to_csv(
        line_t_strength_relationship_summary_path,
        sep="\t",
        index=False,
    )
    print(
        "Saved dataset t-strength relationship summary to "
        f"{dataset_t_strength_relationship_summary_path}"
    )
    print(
        "Saved dataset-line t-strength relationship summary to "
        f"{line_t_strength_relationship_summary_path}"
    )

    if expect_retrieval_metrics and not retrieval_condition_frames:
        raise ValueError(
            "Retrieval metrics were requested in the saved prepare config, but no task retrieval summaries "
            "were found. Rerun from stage 1 so stage 2 writes retrieval-enabled shards."
        )
    if retrieval_condition_frames:
        condition_retrieval_summary = pd.concat(retrieval_condition_frames, ignore_index=True)
        condition_retrieval_summary = drop_duplicate_retrieval_rows(condition_retrieval_summary)
        condition_retrieval_summary = coerce_non_identifier_columns_to_numeric(
            condition_retrieval_summary,
            identifier_columns={
                "dataset_name",
                "cell_type",
                "time_key",
                "representation",
                "condition_key",
                "pubchem_cid",
                "dose_key",
                "perturbagen_display",
            },
        )
        (
            condition_retrieval_summary,
            stratum_retrieval_summary,
            dataset_retrieval_summary,
            line_retrieval_summary,
        ) = (
            build_retrieval_final_summaries(
                condition_retrieval_summary,
                all_dataset_names=all_dataset_names,
            )
        )
        condition_retrieval_summary_path = output_dir / "condition_retrieval_summary.tsv"
        stratum_retrieval_summary_path = output_dir / "line_time_retrieval_summary.tsv"
        dataset_retrieval_summary_path = output_dir / "dataset_retrieval_summary.tsv"
        line_retrieval_summary_path = output_dir / "dataset_line_retrieval_summary.tsv"
        condition_retrieval_summary.to_csv(condition_retrieval_summary_path, sep="\t", index=False)
        stratum_retrieval_summary.to_csv(stratum_retrieval_summary_path, sep="\t", index=False)
        dataset_retrieval_summary.to_csv(dataset_retrieval_summary_path, sep="\t", index=False)
        line_retrieval_summary.to_csv(line_retrieval_summary_path, sep="\t", index=False)
        print(f"Saved condition retrieval summary to {condition_retrieval_summary_path}")
        print(f"Saved line-time retrieval summary to {stratum_retrieval_summary_path}")
        print(f"Saved dataset retrieval summary to {dataset_retrieval_summary_path}")
        print(f"Saved dataset-line retrieval summary to {line_retrieval_summary_path}")
    else:
        for stale_path in [
            output_dir / "condition_retrieval_summary.tsv",
            output_dir / "line_time_retrieval_summary.tsv",
            output_dir / "dataset_retrieval_summary.tsv",
            output_dir / "dataset_line_retrieval_summary.tsv",
        ]:
            if stale_path.exists():
                stale_path.unlink()
                print(f"Removed stale retrieval summary file {stale_path}")


def run_all(args: argparse.Namespace) -> None:
    prepare_result = prepare(
        output_dir=args.output_dir,
        dataset_arg=args.datasets,
        min_replicates_per_condition=args.min_replicates_per_condition,
        conditions_per_task=args.conditions_per_task,
        test_one_line_per_dataset=args.test_one_line_per_dataset,
        test_max_conditions_per_dataset=args.test_max_conditions_per_dataset,
        compute_baseline_metrics=args.compute_baseline_metrics,
        compute_deg_metrics=args.compute_deg_metrics,
        compute_retrieval_metrics=args.compute_retrieval_metrics,
        min_retrieval_compounds_per_line_time=args.min_retrieval_compounds_per_line_time,
        max_baseline_peers=args.max_baseline_peers,
        peer_sampling_seed=args.peer_sampling_seed,
    )
    manifest = load_task_manifest(prepare_result.task_manifest_path)
    n_tasks = int(len(manifest))
    if n_tasks == 0:
        raise ValueError("No task shards were prepared.")
    for task_idx in range(1, n_tasks + 1):
        if task_idx == 1 or task_idx % int(args.progress_every) == 0:
            print(f"Running prepared task {task_idx:,} / {n_tasks:,}")
        run_task(
            output_dir=args.output_dir,
            task_file=prepare_result.task_manifest_path,
            task_id=task_idx,
            task_output_dir_path=prepare_result.task_output_dir,
            top_k=args.top_k,
            compute_baseline_metrics=args.compute_baseline_metrics,
            compute_deg_metrics=args.compute_deg_metrics,
            compute_retrieval_metrics=args.compute_retrieval_metrics,
            min_retrieval_compounds_per_line_time=args.min_retrieval_compounds_per_line_time,
        )
    merge_task_outputs(
        output_dir=args.output_dir,
        task_file=prepare_result.task_manifest_path,
        task_output_dir_path=prepare_result.task_output_dir,
        top_k=args.top_k,
        strict_missing=True,
        existing_results_dir=None,
    )


def main() -> None:
    global MAX_BASELINE_PEERS, PEER_SAMPLING_SEED
    args = parse_args()
    # Read at call time inside the scoring functions, so setting it here covers every command.
    MAX_BASELINE_PEERS = getattr(args, "max_baseline_peers", None)
    PEER_SAMPLING_SEED = int(
        getattr(args, "peer_sampling_seed", DEFAULT_PEER_SAMPLING_SEED)
    )
    if MAX_BASELINE_PEERS is not None and int(MAX_BASELINE_PEERS) <= 0:
        raise SystemExit("--max-baseline-peers must be a positive integer when provided.")
    if args.command == "prepare":
        prepare(
            output_dir=args.output_dir,
            dataset_arg=args.datasets,
            min_replicates_per_condition=args.min_replicates_per_condition,
            conditions_per_task=args.conditions_per_task,
            test_one_line_per_dataset=args.test_one_line_per_dataset,
            test_max_conditions_per_dataset=args.test_max_conditions_per_dataset,
            compute_baseline_metrics=args.compute_baseline_metrics,
            compute_deg_metrics=args.compute_deg_metrics,
            compute_retrieval_metrics=args.compute_retrieval_metrics,
            min_retrieval_compounds_per_line_time=args.min_retrieval_compounds_per_line_time,
            max_baseline_peers=args.max_baseline_peers,
            peer_sampling_seed=args.peer_sampling_seed,
        )
        return
    if args.command == "reshard":
        reshard(
            output_dir=args.output_dir,
            conditions_per_task=args.conditions_per_task,
        )
        return
    if args.command == "run-task":
        run_task(
            output_dir=args.output_dir,
            task_file=args.task_file,
            task_id=args.task_id,
            task_output_dir_path=args.task_output_dir,
            top_k=args.top_k,
            compute_baseline_metrics=args.compute_baseline_metrics,
            compute_deg_metrics=args.compute_deg_metrics,
            compute_retrieval_metrics=args.compute_retrieval_metrics,
            min_retrieval_compounds_per_line_time=args.min_retrieval_compounds_per_line_time,
        )
        return
    if args.command == "merge":
        merge_task_outputs(
            output_dir=args.output_dir,
            task_file=args.task_file,
            task_output_dir_path=args.task_output_dir,
            top_k=args.top_k,
            strict_missing=args.strict_missing,
            existing_results_dir=args.existing_results_dir,
        )
        return
    run_all(args)


if __name__ == "__main__":
    main()
