"""Shared configuration, matching, source, and cache policy for cross-source analyses.

The three reviewer notebooks intentionally compute different metrics, but they must
agree on dataset policy, matched-condition identities, line-source handling, shared-gene
geometry, and cache invalidation.  This module is the single seam for that shared logic.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Hashable, Mapping, Optional, Sequence

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from scripts.cross_source_strata import SignatureStratum
from scripts.notebook_cache import cached_frame, stable_json_fingerprint


ANALYSIS_PROFILES = frozenset({"deg", "signature", "retrieval"})
NUMERIC_SIG_FIGS = 12
TOP_K = 50
INVALID_STRING_VALUES = {"", "nan", "none", "<na>"}


@dataclass(frozen=True)
class DatasetSpec:
    """One dataset's shared label, source location, and supported analyses."""

    name: str
    label: str
    relative_source_dir: tuple[str, ...]
    profiles: frozenset[str] = ANALYSIS_PROFILES


# Relative to --data-root, which on this machine is
# /lustre/groups/ml01/workspace/olga.novitskaia/data_updated.
# Tahoe / sci-Plex / OP3 use a higher min-cells filter than the rest; those
# suffixes are part of the published line-file trees, not a scoring choice.
_LINE_RESULTS = ("deg_data", "group_rep", "full", "qc_false")


def _line_source_dir(dataset: str, min_cells: int) -> tuple[str, ...]:
    return (dataset, *_LINE_RESULTS, f"filter_min_cells_{min_cells}", "results")


DATASET_SPECS = (
    DatasetSpec("l1000_phase1", "L1000 Phase I", _line_source_dir("l1000_phase1", 0)),
    DatasetSpec("l1000_phase2", "L1000 Phase II", _line_source_dir("l1000_phase2", 0)),
    DatasetSpec("tahoe", "Tahoe-100M", _line_source_dir("tahoe", 50)),
    DatasetSpec("cigs_mce", "CIGS-MCE", _line_source_dir("cigs_mce", 0)),
    DatasetSpec(
        "novartis_batch_2500",
        "Novartis/DRUG-seq U2OS",
        _line_source_dir("novartis_batch_2500", 0),
    ),
    DatasetSpec("vcpi_0001", "VCPI-0001", _line_source_dir("vcpi_0001", 0)),
    DatasetSpec("cigs_tcm", "CIGS-TCM", _line_source_dir("cigs_tcm", 0)),
    DatasetSpec("vcpi_0002", "VCPI-0002", _line_source_dir("vcpi_0002", 0)),
    # These sources are retained for Table 8. Their DEG/retrieval layer capability
    # should be verified before adding the corresponding profiles.
    DatasetSpec(
        "gdpx2",
        "GDPx2",
        _line_source_dir("gdpx2", 0),
        frozenset({"signature"}),
    ),
    DatasetSpec("sciplex", "sci-Plex", _line_source_dir("sciplex", 10)),
    DatasetSpec(
        "dilimap_train_val",
        "DILImap",
        _line_source_dir("dilimap_train_val", 0),
        frozenset({"signature"}),
    ),
    DatasetSpec("op3", "OP3", _line_source_dir("op3", 10)),
)

DATASET_SPEC_BY_NAME = {spec.name: spec for spec in DATASET_SPECS}
DISPLAY_LABELS = {spec.name: spec.label for spec in DATASET_SPECS}


def pretty_label(dataset_name: str) -> str:
    """Return the paper-facing label for a dataset key."""
    return DISPLAY_LABELS.get(str(dataset_name), str(dataset_name))


def _validate_profile(profile: str) -> str:
    profile = str(profile)
    if profile not in ANALYSIS_PROFILES:
        raise ValueError(
            f"Unknown cross-source analysis profile {profile!r}; "
            f"expected one of {sorted(ANALYSIS_PROFILES)}"
        )
    return profile


def production_dataset_order(profile: str) -> list[str]:
    """Return the canonical ordered dataset list for an analysis profile."""
    profile = _validate_profile(profile)
    return [spec.name for spec in DATASET_SPECS if profile in spec.profiles]


def source_dataset_dirs(data_root: Path, profile: str) -> dict[str, Path]:
    """Resolve every profile-supported line-level source directory."""
    data_root = Path(data_root)
    profile = _validate_profile(profile)
    return {
        spec.name: data_root.joinpath(*spec.relative_source_dir)
        for spec in DATASET_SPECS
        if profile in spec.profiles
    }


def selected_dataset_order(
    profile: str,
    dataset_subset: Optional[Sequence[str]],
) -> list[str]:
    """Apply an optional subset while preserving caller order and rejecting drift."""
    production_order = production_dataset_order(profile)
    if dataset_subset is None:
        return production_order
    selected = list(dict.fromkeys(str(value) for value in dataset_subset))
    unknown = sorted(set(selected) - set(production_order))
    if unknown:
        raise ValueError(
            f"Datasets unsupported by the {profile!r} profile: {unknown}. "
            f"Supported datasets: {production_order}"
        )
    if len(selected) < 2:
        raise ValueError("Cross-dataset analysis requires at least two selected datasets.")
    return selected


def analysis_output_dir(
    production_output_dir: Path,
    *,
    dataset_subset: Optional[Sequence[str]],
    dataset_names: Sequence[str],
    run_tag: str = "",
) -> Path:
    """Return the shared production/run-tag/subset output layout."""
    production_output_dir = Path(production_output_dir)
    run_tag = str(run_tag).strip()
    if dataset_subset is None and not run_tag:
        return production_output_dir
    if dataset_subset is None:
        return (
            production_output_dir.parent
            / "production_runs"
            / production_output_dir.name
            / run_tag
        )
    subset_run_label = run_tag or "__".join(str(name) for name in dataset_names)
    return (
        production_output_dir.parent
        / "subset_runs"
        / production_output_dir.name
        / subset_run_label
    )


def format_numeric(value: float, sig_figs: int = NUMERIC_SIG_FIGS) -> str:
    formatted = f"{value:.{int(sig_figs)}g}"
    return "0" if formatted == "-0" else formatted


def sanitize_string_values(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").fillna("").astype(str).str.strip()
    normalized.loc[normalized.str.lower().isin(INVALID_STRING_VALUES)] = ""
    return normalized


def format_pubchem_cid(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return format_numeric(float(value))


def normalize_pubchem_cid_values(series: pd.Series) -> pd.Series:
    normalized = sanitize_string_values(series)
    numeric = pd.to_numeric(normalized, errors="coerce")
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float))
    if not finite_mask.any():
        return normalized
    normalized = normalized.copy()
    normalized.loc[finite_mask] = numeric.loc[finite_mask].map(format_pubchem_cid)
    return normalized


def coerce_control_mask(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = (
        values.astype("string").fillna("").astype(str).str.strip().str.lower()
    )
    return normalized.isin({"true", "1", "yes"})


EMPTY_OVERLAP_FRAME = pd.DataFrame(
    columns=[
        "dataset_name",
        "obs_id",
        "plate",
        "well",
        "pubchem_cid",
        "cell_type",
        "pert_time_h",
        "pert_dose_uM",
        "time_key",
        "dose_key",
        "log10_dose",
    ]
)


def load_overlap_obs(dataset_name: str, overlap_dir: Path) -> pd.DataFrame:
    h5ad_path = Path(overlap_dir) / f"{dataset_name}_overlap_filtered.h5ad"
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing overlap file: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        obs = adata.obs.copy()
    finally:
        adata.file.close()
    if obs.empty:
        return EMPTY_OVERLAP_FRAME.copy()
    if "is_control" not in obs.columns:
        raise KeyError(f"{h5ad_path} is missing obs['is_control']")
    obs = obs.loc[~coerce_control_mask(obs["is_control"])].copy()
    if obs.empty:
        return EMPTY_OVERLAP_FRAME.copy()

    frame = pd.DataFrame(index=obs.index.copy())
    frame["dataset_name"] = str(dataset_name)
    frame["obs_id"] = frame.index.astype(str)
    frame["plate"] = (
        obs["plate"].astype("string").fillna("").astype(str).str.strip()
        if "plate" in obs.columns
        else ""
    )
    frame["well"] = (
        obs["well"].astype("string").fillna("").astype(str).str.strip()
        if "well" in obs.columns
        else ""
    )
    context_column = (
        "harmonized_context_key"
        if "harmonized_context_key" in obs.columns
        else "cell_type"
    )
    frame["pubchem_cid"] = normalize_pubchem_cid_values(obs["pubchem_cid"])
    frame["cell_type"] = sanitize_string_values(obs[context_column])
    frame["pert_time_h"] = pd.to_numeric(obs["pert_time_h"], errors="coerce")
    frame["pert_dose_uM"] = pd.to_numeric(obs["pert_dose_uM"], errors="coerce")
    return ensure_overlap_frame_schema(frame, str(dataset_name))


def ensure_overlap_frame_schema(
    frame: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return EMPTY_OVERLAP_FRAME.copy()
    frame = frame.copy()
    if "obs_id" not in frame.columns:
        frame["obs_id"] = pd.Index(frame.index).astype(str)
    for column_name in ["obs_id", "plate", "well"]:
        if column_name not in frame.columns:
            frame[column_name] = ""
        frame[column_name] = sanitize_string_values(frame[column_name])
    required = ["pubchem_cid", "cell_type", "pert_time_h", "pert_dose_uM"]
    missing = [column_name for column_name in required if column_name not in frame]
    if missing:
        raise KeyError(
            f"Overlap frame for {dataset_name} is missing required columns: {missing}"
        )
    frame["pubchem_cid"] = normalize_pubchem_cid_values(frame["pubchem_cid"])
    frame["cell_type"] = sanitize_string_values(frame["cell_type"])
    frame["dataset_name"] = str(dataset_name)
    frame["pert_time_h"] = pd.to_numeric(frame["pert_time_h"], errors="coerce")
    frame["pert_dose_uM"] = pd.to_numeric(frame["pert_dose_uM"], errors="coerce")
    valid_mask = (
        (frame["pubchem_cid"] != "")
        & (frame["cell_type"] != "")
        & np.isfinite(frame["pert_time_h"].to_numpy(dtype=float))
        & np.isfinite(frame["pert_dose_uM"].to_numpy(dtype=float))
        & (frame["pert_dose_uM"].to_numpy(dtype=float) > 0.0)
    )
    frame = frame.loc[valid_mask].copy().reset_index(drop=True)
    if frame.empty:
        return EMPTY_OVERLAP_FRAME.copy()
    frame["time_key"] = frame["pert_time_h"].map(
        lambda value: format_numeric(float(value))
    )
    frame["dose_key"] = frame["pert_dose_uM"].map(
        lambda value: format_numeric(float(value))
    )
    frame["log10_dose"] = np.log10(
        frame["pert_dose_uM"].to_numpy(dtype=np.float64)
    )
    return frame[EMPTY_OVERLAP_FRAME.columns.tolist()]


def build_groups(frame: pd.DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    if frame.empty:
        return {}
    grouped = frame.groupby(
        ["pubchem_cid", "cell_type", "time_key"],
        sort=False,
    ).groups
    return {
        key: np.asarray(list(row_positions), dtype=np.int64)
        for key, row_positions in grouped.items()
    }


def build_dataset_index(
    dataset_name: str,
    overlap_dir: Path,
) -> dict[str, object]:
    frame = ensure_overlap_frame_schema(
        load_overlap_obs(dataset_name, overlap_dir),
        dataset_name,
    )
    return {"frame": frame, "groups": build_groups(frame)}


def build_dataset_indices(
    dataset_names: Sequence[str],
    overlap_dir: Path,
) -> dict[str, dict[str, object]]:
    return {
        str(dataset_name): build_dataset_index(str(dataset_name), overlap_dir)
        for dataset_name in dataset_names
    }


def active_dataset_names(
    dataset_indices: Mapping[str, Mapping[str, object]],
    dataset_order: Sequence[str],
) -> list[str]:
    return [
        str(dataset_name)
        for dataset_name in dataset_order
        if dataset_name in dataset_indices
        and not dataset_indices[dataset_name]["frame"].empty
    ]


@dataclass(frozen=True)
class MatchSettings:
    max_dose_fold_difference: float = 10.0
    min_context_shared_drugs: int = 10

    def __post_init__(self) -> None:
        if not np.isfinite(self.max_dose_fold_difference):
            raise ValueError("max_dose_fold_difference must be finite")
        if self.max_dose_fold_difference < 1.0:
            raise ValueError("max_dose_fold_difference must be at least 1")
        if int(self.min_context_shared_drugs) < 1:
            raise ValueError("min_context_shared_drugs must be positive")

    @property
    def max_log10_dose_diff(self) -> float:
        return float(np.log10(self.max_dose_fold_difference))


DEFAULT_MATCH_SETTINGS = MatchSettings()

MATCH_PAIR_IDENTITY_COLUMNS = [
    "dataset_a",
    "dataset_b",
    "cell_type",
    "pubchem_cid",
    "time_key",
    "left_obs_id",
    "right_obs_id",
    "left_dose_key",
    "right_dose_key",
]

MATCH_PAIR_COLUMNS = [
    "dataset_a",
    "dataset_b",
    "cell_type",
    "pubchem_cid",
    "time_key",
    "left_obs_id",
    "right_obs_id",
    "left_plate",
    "right_plate",
    "left_well",
    "right_well",
    "left_dose_key",
    "right_dose_key",
    "left_dose_uM",
    "right_dose_uM",
    "dose_fold_difference",
    "left_log10_dose",
    "right_log10_dose",
    "abs_delta_log10_dose",
    "matched_condition_key",
    "n_context_matching_drugs",
]


def raw_dose_fold_difference_matrix(
    left_doses_uM: np.ndarray,
    right_doses_uM: np.ndarray,
) -> np.ndarray:
    left = np.asarray(left_doses_uM, dtype=np.float64)
    right = np.asarray(right_doses_uM, dtype=np.float64)
    if (
        left.ndim != 1
        or right.ndim != 1
        or np.any(~np.isfinite(left))
        or np.any(~np.isfinite(right))
        or np.any(left <= 0.0)
        or np.any(right <= 0.0)
    ):
        raise ValueError("Raw doses must be finite, positive one-dimensional arrays.")
    return np.maximum(
        left[:, None] / right[None, :],
        right[None, :] / left[:, None],
    )


def mutual_nearest_dose_pairs(
    left_doses_uM: np.ndarray,
    right_doses_uM: np.ndarray,
    *,
    max_dose_fold_difference: float = 10.0,
) -> np.ndarray:
    if left_doses_uM.size == 0 or right_doses_uM.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    fold_difference = raw_dose_fold_difference_matrix(
        left_doses_uM,
        right_doses_uM,
    )
    left_min = fold_difference.min(axis=1, keepdims=True)
    right_min = fold_difference.min(axis=0, keepdims=True)
    is_mnn = (
        (fold_difference <= float(max_dose_fold_difference) + 1e-12)
        & np.isclose(fold_difference, left_min, rtol=0.0, atol=1e-12)
        & np.isclose(fold_difference, right_min, rtol=0.0, atol=1e-12)
    )
    return np.argwhere(is_mnn)


def pair_match_frame(
    left_dataset: str,
    right_dataset: str,
    left_index: Mapping[str, object],
    right_index: Mapping[str, object],
    *,
    settings: MatchSettings = DEFAULT_MATCH_SETTINGS,
) -> pd.DataFrame:
    left_frame = ensure_overlap_frame_schema(
        left_index["frame"],
        str(left_dataset),
    )
    right_frame = ensure_overlap_frame_schema(
        right_index["frame"],
        str(right_dataset),
    )
    left_groups = left_index["groups"]
    right_groups = right_index["groups"]
    if set(build_groups(left_frame)) != set(left_groups):
        left_groups = build_groups(left_frame)
    if set(build_groups(right_frame)) != set(right_groups):
        right_groups = build_groups(right_frame)
    if left_frame.empty or right_frame.empty:
        return pd.DataFrame(columns=MATCH_PAIR_COLUMNS)

    shared_keys = sorted(set(left_groups) & set(right_groups))
    if not shared_keys:
        return pd.DataFrame(columns=MATCH_PAIR_COLUMNS)

    left_obs_ids = left_frame["obs_id"].to_numpy(dtype=object)
    right_obs_ids = right_frame["obs_id"].to_numpy(dtype=object)
    left_plates = left_frame["plate"].to_numpy(dtype=object)
    right_plates = right_frame["plate"].to_numpy(dtype=object)
    left_wells = left_frame["well"].to_numpy(dtype=object)
    right_wells = right_frame["well"].to_numpy(dtype=object)
    left_doses_uM = left_frame["pert_dose_uM"].to_numpy(dtype=np.float64)
    right_doses_uM = right_frame["pert_dose_uM"].to_numpy(dtype=np.float64)
    left_log10_dose = left_frame["log10_dose"].to_numpy(dtype=np.float64)
    right_log10_dose = right_frame["log10_dose"].to_numpy(dtype=np.float64)
    left_dose_keys = left_frame["dose_key"].to_numpy(dtype=object)
    right_dose_keys = right_frame["dose_key"].to_numpy(dtype=object)

    context_matching_drugs: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows: list[dict[str, object]] = []
    for pubchem_cid, cell_type, time_key in shared_keys:
        left_rows = left_groups[(pubchem_cid, cell_type, time_key)]
        right_rows = right_groups[(pubchem_cid, cell_type, time_key)]
        pairs = mutual_nearest_dose_pairs(
            left_doses_uM[left_rows],
            right_doses_uM[right_rows],
            max_dose_fold_difference=settings.max_dose_fold_difference,
        )
        for left_pos, right_pos in pairs:
            left_row = int(left_rows[left_pos])
            right_row = int(right_rows[right_pos])
            left_dose_key = str(left_dose_keys[left_row])
            right_dose_key = str(right_dose_keys[right_row])
            left_dose_uM = float(left_doses_uM[left_row])
            right_dose_uM = float(right_doses_uM[right_row])
            context_matching_drugs[(str(cell_type), str(time_key))].add(
                str(pubchem_cid)
            )
            rows.append(
                {
                    "dataset_a": str(left_dataset),
                    "dataset_b": str(right_dataset),
                    "cell_type": str(cell_type),
                    "pubchem_cid": str(pubchem_cid),
                    "time_key": str(time_key),
                    "left_obs_id": str(left_obs_ids[left_row]),
                    "right_obs_id": str(right_obs_ids[right_row]),
                    "left_plate": str(left_plates[left_row]),
                    "right_plate": str(right_plates[right_row]),
                    "left_well": str(left_wells[left_row]),
                    "right_well": str(right_wells[right_row]),
                    "left_dose_key": left_dose_key,
                    "right_dose_key": right_dose_key,
                    "left_dose_uM": left_dose_uM,
                    "right_dose_uM": right_dose_uM,
                    "dose_fold_difference": float(
                        max(
                            left_dose_uM / right_dose_uM,
                            right_dose_uM / left_dose_uM,
                        )
                    ),
                    "left_log10_dose": float(left_log10_dose[left_row]),
                    "right_log10_dose": float(right_log10_dose[right_row]),
                    "abs_delta_log10_dose": float(
                        abs(
                            left_log10_dose[left_row]
                            - right_log10_dose[right_row]
                        )
                    ),
                    "matched_condition_key": "|".join(
                        [
                            str(pubchem_cid),
                            str(cell_type),
                            str(time_key),
                            left_dose_key,
                            right_dose_key,
                        ]
                    ),
                }
            )
    if not rows:
        return pd.DataFrame(columns=MATCH_PAIR_COLUMNS)

    frame = pd.DataFrame(rows)
    qualifying_counts = {
        context_key: len(compounds)
        for context_key, compounds in context_matching_drugs.items()
        if len(compounds) >= int(settings.min_context_shared_drugs)
    }
    if not qualifying_counts:
        return pd.DataFrame(columns=MATCH_PAIR_COLUMNS)
    frame["_context_key"] = list(zip(frame["cell_type"], frame["time_key"]))
    frame["n_context_matching_drugs"] = frame["_context_key"].map(
        qualifying_counts
    )
    frame = frame.loc[frame["n_context_matching_drugs"].notna()].copy()
    frame["n_context_matching_drugs"] = frame[
        "n_context_matching_drugs"
    ].astype(int)
    return frame.drop(columns="_context_key")[MATCH_PAIR_COLUMNS].reset_index(
        drop=True
    )


def build_matched_pairs(
    dataset_indices: Mapping[str, Mapping[str, object]],
    dataset_names: Sequence[str],
    *,
    settings: MatchSettings = DEFAULT_MATCH_SETTINGS,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for dataset_a, dataset_b in itertools.combinations(dataset_names, 2):
        frame = pair_match_frame(
            str(dataset_a),
            str(dataset_b),
            dataset_indices[str(dataset_a)],
            dataset_indices[str(dataset_b)],
            settings=settings,
        )
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("No matched grouped-replicate sample pairs were found.")
    return pd.concat(frames, ignore_index=True)


def summarize_matched_pairs(
    matched_pairs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the canonical dataset-pair and dataset-pair-line coverage tables."""
    required = {
        "dataset_a",
        "dataset_b",
        "cell_type",
        "pubchem_cid",
        "matched_condition_key",
        "left_obs_id",
    }
    missing = sorted(required - set(matched_pairs.columns))
    if missing:
        raise KeyError(f"matched_pairs is missing summary columns: {missing}")
    pair_summary = (
        matched_pairs.groupby(["dataset_a", "dataset_b"], as_index=False)
        .agg(
            n_matched_sample_pairs=("left_obs_id", "size"),
            n_matching_drugs=("pubchem_cid", "nunique"),
            n_matching_lines=("cell_type", "nunique"),
            n_matching_conditions=("matched_condition_key", "nunique"),
        )
    )
    line_summary = (
        matched_pairs.groupby(
            ["dataset_a", "dataset_b", "cell_type"],
            as_index=False,
        )
        .agg(
            n_matched_sample_pairs=("left_obs_id", "size"),
            n_matching_drugs=("pubchem_cid", "nunique"),
            n_matching_conditions=("matched_condition_key", "nunique"),
        )
        .sort_values(["dataset_a", "dataset_b", "cell_type"])
        .reset_index(drop=True)
    )
    return pair_summary, line_summary


def _dataset_names(dataset_names: Sequence[str]) -> list[str]:
    normalized = [str(dataset_name) for dataset_name in dataset_names]
    if len(set(normalized)) != len(normalized):
        raise ValueError("dataset_names must be unique")
    return normalized


def _require_match_columns(matched_pairs: pd.DataFrame) -> None:
    required = {"dataset_a", "dataset_b", "cell_type"}
    missing = sorted(required - set(matched_pairs.columns))
    if missing:
        raise KeyError(f"matched_pairs is missing required columns: {missing}")


def matched_dataset_lines(
    matched_pairs: pd.DataFrame,
    dataset_names: Sequence[str],
) -> dict[str, list[str]]:
    normalized_names = _dataset_names(dataset_names)
    _require_match_columns(matched_pairs)
    lines = {dataset_name: set() for dataset_name in normalized_names}
    for dataset_a, dataset_b, cell_type in matched_pairs[
        ["dataset_a", "dataset_b", "cell_type"]
    ].drop_duplicates().itertuples(index=False, name=None):
        for dataset_name in (str(dataset_a), str(dataset_b)):
            if dataset_name not in lines:
                raise ValueError(
                    f"Matched-pair dataset {dataset_name!r} is outside dataset_names"
                )
            lines[dataset_name].add(str(cell_type))
    return {
        dataset_name: sorted(lines[dataset_name])
        for dataset_name in normalized_names
    }


def matched_dataset_names(
    matched_lines: Mapping[str, Sequence[str]],
    dataset_names: Sequence[str],
) -> list[str]:
    """Return active datasets that actually contribute a retained match."""
    return [
        str(dataset_name)
        for dataset_name in dataset_names
        if matched_lines.get(str(dataset_name), ())
    ]


def global_gene_scope_lines(
    available_lines: Mapping[str, Sequence[str]],
    matched_pairs: pd.DataFrame,
    dataset_names: Sequence[str],
) -> dict[str, list[str]]:
    """Skip unused cell types while preserving their historical intersection."""
    normalized_names = _dataset_names(dataset_names)
    _require_match_columns(matched_pairs)
    used_cell_types = {
        str(cell_type)
        for cell_type in matched_pairs["cell_type"].dropna().unique().tolist()
    }
    return {
        dataset_name: sorted(
            {
                str(cell_type)
                for cell_type in available_lines.get(dataset_name, ())
                if str(cell_type) in used_cell_types
            }
        )
        for dataset_name in normalized_names
    }


def file_inventory(paths: Sequence[Path]) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for path in sorted({Path(path).resolve() for path in paths}, key=str):
        stat = path.stat()
        inventory.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return inventory


def line_source_inventory(
    dataset_lines: Mapping[str, Sequence[str]],
    resolve_line_path: Callable[[str, str], Path],
) -> list[dict[str, object]]:
    inventory: list[dict[str, object]] = []
    for dataset_name in sorted(dataset_lines):
        for cell_type in sorted(set(dataset_lines[dataset_name])):
            path = Path(resolve_line_path(str(dataset_name), str(cell_type)))
            stat = path.stat()
            inventory.append(
                {
                    "dataset": str(dataset_name),
                    "cell_type": str(cell_type),
                    "path": str(path.resolve()),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
    return inventory


def matched_pairs_fingerprint(
    dataset_names: Sequence[str],
    overlap_dir: Path,
    *,
    settings: MatchSettings = DEFAULT_MATCH_SETTINGS,
    version: str = "cross-source-matches-v1",
) -> str:
    overlap_paths = [
        Path(overlap_dir) / f"{dataset_name}_overlap_filtered.h5ad"
        for dataset_name in dataset_names
    ]
    return stable_json_fingerprint(
        {
            "version": str(version),
            "datasets": list(dataset_names),
            "max_dose_fold_difference": settings.max_dose_fold_difference,
            "min_context_shared_drugs": settings.min_context_shared_drugs,
            "overlap_sources": file_inventory(overlap_paths),
        }
    )


def ci_fingerprint(
    *,
    version: str,
    upstream_fingerprint: str,
    n_boot: int,
    seed: int,
    metric_columns: Sequence[str] = (),
    group_columns: Sequence[str] = (),
) -> str:
    return stable_json_fingerprint(
        {
            "version": str(version),
            "upstream_fingerprint": str(upstream_fingerprint),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "metric_columns": list(metric_columns),
            "group_columns": list(group_columns),
        }
    )


@dataclass(frozen=True)
class CrossSourceScope:
    """Canonical matched-pair and shared-gene scope for one analysis run."""

    dataset_indices: Mapping[str, Mapping[str, object]]
    active_datasets: list[str]
    retained_lines: Mapping[str, list[str]]
    matched_pairs: pd.DataFrame
    matched_pairs_fingerprint: str
    matched_lines: Mapping[str, list[str]]
    matched_active_datasets: list[str]
    global_gene_lines: Mapping[str, list[str]]
    line_global_gene_keys: Mapping[str, np.ndarray]
    pair_match_summary: pd.DataFrame
    line_match_summary: pd.DataFrame


@dataclass(frozen=True)
class MatchedPairShard:
    """One deterministic row chunk within a scale/context scoring plan."""

    key: tuple[str, ...]
    scale_variant: str
    context_key: tuple[str, ...]
    row_start: int
    row_stop: int


@dataclass
class MatchedPairShardPlan:
    """Partition matched-pair contexts into deterministic resumable chunks."""

    context_columns: tuple[str, ...]
    groups: dict[tuple[str, ...], pd.DataFrame]
    shards: list[MatchedPairShard]
    max_rows: int

    @classmethod
    def build(
        cls,
        pairs: pd.DataFrame,
        *,
        context_columns: Sequence[str],
        scale_variants: Sequence[str],
        max_rows: int,
    ) -> "MatchedPairShardPlan":
        columns = tuple(str(column) for column in context_columns)
        missing = sorted(set(columns) - set(pairs.columns))
        if missing:
            raise KeyError(f"Matched-pair shard plan is missing columns: {missing}")
        max_rows = int(max_rows)
        if max_rows < 1:
            raise ValueError("Matched-pair shard max_rows must be positive")

        groups = {
            tuple(str(value) for value in context_key): group.copy()
            for context_key, group in pairs.groupby(list(columns), sort=False)
        }
        shards: list[MatchedPairShard] = []
        for scale_variant in scale_variants:
            normalized_scale = str(scale_variant)
            for context_key, group in groups.items():
                for row_start in range(0, len(group), max_rows):
                    row_stop = min(row_start + max_rows, len(group))
                    shards.append(
                        MatchedPairShard(
                            key=(
                                normalized_scale,
                                *context_key,
                                f"rows={row_start}:{row_stop}",
                            ),
                            scale_variant=normalized_scale,
                            context_key=context_key,
                            row_start=row_start,
                            row_stop=row_stop,
                        )
                    )
        return cls(
            context_columns=columns,
            groups=groups,
            shards=shards,
            max_rows=max_rows,
        )

    def frame(self, shard: MatchedPairShard) -> pd.DataFrame:
        group = self.groups[shard.context_key]
        return group.iloc[shard.row_start : shard.row_stop]

    def context_size(self, shard: MatchedPairShard) -> int:
        return int(len(self.groups[shard.context_key]))


def prepare_cross_source_scope(
    *,
    dataset_order: Sequence[str],
    overlap_dir: Path,
    output_dir: Path,
    source_catalog: "LineSourceCatalog",
    settings: MatchSettings = DEFAULT_MATCH_SETTINGS,
) -> CrossSourceScope:
    """Load or build the shared matched-pair scope used by every notebook.

    Matching is cached before any line-level source is opened. Only cell types that
    participate in at least one retained match are then loaded to construct the
    historical across-source shared-gene universe.
    """
    normalized_order = _dataset_names(dataset_order)
    overlap_dir = Path(overlap_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_indices = build_dataset_indices(normalized_order, overlap_dir)
    active_datasets = active_dataset_names(dataset_indices, normalized_order)
    if not active_datasets:
        raise ValueError(
            "No overlap-filtered non-control samples were found for the "
            "configured datasets."
        )

    retained_lines = {
        dataset_name: sorted(
            dataset_indices[dataset_name]["frame"]["cell_type"]
            .unique()
            .tolist()
        )
        for dataset_name in active_datasets
    }
    fingerprint = matched_pairs_fingerprint(
        active_datasets,
        overlap_dir,
        settings=settings,
    )

    def build_scope_matched_pairs() -> pd.DataFrame:
        return build_matched_pairs(
            dataset_indices,
            active_datasets,
            settings=settings,
        )

    matched_pairs = cached_frame(
        "matched_pairs",
        output_dir / "matched_sample_pairs.tsv",
        build_scope_matched_pairs,
        fingerprint=fingerprint,
        required_columns=MATCH_PAIR_COLUMNS,
    )
    matched_lines = matched_dataset_lines(matched_pairs, active_datasets)
    matched_active_datasets = matched_dataset_names(
        matched_lines,
        active_datasets,
    )
    global_gene_lines = global_gene_scope_lines(
        retained_lines,
        matched_pairs,
        active_datasets,
    )
    line_global_gene_keys = source_catalog.set_global_shared_gene_keys(
        global_gene_lines,
        active_datasets,
    )
    pair_summary, line_summary = summarize_matched_pairs(matched_pairs)
    return CrossSourceScope(
        dataset_indices=dataset_indices,
        active_datasets=active_datasets,
        retained_lines=retained_lines,
        matched_pairs=matched_pairs,
        matched_pairs_fingerprint=fingerprint,
        matched_lines=matched_lines,
        matched_active_datasets=matched_active_datasets,
        global_gene_lines=global_gene_lines,
        line_global_gene_keys=line_global_gene_keys,
        pair_match_summary=pair_summary,
        line_match_summary=line_summary,
    )


@dataclass
class LineSource:
    dataset_name: str
    cell_type: str
    path: Path
    adata: ad.AnnData = field(init=False, repr=False)
    obs: pd.DataFrame = field(init=False, repr=False)
    unique_gene_keys: np.ndarray = field(init=False, repr=False)
    unique_gene_positions: np.ndarray = field(init=False, repr=False)
    gene_to_pos: dict[str, int] = field(init=False, repr=False)
    lookup_row_pos: dict[str, int] = field(init=False, repr=False)
    _vector_cache: dict[tuple[str, int], np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _baseline_cache: dict[tuple[str, str, str, str], object] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _baseline_peer_counts: dict[tuple[str, str, str], int] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _stratum_cache: dict[tuple[str, str, str], SignatureStratum] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.adata = ad.read_h5ad(self.path, backed="r")
        obs = self.adata.obs.copy()
        row_positions = np.arange(self.adata.n_obs, dtype=np.int64)
        if "is_control" not in obs.columns:
            raise KeyError(f"{self.path} is missing obs['is_control']")

        obs["source_index"] = obs.index.astype(str)
        control_mask = coerce_control_mask(obs["is_control"]).to_numpy(dtype=bool)
        obs = obs.loc[~control_mask].copy()
        row_positions = row_positions[~control_mask]
        obs["source_row_pos"] = row_positions
        for column_name in [
            "id",
            "plate",
            "well",
            "cell_type",
            "perturbagen",
            "perturbagen_name",
            "perturbation_label",
            "pubchem_cid",
        ]:
            if column_name in obs.columns:
                obs[column_name] = (
                    obs[column_name]
                    .astype("string")
                    .fillna("")
                    .astype(str)
                    .str.strip()
                )
        if "pubchem_cid" in obs.columns:
            obs["pubchem_cid"] = normalize_pubchem_cid_values(
                obs["pubchem_cid"]
            )
        obs["pert_time_h"] = pd.to_numeric(obs["pert_time_h"], errors="coerce")
        obs["pert_dose_uM"] = pd.to_numeric(
            obs["pert_dose_uM"],
            errors="coerce",
        )
        obs["time_key"] = obs["pert_time_h"].map(
            lambda value: format_numeric(float(value))
            if pd.notna(value)
            else ""
        )
        obs["dose_key"] = obs["pert_dose_uM"].map(
            lambda value: format_numeric(float(value))
            if pd.notna(value) and float(value) > 0
            else ""
        )
        valid_mask = (
            (obs["pubchem_cid"] != "")
            & np.isfinite(obs["pert_time_h"].to_numpy(dtype=float))
            & np.isfinite(obs["pert_dose_uM"].to_numpy(dtype=float))
            & (obs["pert_dose_uM"].to_numpy(dtype=float) > 0.0)
        )
        obs = obs.loc[valid_mask].copy()
        obs = obs.set_index("source_row_pos", drop=False)
        self.obs = obs
        self.lookup_row_pos = self._build_lookup_row_pos()

        var = self.adata.var.copy()
        if "symbol" in var.columns:
            keys = (
                var["symbol"]
                .astype("string")
                .fillna("")
                .astype(str)
                .str.strip()
                .to_numpy()
            )
        else:
            keys = (
                pd.Index(self.adata.var_names.astype(str))
                .astype(str)
                .str.strip()
                .to_numpy()
            )
        gene_key_series = pd.Series(
            keys,
            index=np.arange(self.adata.n_vars, dtype=np.int64),
        )
        keep_mask = (gene_key_series != "") & ~gene_key_series.duplicated(
            keep="first"
        )
        self.unique_gene_positions = gene_key_series.index[
            keep_mask
        ].to_numpy(dtype=np.int64)
        self.unique_gene_keys = gene_key_series.loc[keep_mask].to_numpy(
            dtype=object
        )
        self.gene_to_pos = {
            str(gene_key): int(pos)
            for pos, gene_key in enumerate(self.unique_gene_keys.tolist())
        }

    def _build_lookup_row_pos(self) -> dict[str, int]:
        lookup_row_pos: dict[str, int] = {}

        def add_lookup_key(key: str, row_pos: int) -> None:
            if key and key.lower() != "nan":
                lookup_row_pos.setdefault(key, row_pos)

        for row_pos, row in self.obs.iterrows():
            row_pos = int(row_pos)
            for column_name in ["source_index", "id"]:
                if column_name in row.index:
                    add_lookup_key(str(row[column_name]).strip(), row_pos)
            cell_type = str(row.get("cell_type", "")).strip()
            pubchem_cid = str(row.get("pubchem_cid", "")).strip()
            dose_key = str(row.get("dose_key", "")).strip()
            time_key = str(row.get("time_key", "")).strip()
            if pubchem_cid and dose_key and time_key and cell_type:
                add_lookup_key(
                    f"{pubchem_cid}|{dose_key}|{time_key}|{cell_type}",
                    row_pos,
                )
        return lookup_row_pos

    def resolve_row_pos(
        self,
        obs_id: str,
        *,
        pubchem_cid=None,
        dose_key=None,
        time_key=None,
        plate=None,
        well=None,
    ) -> int:
        del plate, well
        candidate_keys = [str(obs_id)]
        pubchem_cid = "" if pubchem_cid is None else str(pubchem_cid).strip()
        dose_key = "" if dose_key is None else str(dose_key).strip()
        time_key = "" if time_key is None else str(time_key).strip()
        if pubchem_cid and dose_key and time_key:
            candidate_keys.append(
                f"{pubchem_cid}|{dose_key}|{time_key}|{self.cell_type}"
            )
        for candidate_key in candidate_keys:
            if candidate_key in self.lookup_row_pos:
                return int(self.lookup_row_pos[candidate_key])
        raise KeyError(
            f"Could not resolve obs_id={obs_id!r} in {self.path}. "
            f"Tried {candidate_keys!r}. Available lookup keys: "
            f"{len(self.lookup_row_pos):,}"
        )

    def get_vector(self, obs_id: str, layer_name: str, **lookup) -> np.ndarray:
        row_pos = self.resolve_row_pos(obs_id, **lookup)
        cache_key = (str(layer_name), row_pos)
        if cache_key not in self._vector_cache:
            if len(self._vector_cache) >= 512:
                self._vector_cache.pop(next(iter(self._vector_cache)))
            vector = np.asarray(
                self.adata.layers[layer_name][row_pos],
                dtype=np.float32,
            ).reshape(-1)
            self._vector_cache[cache_key] = vector[self.unique_gene_positions]
        return self._vector_cache[cache_key]

    def get_value_stratum(
        self,
        layer_name: str,
        dose_key: str,
        time_key: str,
    ) -> SignatureStratum:
        cache_key = (str(layer_name), str(dose_key), str(time_key))
        if cache_key not in self._stratum_cache:
            if any(key[1:] != cache_key[1:] for key in self._stratum_cache):
                self._stratum_cache.clear()
            context_obs = self.obs.loc[
                (self.obs["dose_key"] == cache_key[1])
                & (self.obs["time_key"] == cache_key[2])
            ]
            rows = context_obs["source_row_pos"].to_numpy(dtype=np.int64)
            matrix = np.asarray(
                self.adata.layers[layer_name][rows],
                dtype=np.float32,
            )
            if matrix.ndim == 1:
                matrix = matrix[np.newaxis, :]
            self._stratum_cache[cache_key] = SignatureStratum(
                context_obs["pubchem_cid"].astype(str).to_numpy(),
                matrix[:, self.unique_gene_positions],
                gene_keys=np.asarray(self.unique_gene_keys).astype(str),
            )
        return self._stratum_cache[cache_key]

    def get_baseline_vector(
        self,
        obs_id: str,
        layer_name: str,
        **lookup,
    ):
        row_pos = self.resolve_row_pos(obs_id, **lookup)
        row = self.obs.loc[row_pos]
        context_key = (
            str(row["dose_key"]),
            str(row["time_key"]),
            str(row["pubchem_cid"]),
        )
        cache_key = (str(layer_name), *context_key)
        if cache_key not in self._baseline_cache:
            if len(self._baseline_cache) >= 512:
                self._baseline_cache.pop(next(iter(self._baseline_cache)))
            stratum = self.get_value_stratum(
                layer_name,
                str(row["dose_key"]),
                str(row["time_key"]),
            )
            peer_count = stratum.n_rows - len(
                stratum.compound_row_indices(str(row["pubchem_cid"]))
            )
            self._baseline_peer_counts[context_key] = int(peer_count)
            if peer_count == 0:
                self._baseline_cache[cache_key] = None
            else:
                self._baseline_cache[cache_key] = np.asarray(
                    stratum.different_compound_centroid(
                        str(row["pubchem_cid"]),
                        require_all_finite=True,
                    ),
                    dtype=np.float32,
                )
        return self._baseline_cache[cache_key]

    def baseline_peer_count(self, obs_id: str, **lookup) -> int:
        row_pos = self.resolve_row_pos(obs_id, **lookup)
        row = self.obs.loc[row_pos]
        context_key = (
            str(row["dose_key"]),
            str(row["time_key"]),
            str(row["pubchem_cid"]),
        )
        if context_key not in self._baseline_peer_counts:
            self.get_baseline_vector(obs_id, "logFC", **lookup)
        return int(self._baseline_peer_counts.get(context_key, 0))

    def close(self) -> None:
        self.adata.file.close()


@dataclass(frozen=True)
class LineSourcePeerSelection:
    """One query's selected different-compound rows within a source stratum."""

    stratum: SignatureStratum
    row_indices: np.ndarray
    total_count: int

    @property
    def selected_count(self) -> int:
        return int(self.row_indices.size)


@dataclass
class ContextArrayCache:
    """Cache deterministic arrays only within one active scoring context.

    Standardizing a complete source/dose/time stratum once and selecting rows
    afterwards is exactly equivalent to standardizing those selected rows
    independently. Peer score vectors are likewise reusable for a fixed scale,
    query mask, and selected peer population. The context activation seam keeps
    memory bounded while allowing chunked checkpoint workers to share prepared
    arrays across adjacent chunks.
    """

    _active_context: Optional[Hashable] = field(default=None, init=False, repr=False)
    _standardized_strata: dict[Hashable, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _peer_scores: dict[Hashable, np.ndarray] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    stratum_hits: int = field(default=0, init=False)
    stratum_misses: int = field(default=0, init=False)
    score_hits: int = field(default=0, init=False)
    score_misses: int = field(default=0, init=False)

    def activate(self, context_key: Hashable) -> bool:
        """Activate ``context_key`` and clear prepared arrays on a change."""
        if context_key == self._active_context:
            return False
        self._active_context = context_key
        self._standardized_strata.clear()
        self._peer_scores.clear()
        return True

    def standardized_stratum(
        self,
        key: Hashable,
        build: Callable[[], np.ndarray],
    ) -> np.ndarray:
        if key in self._standardized_strata:
            self.stratum_hits += 1
            return self._standardized_strata[key]
        value = np.asarray(build())
        if value.ndim != 2:
            raise ValueError("A standardized z-score stratum must be a matrix")
        self._standardized_strata[key] = value
        self.stratum_misses += 1
        return value

    def peer_scores(
        self,
        key: Hashable,
        build: Callable[[], np.ndarray],
    ) -> np.ndarray:
        if key in self._peer_scores:
            self.score_hits += 1
            return self._peer_scores[key]
        value = np.asarray(build(), dtype=np.float64).reshape(-1)
        self._peer_scores[key] = value
        self.score_misses += 1
        return value

    def status(self) -> dict[str, int]:
        return {
            "stratum_hits": self.stratum_hits,
            "stratum_misses": self.stratum_misses,
            "score_hits": self.score_hits,
            "score_misses": self.score_misses,
        }


def select_line_source_peers(
    source: LineSource,
    obs_id: str,
    *,
    pubchem_cid: str,
    dose_key: str,
    time_key: str,
    max_peers: Optional[int],
    sampling_seed: int,
) -> LineSourcePeerSelection:
    """Resolve and deterministically sample the canonical centroid peer set."""
    row_pos = source.resolve_row_pos(
        obs_id,
        pubchem_cid=pubchem_cid,
        dose_key=dose_key,
        time_key=time_key,
    )
    row = source.obs.loc[row_pos]
    row_dose_key = str(row["dose_key"])
    row_time_key = str(row["time_key"])
    row_compound = str(row["pubchem_cid"])
    stratum = source.get_value_stratum(
        "logFC",
        row_dose_key,
        row_time_key,
    )
    selection = stratum.select_different_compound_peers(
        row_compound,
        max_peers=max_peers,
        seed_key="|".join(
            [
                source.dataset_name,
                source.cell_type,
                row_dose_key,
                row_time_key,
                row_compound,
            ]
        ),
        sampling_seed=int(sampling_seed),
    )
    return LineSourcePeerSelection(
        stratum=stratum,
        row_indices=selection.row_indices,
        total_count=selection.total_count,
    )


def line_source_stratum_arrays(
    source: LineSource,
    *,
    dose_key: str,
    time_key: str,
    layer_name: str = "logFC",
) -> tuple[np.ndarray, np.ndarray]:
    """Return compound labels and values for one cached source stratum."""
    stratum = source.get_value_stratum(
        str(layer_name),
        str(dose_key),
        str(time_key),
    )
    return stratum.compounds, stratum.values


def first_available_layer(
    source: LineSource,
    preferences: Sequence[str],
) -> str:
    """Return the first preferred AnnData layer exposed by a source."""
    available = set(source.adata.layers.keys())
    for candidate in preferences:
        if str(candidate) in available:
            return str(candidate)
    raise KeyError(
        f"None of {tuple(preferences)!r} are present in {source.path}; "
        f"available layers: {sorted(available)!r}"
    )


class LineSourceCatalog:
    """Own line sources and gene-position caches for one notebook kernel."""

    def __init__(self, source_dirs: Mapping[str, Path]) -> None:
        self.source_dirs = {
            str(dataset_name): Path(path)
            for dataset_name, path in source_dirs.items()
        }
        self.line_sources: dict[tuple[str, str], LineSource] = {}
        self.common_gene_cache: dict[
            tuple[str, str, str],
            tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self.line_global_shared_gene_keys: dict[str, np.ndarray] = {}
        self.global_gene_position_cache: dict[
            tuple[str, str],
            np.ndarray,
        ] = {}

    def resolve_line_path(self, dataset_name: str, cell_type: str) -> Path:
        dataset_dir = self.source_dirs[str(dataset_name)]
        candidates = [
            dataset_dir / f"{cell_type}_de.h5ad",
            dataset_dir / f"{cell_type}.h5ad",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Could not find a line file for dataset={dataset_name}, "
            f"cell_type={cell_type} in {dataset_dir}"
        )

    def get_line_source(self, dataset_name: str, cell_type: str) -> LineSource:
        cache_key = (str(dataset_name), str(cell_type))
        if cache_key not in self.line_sources:
            self.line_sources[cache_key] = LineSource(
                dataset_name=cache_key[0],
                cell_type=cache_key[1],
                path=self.resolve_line_path(*cache_key),
            )
        return self.line_sources[cache_key]

    def shared_gene_positions(
        self,
        left_source: LineSource,
        right_source: LineSource,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cache_key = (
            left_source.dataset_name,
            right_source.dataset_name,
            left_source.cell_type,
        )
        if cache_key not in self.common_gene_cache:
            shared_genes = [
                gene_key
                for gene_key in left_source.unique_gene_keys.tolist()
                if str(gene_key) in right_source.gene_to_pos
            ]
            left_positions = np.fromiter(
                (
                    left_source.gene_to_pos[str(gene_key)]
                    for gene_key in shared_genes
                ),
                dtype=np.int64,
                count=len(shared_genes),
            )
            right_positions = np.fromiter(
                (
                    right_source.gene_to_pos[str(gene_key)]
                    for gene_key in shared_genes
                ),
                dtype=np.int64,
                count=len(shared_genes),
            )
            self.common_gene_cache[cache_key] = (
                np.asarray(shared_genes, dtype=object),
                left_positions,
                right_positions,
            )
        return self.common_gene_cache[cache_key]

    def set_global_shared_gene_keys(
        self,
        retained_lines: Mapping[str, Sequence[str]],
        dataset_names: Sequence[str],
    ) -> dict[str, np.ndarray]:
        cell_types = sorted(
            {
                str(cell_type)
                for dataset_name in dataset_names
                for cell_type in retained_lines.get(str(dataset_name), ())
            }
        )
        line_gene_map: dict[str, np.ndarray] = {}
        for cell_type in cell_types:
            gene_sets: list[set[str]] = []
            for dataset_name in dataset_names:
                if cell_type not in retained_lines.get(str(dataset_name), ()):
                    continue
                source = self.get_line_source(str(dataset_name), cell_type)
                gene_sets.append(
                    {
                        str(gene_key)
                        for gene_key in source.unique_gene_keys.tolist()
                    }
                )
            line_gene_map[cell_type] = (
                np.asarray(sorted(set.intersection(*gene_sets)), dtype=object)
                if gene_sets
                else np.empty(0, dtype=object)
            )
        self.line_global_shared_gene_keys = line_gene_map
        self.global_gene_position_cache = {}
        return self.line_global_shared_gene_keys

    def global_gene_positions(
        self,
        source: LineSource,
    ) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (source.dataset_name, source.cell_type)
        line_gene_keys = self.line_global_shared_gene_keys.get(
            source.cell_type,
            np.empty(0, dtype=object),
        )
        if line_gene_keys.size == 0:
            return line_gene_keys, np.empty(0, dtype=np.int64)
        if cache_key not in self.global_gene_position_cache:
            missing = [
                str(gene_key)
                for gene_key in line_gene_keys.tolist()
                if str(gene_key) not in source.gene_to_pos
            ]
            if missing:
                raise KeyError(
                    f"Line-specific shared-gene set is inconsistent for "
                    f"{cache_key}; missing {len(missing)} genes."
                )
            self.global_gene_position_cache[cache_key] = np.fromiter(
                (
                    source.gene_to_pos[str(gene_key)]
                    for gene_key in line_gene_keys.tolist()
                ),
                dtype=np.int64,
                count=int(line_gene_keys.size),
            )
        return line_gene_keys, self.global_gene_position_cache[cache_key]

    def close(self) -> None:
        for source in self.line_sources.values():
            source.close()


def filter_finite_pair(
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    return left_values[mask], right_values[mask]


def signed_spearman(left_values: np.ndarray, right_values: np.ndarray) -> float:
    left_values, right_values = filter_finite_pair(
        np.asarray(left_values, dtype=np.float64),
        np.asarray(right_values, dtype=np.float64),
    )
    if left_values.size < 2:
        return float("nan")
    left_ranks = rankdata(left_values, method="average")
    right_ranks = rankdata(right_values, method="average")
    if np.allclose(left_ranks, left_ranks[0]) or np.allclose(
        right_ranks,
        right_ranks[0],
    ):
        return float("nan")
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def signed_overlap_at_k(
    left_values: np.ndarray,
    right_values: np.ndarray,
    *,
    k: int = TOP_K,
) -> float:
    left_values, right_values = filter_finite_pair(
        np.asarray(left_values, dtype=np.float64),
        np.asarray(right_values, dtype=np.float64),
    )
    k_eff = min(int(k), int(left_values.size) // 2)
    if k_eff < 1:
        return float("nan")
    top_left = np.argpartition(left_values, -k_eff)[-k_eff:]
    top_right = np.argpartition(right_values, -k_eff)[-k_eff:]
    bottom_left = np.argpartition(left_values, k_eff - 1)[:k_eff]
    bottom_right = np.argpartition(right_values, k_eff - 1)[:k_eff]
    top_overlap = (
        np.intersect1d(top_left, top_right, assume_unique=False).size / k_eff
    )
    bottom_overlap = (
        np.intersect1d(bottom_left, bottom_right, assume_unique=False).size
        / k_eff
    )
    return float(0.5 * (top_overlap + bottom_overlap))


def score_signature_pair(
    left_logfc: np.ndarray,
    right_logfc: np.ndarray,
    left_t: np.ndarray,
    right_t: np.ndarray,
) -> dict[str, float]:
    return {
        "spearman_logfc": signed_spearman(left_logfc, right_logfc),
        "spearman_t": signed_spearman(left_t, right_t),
        f"signed_overlap_t_top{TOP_K}": signed_overlap_at_k(
            left_t,
            right_t,
            k=TOP_K,
        ),
    }


def mean_available(values: Sequence[float]) -> float:
    finite_values = [value for value in values if pd.notna(value)]
    return (
        float(np.mean(finite_values))
        if finite_values
        else float("nan")
    )


def difference_if_both_defined(observed: float, baseline: float) -> float:
    """Subtract two scalar metrics while preserving an undefined result."""
    if pd.isna(observed) or pd.isna(baseline):
        return float("nan")
    return float(observed - baseline)


def build_symmetric_pair_metric_matrix(
    summary_frame: pd.DataFrame,
    value_col: str,
    dataset_order: Sequence[str],
) -> pd.DataFrame:
    """Expand one row per unordered dataset pair into a symmetric matrix."""
    matrix = pd.DataFrame(
        np.nan,
        index=list(dataset_order),
        columns=list(dataset_order),
        dtype=float,
    )
    if value_col not in summary_frame.columns:
        raise KeyError(f"summary_frame is missing value column {value_col!r}")
    for _, row in summary_frame.iterrows():
        dataset_a = str(row["dataset_a"])
        dataset_b = str(row["dataset_b"])
        if dataset_a not in matrix.index or dataset_b not in matrix.columns:
            continue
        value = pd.to_numeric(
            pd.Series([row[value_col]]),
            errors="coerce",
        ).iloc[0]
        matrix.loc[dataset_a, dataset_b] = value
        matrix.loc[dataset_b, dataset_a] = value
    return matrix


def format_ci_cell(row: pd.Series) -> str:
    """Format one mean and optional interval for reviewer-facing tables."""
    if not np.isfinite(row["mean"]):
        return ""
    if not (
        np.isfinite(row["ci_low"])
        and np.isfinite(row["ci_high"])
    ):
        return f"{row['mean']:.3f}"
    return (
        f"{row['mean']:.3f} "
        f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
    )


def empty_score_dict() -> dict[str, float]:
    return {
        "spearman_logfc": float("nan"),
        "spearman_t": float("nan"),
        f"signed_overlap_t_top{TOP_K}": float("nan"),
    }


__all__ = [
    "ANALYSIS_PROFILES",
    "DATASET_SPECS",
    "DEFAULT_MATCH_SETTINGS",
    "DISPLAY_LABELS",
    "CrossSourceScope",
    "ContextArrayCache",
    "DatasetSpec",
    "EMPTY_OVERLAP_FRAME",
    "INVALID_STRING_VALUES",
    "LineSource",
    "LineSourceCatalog",
    "LineSourcePeerSelection",
    "MATCH_PAIR_COLUMNS",
    "MATCH_PAIR_IDENTITY_COLUMNS",
    "MatchedPairShard",
    "MatchedPairShardPlan",
    "MatchSettings",
    "NUMERIC_SIG_FIGS",
    "TOP_K",
    "active_dataset_names",
    "analysis_output_dir",
    "build_dataset_index",
    "build_dataset_indices",
    "build_groups",
    "build_matched_pairs",
    "build_symmetric_pair_metric_matrix",
    "ci_fingerprint",
    "coerce_control_mask",
    "difference_if_both_defined",
    "empty_score_dict",
    "ensure_overlap_frame_schema",
    "file_inventory",
    "filter_finite_pair",
    "first_available_layer",
    "format_ci_cell",
    "format_numeric",
    "format_pubchem_cid",
    "global_gene_scope_lines",
    "line_source_inventory",
    "line_source_stratum_arrays",
    "load_overlap_obs",
    "matched_dataset_lines",
    "matched_dataset_names",
    "matched_pairs_fingerprint",
    "mean_available",
    "mutual_nearest_dose_pairs",
    "normalize_pubchem_cid_values",
    "pair_match_frame",
    "prepare_cross_source_scope",
    "pretty_label",
    "production_dataset_order",
    "raw_dose_fold_difference_matrix",
    "sanitize_string_values",
    "score_signature_pair",
    "select_line_source_peers",
    "selected_dataset_order",
    "signed_overlap_at_k",
    "signed_spearman",
    "source_dataset_dirs",
    "summarize_matched_pairs",
]
