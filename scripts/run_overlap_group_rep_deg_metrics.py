#!/usr/bin/env python3
"""Parallel, resumable DEG scoring with logFC, population z-scores, or moderated t.

Select components with --compute: raw, z-score-dataset,
z-score-dataset-cell-type, moderated-t. Moderated-t retains the raw logFC metrics,
uses logFC-derived DEG masks, and reuses the same selected peers.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cross_source_core
from scripts.cross_source_parallel import (
    TaskSpec,
    add_computation_arguments,
    add_common_arguments,
    diagnostic_frame,
    make_worker_catalog,
    make_worker_z_score_catalog,
    print_computations,
    read_task_input,
    resolve_computations,
    run_analysis,
    selected_z_score_scale_variants,
)
from scripts.cross_source_scoring import (
    ADJ_PVALUE_LAYER_PREFERENCES,
    DEG_DEFINITIONS,
    DE_OVERLAP_K_VALUES,
    PEER_BASELINE_VARIANTS,
    deg_mask,
    deg_restricted_spearman,
    direction_agreement_with_masks,
    finite_values_mask,
    observed_deg_metrics,
    sample_baseline_deg_metrics,
    score_deg_peer_distribution,
    select_source_peers,
    strict_symmetric_mean,
)
from scripts.peer_baselines import spearman_against_peers, summarize_peer_scores
from scripts.helper_group_rep_deg_signature_ci import summarize_run
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
    POPULATION_SCALE_VARIANTS,
)


ANALYSIS = "deg"
FINAL_METRICS_NAME = "deg_scored_metrics.tsv"
MODERATED_T_ANALYSIS = "deg_moderated_t"
MODERATED_T_METRICS_NAME = "deg_moderated_t_scored_metrics.tsv"
EFFECT_LAYER = "t"
PEER_PREFIX = "pbt"
EFFECT_METRICS = ("deg_t_spearman",)
CONTEXT_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "left_dose_key",
    "right_dose_key",
)
PEER_DEG_DEFINITION = "p05"
PEER_METRICS = ("deg_lfc_spearman", "direction_agreement")
COMPUTATIONS = {
    "raw": "Raw matched-pair DEG metrics and individual-peer baselines.",
    "z-score-dataset": "Dataset-wide per-gene population-z-score DEG metrics.",
    "z-score-dataset-cell-type": (
        "Dataset-by-cell-type per-gene population-z-score DEG metrics."
    ),
    "moderated-t": "Moderated-t DEG Spearman and peer baselines, retaining raw logFC metrics.",
}
COMPUTATION_TO_SCALE = {
    "z-score-dataset": PER_GENE_DATASET_VARIANT,
    "z-score-dataset-cell-type": PER_GENE_DATASET_CELL_TYPE_VARIANT,
}


def _lookup(row: Mapping[str, Any], side: str) -> dict[str, str]:
    return {
        "pubchem_cid": str(row["pubchem_cid"]),
        "dose_key": str(row[f"{side}_dose_key"]),
        "time_key": str(row["time_key"]),
    }


def _adj_layer(source: cross_source_core.LineSource) -> str:
    return cross_source_core.first_available_layer(
        source,
        ADJ_PVALUE_LAYER_PREFERENCES,
    )


def _centroid(
    source: cross_source_core.LineSource,
    obs_id: str,
    lookup: Mapping[str, str],
    positions: np.ndarray,
) -> Optional[np.ndarray]:
    value = source.get_baseline_vector(obs_id, "logFC", **lookup)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)[positions]


def _pair_baseline_fields(
    metrics: dict[str, Any],
    *,
    definition_key: str,
    suffix: str = "",
) -> None:
    observed_lookup = {
        "deg_lfc_spearman": (
            f"observed_deg_lfc_spearman_sym_{definition_key}{suffix}"
        ),
        "de_overlap_refn": (
            f"observed_de_overlap_refn_sym_{definition_key}{suffix}"
        ),
        "direction_agreement": (
            f"observed_direction_agreement_{definition_key}{suffix}"
        ),
        **{
            f"de_overlap_k{k}": (
                f"observed_de_overlap_k{k}_{definition_key}{suffix}"
            )
            for k in DE_OVERLAP_K_VALUES
        },
    }
    for metric_name, observed_key in observed_lookup.items():
        left_key = (
            f"left_baseline_{metric_name}_{definition_key}{suffix}"
        )
        right_key = (
            f"right_baseline_{metric_name}_{definition_key}{suffix}"
        )
        pair_key = (
            f"baseline_pair_{metric_name}_{definition_key}{suffix}"
        )
        delta_key = (
            f"delta_vs_baseline_pair_{metric_name}_{definition_key}{suffix}"
        )
        pair_value = cross_source_core.mean_available(
            [metrics[left_key], metrics[right_key]]
        )
        metrics[pair_key] = pair_value
        metrics[delta_key] = cross_source_core.difference_if_both_defined(
            metrics[observed_key],
            pair_value,
        )


def _l1000_restricted_positions(
    *,
    catalog: cross_source_core.LineSourceCatalog,
    left_source: cross_source_core.LineSource,
    right_source: cross_source_core.LineSource,
    retained_lines: Mapping[str, list[str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if {left_source.dataset_name, right_source.dataset_name} != {
        "sciplex",
        "tahoe",
    }:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=object), empty, empty
    cell_type = left_source.cell_type
    l1000_names = [
        dataset
        for dataset in ("l1000_phase1", "l1000_phase2")
        if cell_type in retained_lines.get(dataset, [])
    ]
    if not l1000_names:
        empty = np.empty(0, dtype=np.int64)
        return np.empty(0, dtype=object), empty, empty
    sources = [
        left_source,
        right_source,
        *[
            catalog.get_line_source(dataset, cell_type)
            for dataset in l1000_names
        ],
    ]
    shared = sorted(
        set.intersection(
            *[
                {str(gene) for gene in source.unique_gene_keys.tolist()}
                for source in sources
            ]
        )
    )
    genes = np.asarray(shared, dtype=object)
    left_positions = np.fromiter(
        (left_source.gene_to_pos[str(gene)] for gene in shared),
        dtype=np.int64,
        count=len(shared),
    )
    right_positions = np.fromiter(
        (right_source.gene_to_pos[str(gene)] for gene in shared),
        dtype=np.int64,
        count=len(shared),
    )
    return genes, left_positions, right_positions


def _metric_value(
    *,
    metric: str,
    sample: np.ndarray,
    comparison: Optional[np.ndarray],
    sample_adj_p: np.ndarray,
) -> float:
    if comparison is None:
        return float("nan")
    mask = deg_mask(sample, sample_adj_p, PEER_DEG_DEFINITION)
    if metric == "deg_lfc_spearman":
        return deg_restricted_spearman(sample, comparison, mask)
    if metric == "direction_agreement":
        return direction_agreement_with_masks(
            sample,
            comparison,
            mask,
            finite_values_mask(comparison),
        )
    raise ValueError(f"Unsupported DEG metric: {metric}")


def _score_deg_baselines(
    *,
    left_sample: np.ndarray,
    right_sample: np.ndarray,
    left_adj_p: np.ndarray,
    right_adj_p: np.ndarray,
    left_centroid: Optional[np.ndarray],
    right_centroid: Optional[np.ndarray],
    left_peers: np.ndarray,
    right_peers: np.ndarray,
    prefix: str,
    include_direction_agreement: bool = True,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    observed_by_metric = {
        "deg_lfc_spearman": (
            deg_restricted_spearman(
                left_sample,
                right_sample,
                deg_mask(
                    left_sample,
                    left_adj_p,
                    PEER_DEG_DEFINITION,
                ),
            ),
            deg_restricted_spearman(
                left_sample,
                right_sample,
                deg_mask(
                    right_sample,
                    right_adj_p,
                    PEER_DEG_DEFINITION,
                ),
            ),
        ),
        "direction_agreement": (
            direction_agreement_with_masks(
                left_sample,
                right_sample,
                deg_mask(
                    left_sample,
                    left_adj_p,
                    PEER_DEG_DEFINITION,
                ),
                deg_mask(
                    right_sample,
                    right_adj_p,
                    PEER_DEG_DEFINITION,
                ),
            ),
        )
        * 2,
    }
    sides = {
        "left": {
            "sample": left_sample,
            "adj_p": left_adj_p,
            "source_centroid": left_centroid,
            "target_centroid": right_centroid,
            "source_peer": left_peers,
            "target_peer": right_peers,
            "observed_index": 0,
        },
        "right": {
            "sample": right_sample,
            "adj_p": right_adj_p,
            "source_centroid": right_centroid,
            "target_centroid": left_centroid,
            "source_peer": right_peers,
            "target_peer": left_peers,
            "observed_index": 1,
        },
    }
    metrics = (
        PEER_METRICS
        if include_direction_agreement
        else ("deg_lfc_spearman",)
    )
    for metric in metrics:
        observed_left, observed_right = observed_by_metric[metric]
        observed_pair = (
            strict_symmetric_mean(observed_left, observed_right)
            if metric == "deg_lfc_spearman"
            else observed_left
        )
        if prefix == "w4" and metric == "deg_lfc_spearman":
            record[
                "w4_observed_deg_lfc_spearman_left_ref_p05"
            ] = observed_left
            record[
                "w4_observed_deg_lfc_spearman_right_ref_p05"
            ] = observed_right
            record["w4_observed_deg_lfc_spearman_sym_p05"] = observed_pair
        else:
            record[
                f"{prefix}_observed_{metric}_{PEER_DEG_DEFINITION}"
            ] = observed_pair
        for variant in PEER_BASELINE_VARIANTS:
            means: list[float] = []
            sds: list[float] = []
            fractions: list[float] = []
            percentiles: list[float] = []
            for side, spec in sides.items():
                observed_side = (observed_left, observed_right)[
                    spec["observed_index"]
                ]
                stem = (
                    f"{prefix}_{variant}_{metric}_{side}_"
                    f"{PEER_DEG_DEFINITION}"
                )
                if variant.endswith("_centroid"):
                    value = _metric_value(
                        metric=metric,
                        sample=spec["sample"],
                        comparison=spec[variant],
                        sample_adj_p=spec["adj_p"],
                    )
                    record[stem] = value
                    means.append(value)
                    continue
                summary = score_deg_peer_distribution(
                    observed=observed_side,
                    sample=spec["sample"],
                    comparison_matrix=spec[variant],
                    sample_adj_p=spec["adj_p"],
                    metric=metric,
                    stem=stem,
                )
                record.update(summary)
                means.append(float(summary[f"{stem}_mean_score"]))
                sds.append(float(summary[f"{stem}_sd_score"]))
                fractions.append(
                    float(summary[f"{stem}_fraction_below_observed"])
                )
                percentiles.append(
                    float(summary[f"{stem}_corrected_percentile"])
                )
            pair_stem = (
                f"{prefix}_{variant}_{metric}_pair_"
                f"{PEER_DEG_DEFINITION}"
            )
            pair_value = cross_source_core.mean_available(means)
            record[pair_stem] = pair_value
            record[
                f"{prefix}_delta_vs_{variant}_{metric}_"
                f"{PEER_DEG_DEFINITION}"
            ] = (
                cross_source_core.difference_if_both_defined(
                    observed_pair,
                    pair_value,
                )
            )
            if not variant.endswith("_centroid"):
                record[f"{pair_stem}_sd_score"] = (
                    cross_source_core.mean_available(sds)
                )
                record[f"{pair_stem}_fraction_below_observed"] = (
                    cross_source_core.mean_available(fractions)
                )
                record[f"{pair_stem}_corrected_percentile"] = (
                    cross_source_core.mean_available(percentiles)
                )
    return record


def _raw_deg_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    retained_lines: Mapping[str, list[str]],
    max_peers: int,
    sampling_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_a = str(row["dataset_a"])
    dataset_b = str(row["dataset_b"])
    cell_type = str(row["cell_type"])
    left_source = catalog.get_line_source(dataset_a, cell_type)
    right_source = catalog.get_line_source(dataset_b, cell_type)
    shared_genes, left_positions, right_positions = (
        catalog.shared_gene_positions(left_source, right_source)
    )
    if shared_genes.size < 2:
        raise ValueError(
            f"Fewer than two shared genes for {dataset_a} vs "
            f"{dataset_b} / {cell_type}"
        )
    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
    left_adj_layer = _adj_layer(left_source)
    right_adj_layer = _adj_layer(right_source)
    left_logfc_full = left_source.get_vector(
        left_obs_id,
        "logFC",
        **left_lookup,
    )
    right_logfc_full = right_source.get_vector(
        right_obs_id,
        "logFC",
        **right_lookup,
    )
    left_adj_full = left_source.get_vector(
        left_obs_id,
        left_adj_layer,
        **left_lookup,
    )
    right_adj_full = right_source.get_vector(
        right_obs_id,
        right_adj_layer,
        **right_lookup,
    )
    left_logfc = left_logfc_full[left_positions]
    right_logfc = right_logfc_full[right_positions]
    left_adj_p = left_adj_full[left_positions]
    right_adj_p = right_adj_full[right_positions]
    left_centroid = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        left_positions,
    )
    right_centroid = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        right_positions,
    )

    values: dict[str, Any] = {}
    for definition in DEG_DEFINITIONS:
        values.update(
            observed_deg_metrics(
                gene_keys=shared_genes,
                logfc_left=left_logfc,
                logfc_right=right_logfc,
                adj_p_left=left_adj_p,
                adj_p_right=right_adj_p,
                definition_key=definition,
            )
        )
        values.update(
            sample_baseline_deg_metrics(
                prefix="left",
                gene_keys=shared_genes,
                sample_logfc=left_logfc,
                baseline_logfc=left_centroid,
                sample_adj_p=left_adj_p,
                definition_key=definition,
            )
        )
        values.update(
            sample_baseline_deg_metrics(
                prefix="right",
                gene_keys=shared_genes,
                sample_logfc=right_logfc,
                baseline_logfc=right_centroid,
                sample_adj_p=right_adj_p,
                definition_key=definition,
            )
        )
        _pair_baseline_fields(values, definition_key=definition)

    restricted_genes, left_restricted, right_restricted = (
        _l1000_restricted_positions(
            catalog=catalog,
            left_source=left_source,
            right_source=right_source,
            retained_lines=retained_lines,
        )
    )
    suffix = "_l1000_restricted"
    if restricted_genes.size >= 2:
        left_restricted_centroid = (
            None
            if left_source.get_baseline_vector(
                left_obs_id,
                "logFC",
                **left_lookup,
            )
            is None
            else np.asarray(
                left_source.get_baseline_vector(
                    left_obs_id,
                    "logFC",
                    **left_lookup,
                )
            )[left_restricted]
        )
        right_restricted_centroid = (
            None
            if right_source.get_baseline_vector(
                right_obs_id,
                "logFC",
                **right_lookup,
            )
            is None
            else np.asarray(
                right_source.get_baseline_vector(
                    right_obs_id,
                    "logFC",
                    **right_lookup,
                )
            )[right_restricted]
        )
        for definition in DEG_DEFINITIONS:
            observed = observed_deg_metrics(
                gene_keys=restricted_genes,
                logfc_left=left_logfc_full[left_restricted],
                logfc_right=right_logfc_full[right_restricted],
                adj_p_left=left_adj_full[left_restricted],
                adj_p_right=right_adj_full[right_restricted],
                definition_key=definition,
            )
            left_baseline = sample_baseline_deg_metrics(
                prefix="left",
                gene_keys=restricted_genes,
                sample_logfc=left_logfc_full[left_restricted],
                baseline_logfc=left_restricted_centroid,
                sample_adj_p=left_adj_full[left_restricted],
                definition_key=definition,
            )
            right_baseline = sample_baseline_deg_metrics(
                prefix="right",
                gene_keys=restricted_genes,
                sample_logfc=right_logfc_full[right_restricted],
                baseline_logfc=right_restricted_centroid,
                sample_adj_p=right_adj_full[right_restricted],
                definition_key=definition,
            )
            for mapping in (observed, left_baseline, right_baseline):
                values.update(
                    {f"{key}{suffix}": value for key, value in mapping.items()}
                )
            _pair_baseline_fields(
                values,
                definition_key=definition,
                suffix=suffix,
            )

    left_selected = select_source_peers(
        left_source,
        left_obs_id,
        left_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    right_selected = select_source_peers(
        right_source,
        right_obs_id,
        right_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    if (
        left_selected.total_count
        != left_source.baseline_peer_count(left_obs_id, **left_lookup)
    ):
        raise AssertionError("Left peer membership disagrees with centroid")
    if (
        right_selected.total_count
        != right_source.baseline_peer_count(right_obs_id, **right_lookup)
    ):
        raise AssertionError("Right peer membership disagrees with centroid")
    left_peers = left_selected.stratum.values[
        left_selected.row_indices
    ][:, left_positions]
    right_peers = right_selected.stratum.values[
        right_selected.row_indices
    ][:, right_positions]
    record = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "n_l1000_restricted_common_genes": int(restricted_genes.size),
        "left_adj_pvalue_layer": left_adj_layer,
        "right_adj_pvalue_layer": right_adj_layer,
        "left_baseline_peer_count": left_selected.total_count,
        "right_baseline_peer_count": right_selected.total_count,
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "peer_sampling_seed": sampling_seed,
        "max_baseline_peers": max_peers,
        **values,
    }
    record.update(
        _score_deg_baselines(
            left_sample=left_logfc,
            right_sample=right_logfc,
            left_adj_p=left_adj_p,
            right_adj_p=right_adj_p,
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peers=left_peers,
            right_peers=right_peers,
            prefix="pb",
        )
    )
    state = {
        "pubchem_cid": str(row["pubchem_cid"]),
        "left_source": left_source,
        "right_source": right_source,
        "left_positions": left_positions,
        "right_positions": right_positions,
        "left_logfc_full": left_logfc_full,
        "right_logfc_full": right_logfc_full,
        "left_adj_p": left_adj_p,
        "right_adj_p": right_adj_p,
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _deg_z_score_base_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    max_peers: int,
    sampling_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare only the shared inputs required by selected z-score kernels."""
    dataset_a = str(row["dataset_a"])
    dataset_b = str(row["dataset_b"])
    cell_type = str(row["cell_type"])
    left_source = catalog.get_line_source(dataset_a, cell_type)
    right_source = catalog.get_line_source(dataset_b, cell_type)
    shared_genes, left_positions, right_positions = (
        catalog.shared_gene_positions(left_source, right_source)
    )
    if shared_genes.size < 2:
        raise ValueError(
            f"Fewer than two shared genes for {dataset_a} vs "
            f"{dataset_b} / {cell_type}"
        )
    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
    left_adj_layer = _adj_layer(left_source)
    right_adj_layer = _adj_layer(right_source)
    left_logfc_full = left_source.get_vector(
        left_obs_id,
        "logFC",
        **left_lookup,
    )
    right_logfc_full = right_source.get_vector(
        right_obs_id,
        "logFC",
        **right_lookup,
    )
    left_adj_p = left_source.get_vector(
        left_obs_id,
        left_adj_layer,
        **left_lookup,
    )[left_positions]
    right_adj_p = right_source.get_vector(
        right_obs_id,
        right_adj_layer,
        **right_lookup,
    )[right_positions]
    left_selected = select_source_peers(
        left_source,
        left_obs_id,
        left_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    right_selected = select_source_peers(
        right_source,
        right_obs_id,
        right_lookup,
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    if (
        left_selected.total_count
        != left_source.baseline_peer_count(left_obs_id, **left_lookup)
    ):
        raise AssertionError("Left peer membership disagrees with centroid")
    if (
        right_selected.total_count
        != right_source.baseline_peer_count(right_obs_id, **right_lookup)
    ):
        raise AssertionError("Right peer membership disagrees with centroid")
    record = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "left_adj_pvalue_layer": left_adj_layer,
        "right_adj_pvalue_layer": right_adj_layer,
        "left_baseline_peer_count": left_selected.total_count,
        "right_baseline_peer_count": right_selected.total_count,
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "peer_sampling_seed": sampling_seed,
        "max_baseline_peers": max_peers,
    }
    state = {
        "pubchem_cid": str(row["pubchem_cid"]),
        "left_source": left_source,
        "right_source": right_source,
        "left_positions": left_positions,
        "right_positions": right_positions,
        "left_logfc_full": left_logfc_full,
        "right_logfc_full": right_logfc_full,
        "left_adj_p": left_adj_p,
        "right_adj_p": right_adj_p,
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _z_score_deg_columns(
    *,
    state: Mapping[str, Any],
    z_score_catalog,
    scale_variant: str,
) -> dict[str, Any]:
    left_source = state["left_source"]
    right_source = state["right_source"]
    left_positions = state["left_positions"]
    right_positions = state["right_positions"]
    left_selected = state["left_selected"]
    right_selected = state["right_selected"]
    pubchem_cid = str(state["pubchem_cid"])
    left_sample = z_score_catalog.standardize_vector(
        left_source,
        state["left_logfc_full"],
        scale_variant=scale_variant,
    )[left_positions]
    right_sample = z_score_catalog.standardize_vector(
        right_source,
        state["right_logfc_full"],
        scale_variant=scale_variant,
    )[right_positions]
    left_matrix = z_score_catalog.standardize_matrix(
        left_source,
        left_selected.stratum.values,
        scale_variant=scale_variant,
    )[:, left_positions]
    right_matrix = z_score_catalog.standardize_matrix(
        right_source,
        right_selected.stratum.values,
        scale_variant=scale_variant,
    )[:, right_positions]
    left_all_rows = left_selected.stratum.different_compound_peer_indices(
        pubchem_cid
    )
    right_all_rows = right_selected.stratum.different_compound_peer_indices(
        pubchem_cid
    )
    left_centroid = (
        None
        if not len(left_all_rows)
        else left_matrix[left_all_rows].mean(axis=0, dtype=np.float64)
    )
    right_centroid = (
        None
        if not len(right_all_rows)
        else right_matrix[right_all_rows].mean(axis=0, dtype=np.float64)
    )
    values: dict[str, Any] = {
        "n_common_genes": int(len(left_positions)),
        "n_w4_common_genes": int(
            np.sum(np.isfinite(left_sample) & np.isfinite(right_sample))
        ),
        "left_population_row_count": (
            z_score_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        "right_population_row_count": (
            z_score_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        "left_population_valid_genes": (
            z_score_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        "right_population_valid_genes": (
            z_score_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "n_deg_left_p05": int(
            deg_mask(
                state["left_logfc_full"][left_positions],
                state["left_adj_p"],
                "p05",
            ).sum()
        ),
        "n_deg_right_p05": int(
            deg_mask(
                state["right_logfc_full"][right_positions],
                state["right_adj_p"],
                "p05",
            ).sum()
        ),
    }
    values.update(
        _score_deg_baselines(
            left_sample=left_sample,
            right_sample=right_sample,
            left_adj_p=state["left_adj_p"],
            right_adj_p=state["right_adj_p"],
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peers=left_matrix[left_selected.row_indices],
            right_peers=right_matrix[right_selected.row_indices],
            prefix="w4",
            include_direction_agreement=False,
        )
    )
    return {
        f"{scale_variant}__{key}": value
        for key, value in values.items()
    }


def assert_aligned(logfc_stratum, effect_stratum, source, dose_key: str, time_key: str) -> None:
    """Pin the assumption that lets us reuse logFC peer indices on the t stratum.

    Both strata are selected by (dose_key, time_key) and cached per layer, so their row
    order matches today. If that stopped holding, reusing row_indices would pair each
    peer's identity with another peer's moderated t: every column would stay finite and
    plausible, so the error would not surface downstream.
    """
    if effect_stratum.values.shape[0] != logfc_stratum.values.shape[0]:
        raise AssertionError(
            f"{source.dataset_name}/{source.cell_type} {dose_key}/{time_key}: "
            f"{EFFECT_LAYER} stratum has {effect_stratum.values.shape[0]} rows, "
            f"logFC has {logfc_stratum.values.shape[0]}"
        )
    if not np.array_equal(
        np.asarray(effect_stratum.compounds), np.asarray(logfc_stratum.compounds)
    ):
        raise AssertionError(
            f"{source.dataset_name}/{source.cell_type} {dose_key}/{time_key}: "
            f"{EFFECT_LAYER} and logFC strata disagree on the compound axis"
        )


def effect_peer_matrix(source, selected, lookup: Mapping[str, str], positions) -> np.ndarray:
    """The t values of exactly the peers the logFC family scored."""
    dose_key = str(lookup["dose_key"])
    time_key = str(lookup["time_key"])
    effect_stratum = source.get_value_stratum(EFFECT_LAYER, dose_key, time_key)
    assert_aligned(selected.stratum, effect_stratum, source, dose_key, time_key)
    return effect_stratum.values[selected.row_indices][:, positions]


def effect_centroid(source, obs_id: str, lookup: Mapping[str, str], positions) -> Optional[np.ndarray]:
    """Different-compound centroid on the t layer.

    Use the same centroid convention as logFC, evaluated on the t layer.
    """
    value = source.get_baseline_vector(obs_id, EFFECT_LAYER, **lookup)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)[positions]


def _effect_value(
    *,
    metric: str,
    effect: np.ndarray,
    comparison: Optional[np.ndarray],
    mask: np.ndarray,
) -> float:
    """One scalar comparison of the query's t profile against another profile.

    mask is supplied by the caller, built from logFC and adjusted p, which is the whole
    point of not reusing the driver's _metric_value: that helper takes the DEG set from the
    same array it correlates, and here the set has to come from logFC while the values come
    from t.
    """
    if comparison is None:
        return float("nan")
    if metric == "deg_t_spearman":
        return deg_restricted_spearman(effect, comparison, mask)
    raise ValueError(f"Unsupported moderated-t DEG metric: {metric}")


def _effect_peer_scores(
    *,
    metric: str,
    effect: np.ndarray,
    comparison_matrix: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Per-peer scores, evaluated on the query's logFC-derived DEG mask."""
    matrix = np.atleast_2d(np.asarray(comparison_matrix, dtype=np.float64))
    if metric == "deg_t_spearman":
        return spearman_against_peers(effect, matrix, mask)
    raise ValueError(f"Unsupported moderated-t DEG metric: {metric}")


def score_moderated_t_baselines(
    *,
    left_effect: np.ndarray,
    right_effect: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
    left_centroid: Optional[np.ndarray],
    right_centroid: Optional[np.ndarray],
    left_peers: np.ndarray,
    right_peers: np.ndarray,
) -> dict[str, Any]:
    """The moderated-t analogue of _score_deg_baselines, with the mask decoupled.

    Structure follows the driver's function deliberately, including which side's DEG mask
    references which observed value, so the two families line up column for column.
    """
    record: dict[str, Any] = {}
    sides = {
        "left": {
            "effect": left_effect,
            "mask": left_mask,
            "source_centroid": left_centroid,
            "target_centroid": right_centroid,
            "source_peer": left_peers,
            "target_peer": right_peers,
        },
        "right": {
            "effect": right_effect,
            "mask": right_mask,
            "source_centroid": right_centroid,
            "target_centroid": left_centroid,
            "source_peer": right_peers,
            "target_peer": left_peers,
        },
    }
    for metric in EFFECT_METRICS:
        # Left-referenced and right-referenced observed values: same two profiles, scored
        # on whichever side's DEG set defines the panel. sym is the strict mean of the two.
        observed_left = _effect_value(
            metric=metric,
            effect=left_effect,
            comparison=right_effect,
            mask=left_mask,
        )
        observed_right = _effect_value(
            metric=metric,
            effect=right_effect,
            comparison=left_effect,
            mask=right_mask,
        )
        observed_pair = strict_symmetric_mean(observed_left, observed_right)
        record[f"{PEER_PREFIX}_observed_{metric}_left_ref_{PEER_DEG_DEFINITION}"] = observed_left
        record[f"{PEER_PREFIX}_observed_{metric}_right_ref_{PEER_DEG_DEFINITION}"] = observed_right
        record[f"{PEER_PREFIX}_observed_{metric}_sym_{PEER_DEG_DEFINITION}"] = observed_pair

        for variant in PEER_BASELINE_VARIANTS:
            # The pair-level aggregation below mirrors the driver exactly, including three
            # choices that are easy to get wrong and impossible to spot afterwards:
            #
            #   * pair_value uses mean_available, not strict_symmetric_mean. The observed
            #     value is strict (NaN if either side is NaN) but the baseline is not: if
            #     one side is missing the driver keeps the other rather than dropping the
            #     row. Using the strict form here would silently shrink the panel.
            #   * the delta goes through difference_if_both_defined rather than bare
            #     subtraction, so the NaN policy is the driver's and not float semantics'.
            #   * the pair-level sd / fraction / corrected percentile exist only for peer
            #     variants, not centroids, and are means of the per-side values rather
            #     than recomputed from a pooled distribution. The published Table 4
            #     "Corrected percentile" column is this pair-level value.
            means: list[float] = []
            sds: list[float] = []
            fractions: list[float] = []
            percentiles: list[float] = []
            for side, spec in sides.items():
                observed_side = observed_left if side == "left" else observed_right
                stem = f"{PEER_PREFIX}_{variant}_{metric}_{side}_{PEER_DEG_DEFINITION}"
                if variant.endswith("_centroid"):
                    value = _effect_value(
                        metric=metric,
                        effect=spec["effect"],
                        comparison=spec[variant],
                        mask=spec["mask"],
                    )
                    record[stem] = value
                    means.append(value)
                    continue
                scores = _effect_peer_scores(
                    metric=metric,
                    effect=spec["effect"],
                    comparison_matrix=spec[variant],
                    mask=spec["mask"],
                )
                summary = summarize_peer_scores(observed_side, scores, stem)
                record.update(summary)
                means.append(float(summary[f"{stem}_mean_score"]))
                sds.append(float(summary[f"{stem}_sd_score"]))
                fractions.append(float(summary[f"{stem}_fraction_below_observed"]))
                percentiles.append(float(summary[f"{stem}_corrected_percentile"]))
            pair_stem = f"{PEER_PREFIX}_{variant}_{metric}_pair_{PEER_DEG_DEFINITION}"
            pair_value = cross_source_core.mean_available(means)
            record[pair_stem] = pair_value
            record[
                f"{PEER_PREFIX}_delta_vs_{variant}_{metric}_{PEER_DEG_DEFINITION}"
            ] = cross_source_core.difference_if_both_defined(observed_pair, pair_value)
            if not variant.endswith("_centroid"):
                record[f"{pair_stem}_sd_score"] = cross_source_core.mean_available(sds)
                record[f"{pair_stem}_fraction_below_observed"] = (
                    cross_source_core.mean_available(fractions)
                )
                record[f"{pair_stem}_corrected_percentile"] = (
                    cross_source_core.mean_available(percentiles)
                )
    return record


def _moderated_t_deg_columns(
    row: pd.Series,
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Score t on the raw logFC DEG masks and the already-selected peer rows."""
    left_source = state["left_source"]
    right_source = state["right_source"]
    left_positions = state["left_positions"]
    right_positions = state["right_positions"]
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])

    left_logfc = np.asarray(state["left_logfc_full"], dtype=np.float64)[left_positions]
    right_logfc = np.asarray(state["right_logfc_full"], dtype=np.float64)[right_positions]
    left_adj_p = state["left_adj_p"]
    right_adj_p = state["right_adj_p"]
    # The published DEG panel: selection from logFC and adjusted p, never from t.
    left_mask = deg_mask(left_logfc, left_adj_p, PEER_DEG_DEFINITION)
    right_mask = deg_mask(right_logfc, right_adj_p, PEER_DEG_DEFINITION)

    left_effect = left_source.get_vector(left_obs_id, EFFECT_LAYER, **left_lookup)[
        left_positions
    ]
    right_effect = right_source.get_vector(right_obs_id, EFFECT_LAYER, **right_lookup)[
        right_positions
    ]
    return score_moderated_t_baselines(
        left_effect=left_effect,
        right_effect=right_effect,
        left_mask=left_mask,
        right_mask=right_mask,
        left_centroid=effect_centroid(
            left_source, left_obs_id, left_lookup, left_positions
        ),
        right_centroid=effect_centroid(
            right_source, right_obs_id, right_lookup, right_positions
        ),
        left_peers=effect_peer_matrix(
            left_source, state["left_selected"], left_lookup, left_positions
        ),
        right_peers=effect_peer_matrix(
            right_source, state["right_selected"], right_lookup, right_positions
        ),
    )


def score_deg_row(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    z_score_catalog,
    retained_lines: Mapping[str, list[str]],
    max_peers: int,
    sampling_seed: int,
    include_raw_metrics: bool = True,
    z_score_scale_variants: tuple[str, ...] = POPULATION_SCALE_VARIANTS,
    include_moderated_t: bool = False,
) -> dict[str, Any]:
    if include_raw_metrics or include_moderated_t:
        record, state = _raw_deg_record(
            row,
            catalog=catalog,
            retained_lines=retained_lines,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    else:
        record, state = _deg_z_score_base_record(
            row,
            catalog=catalog,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    if include_moderated_t:
        record.update(_moderated_t_deg_columns(row, state=state))
    for scale_variant in z_score_scale_variants:
        record.update(
            _z_score_deg_columns(
                state=state,
                z_score_catalog=z_score_catalog,
                scale_variant=scale_variant,
            )
        )
    return record


def score_task(
    task: TaskSpec,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = read_task_input(config, task)
    settings = config["settings"]
    include_moderated_t = bool(settings.get("include_moderated_t", False))
    diagnostic_analysis = MODERATED_T_ANALYSIS if include_moderated_t else ANALYSIS
    catalog = make_worker_catalog(config)
    diagnostics: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    try:
        z_score_catalog = (
            make_worker_z_score_catalog(config, catalog)
            if settings["z_score_scale_variants"]
            else None
        )
        for _, row in frame.iterrows():
            try:
                records.append(
                    score_deg_row(
                        row,
                        catalog=catalog,
                        z_score_catalog=z_score_catalog,
                        retained_lines=config["retained_lines"],
                        max_peers=int(settings["max_baseline_peers"]),
                        sampling_seed=int(settings["peer_sampling_seed"]),
                        include_raw_metrics=bool(
                            settings["include_raw_metrics"]
                        ),
                        z_score_scale_variants=tuple(
                            settings["z_score_scale_variants"]
                        ),
                        include_moderated_t=include_moderated_t,
                    )
                )
            except KeyError as exc:
                if "Could not resolve obs_id=" not in str(exc):
                    raise
                diagnostics.append(
                    {
                        "task_id": task.task_id,
                        "analysis": diagnostic_analysis,
                        "dataset_a": row.get("dataset_a", ""),
                        "dataset_b": row.get("dataset_b", ""),
                        "cell_type": row.get("cell_type", ""),
                        "time_key": row.get("time_key", ""),
                        "left_dose_key": row.get("left_dose_key", ""),
                        "right_dose_key": row.get("right_dose_key", ""),
                        "left_obs_id": row.get("left_obs_id", ""),
                        "right_obs_id": row.get("right_obs_id", ""),
                        "stage": f"{diagnostic_analysis}_scoring",
                        "reason": "unresolved_source_row",
                        "error": str(exc),
                    }
                )
    finally:
        catalog.close()
    return pd.DataFrame(records), diagnostic_frame(diagnostics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Parallel, resumable matched-pair DEG metric scoring.",
        epilog=(
            "Without --compute, run raw plus the z-score scopes selected by --z-score-scales. "
            "Moderated-t is opt-in and retains raw logFC metrics."
        ),
    )
    add_common_arguments(parser, analysis=ANALYSIS)
    add_computation_arguments(parser, computations=COMPUTATIONS)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_computations:
        print_computations(COMPUTATIONS)
        return 0
    if args.compute:
        selected_computations = resolve_computations(
            args.compute,
            computations=COMPUTATIONS,
        )
        args.required_layers_override = (
            ("logFC", EFFECT_LAYER)
            if "moderated-t" in selected_computations
            else ("logFC",)
        )
        args.require_adj_p_override = True
        z_score_scale_variants = tuple(
            COMPUTATION_TO_SCALE[name]
            for name in selected_computations
            if name in COMPUTATION_TO_SCALE
        )
    else:
        z_score_scale_variants = selected_z_score_scale_variants(args.z_score_scales)
        selected_computations = (
            "raw",
            *(
                name
                for name in COMPUTATIONS
                if COMPUTATION_TO_SCALE.get(name) in z_score_scale_variants
            ),
        )
    include_moderated_t = "moderated-t" in selected_computations
    settings = {
        "deg_p_threshold": 0.05,
        "deg_abs_logfc_threshold": 0.2,
        "deg_definitions": list(DEG_DEFINITIONS),
        "de_overlap_k": list(DE_OVERLAP_K_VALUES),
        "max_baseline_peers": args.max_baseline_peers,
        "peer_sampling_seed": args.peer_sampling_seed,
        "computations": list(selected_computations),
        "include_raw_metrics": "raw" in selected_computations or include_moderated_t,
        "include_moderated_t": include_moderated_t,
        "z_score_scale_variants": list(z_score_scale_variants),
        "scorer_version": "parallel-deg-v4",
    }
    metrics_path, _ = run_analysis(
        analysis=ANALYSIS,
        scorer_module="scripts.run_overlap_group_rep_deg_metrics",
        args=args,
        context_columns=CONTEXT_COLUMNS,
        rows_per_shard=args.rows_per_shard,
        final_metrics_name=(
            MODERATED_T_METRICS_NAME if include_moderated_t else FINAL_METRICS_NAME
        ),
        settings=settings,
        code_paths=[
            Path(__file__),
            REPO_ROOT / "scripts" / "cross_source_parallel.py",
            REPO_ROOT / "scripts" / "cross_source_scoring.py",
            REPO_ROOT / "scripts" / "cross_source_core.py",
            REPO_ROOT / "scripts" / "peer_baselines.py",
            REPO_ROOT / "scripts" / "population_zscore.py",
        ],
    )
    summarize_run(metrics_path, analysis=ANALYSIS, computations=selected_computations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
