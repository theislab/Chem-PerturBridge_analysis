#!/usr/bin/env python3
"""Parallel Tables 12/13 retrieval for raw, direction, moderated-t, and two z-score scopes.

The z-score peer path preserves elmella/cigs-production-hardening (46a72c5).
Strict whole-stratum centroids retain the historical retrieval notebook convention. Historical
metric/scale names are data and bootstrap identifiers, independent of CLI names.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

# The paper's retrieval notebooks scored in one process with eight BLAS threads; exact
# sign(logFC) cosine ties depend on that summation order. Threads must be set before NumPy loads.
# Spawned --workers processes re-run this module as __mp_main__ and keep one thread each.
PAPER_RUNTIME = {"workers": 1, "blas_threads": 8}
_blas_threads = PAPER_RUNTIME["blas_threads"] if __name__ == "__main__" else 1
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = str(_blas_threads)

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import cross_source_core
from scripts.cross_source_parallel import (
    TaskSpec, add_common_arguments, add_computation_arguments, resolve_computations,
    print_computations, make_worker_catalog, make_worker_z_score_catalog, read_task_input,
    read_overlap_metadata, report_task_progress, diagnostic_frame, run_analysis,
)
from scripts.cross_source_scoring import (
    ADJ_PVALUE_LAYER_PREFERENCES, RETRIEVAL_SIMILARITIES, build_logfc_pool_matrices,
    build_pool_matrices, pool_frame_for_context, strict_positive_maps, strict_positive_mask,
    similarity_matrix, score_vector_against_matrix, score_vector_pair,
    peer_vector_scores, peer_vector_pair, summarize_retrieval_scores, select_peer_indices, retrieval_peer_summary,
    expected_single_signature_metrics, raw_dose_fold_differences, exact_cross_assay_null,
    exact_null_rank_calibration, exact_null_mid_p,
)
from scripts.population_zscore import (
    PER_GENE_DATASET_VARIANT, PER_GENE_DATASET_CELL_TYPE_VARIANT,
)
from scripts.helper_group_rep_retrieval_ci import summarize_run

ANALYSIS = "retrieval"
FINAL_METRICS_NAME = "retrieval_scored_metrics.tsv"
CONTEXT_COLUMNS = ("dataset_a", "dataset_b", "cell_type", "time_key")
MIN_TARGET_CANDIDATES = 5
MIN_UNIQUE_COMPOUNDS_PER_SIDE = 2
RAW_SCALE_VARIANT = "raw"
COMPUTATIONS = {
    "raw": "Raw logFC strict retrieval and peers.",
    "direction": "sign(logFC) strict retrieval and peers.",
    "moderated-t": "Moderated-t strict retrieval and peers.",
    "z-score-dataset": "Dataset population z-scores with original reviewer peer CIs.",
    "z-score-dataset-cell-type": "Dataset/cell-type population z-scores with original reviewer peer CIs.",
}
REPRESENTATIONS = {"raw": "logFC", "direction": "logFC_direction", "moderated-t": "moderated_t"}
Z_SCORE_SCALES = {"z-score-dataset": PER_GENE_DATASET_VARIANT,
                  "z-score-dataset-cell-type": PER_GENE_DATASET_CELL_TYPE_VARIANT}
IDENTITY_COLUMNS = [
    "dataset_a", "dataset_b", "direction", "query_dataset", "target_dataset",
    "cell_type", "time_key", "query_obs_id", "query_pubchem_cid", "query_dose_key",
    "query_dose_uM", "representation", "retrieval_variant", "similarity_metric", "scale_variant",
]


COUNT_COLUMNS = [
    "target_pool_size",
    "target_pool_size_before_similarity_filter",
    "n_query_side_conditions",
    "n_target_side_conditions",
    "n_query_side_unique_compounds",
    "n_target_side_unique_compounds",
    "n_shared_genes",
    "n_raw_shared_genes",
    "n_positive_candidates",
    "n_negative_candidates",
    "n_zero_norm_target_candidates",
    "query_baseline_peer_count",
    "centroid_baseline_peer_count",
    "single_signature_baseline_peer_count",
    "n_excluded_single_signature_peers",
    "source_centroid_peer_count",
    "target_centroid_peer_count",
    "source_individual_total_count",
    "source_individual_scored_count",
    "target_individual_total_count",
    "target_individual_scored_count",
    "left_population_row_count",
    "right_population_row_count",
]
METRIC_COLUMNS = [
    "observed_best_positive_rank",
    "observed_normalized_best_positive_rank",
    "observed_recall_at_1",
    "observed_auroc",
    "best_positive_rank",
    "normalized_best_positive_rank",
    "recall_at_1",
    "auroc",
    "observed_best_positive_similarity",
    "observed_mean_positive_similarity",
    "observed_sd_positive_similarity",
    "best_target_obs_id",
    "best_target_dose_key",
    "best_target_dose_uM",
    "best_target_dose_fold_difference",
    "source_centroid_similarity",
    "baseline_candidate_similarity",
    "baseline_available",
    "baseline_best_positive_rank",
    "baseline_normalized_best_positive_rank",
    "baseline_recall_at_1",
    "baseline_auroc",
    "centroid_baseline_candidate_similarity",
    "centroid_baseline_best_positive_rank",
    "centroid_baseline_normalized_best_positive_rank",
    "centroid_baseline_recall_at_1",
    "centroid_baseline_auroc",
    "single_signature_baseline_best_positive_rank",
    "single_signature_baseline_normalized_best_positive_rank",
    "single_signature_baseline_recall_at_1",
    "single_signature_baseline_auroc",
    "delta_vs_baseline_normalized_best_positive_rank",
    "delta_vs_baseline_recall_at_1",
    "delta_vs_baseline_auroc",
    "random_normalized_best_positive_rank",
    "random_recall_at_1",
    "random_auroc",
    "target_centroid_similarity",
    "target_decoy_null_normalized_best_positive_rank",
    "target_decoy_null_best_positive_rank",
    "target_decoy_null_recall_at_1",
    "target_decoy_null_auroc",
    "exact_null_mid_p",
    "null_pit",
    "null_p_value",
    "min_achievable_null_p_value",
    "null_pit_below_half",
    "observed_auroc_minus_half",
    "within_source_best_positive_rank",
    "within_source_normalized_best_positive_rank",
    "within_source_recall_at_1",
    "within_source_auroc",
    "source_centroid_best_rank",
    "source_centroid_normalized_rank",
    "source_centroid_recall_at_1",
    "source_centroid_auroc",
    "target_centroid_best_rank",
    "target_centroid_normalized_rank",
    "target_centroid_recall_at_1",
    "target_centroid_auroc",
    "source_peer_best_rank",
    "source_peer_normalized_rank",
    "source_peer_recall_at_1",
    "source_peer_auroc",
    "target_peer_best_rank",
    "target_peer_normalized_rank",
    "target_peer_recall_at_1",
    "target_peer_auroc",
    "within_dataset_positive_control_best_positive_rank",
    "within_dataset_positive_control_normalized_best_positive_rank",
    "within_dataset_positive_control_recall_at_1",
    "within_dataset_positive_control_auroc",
    "delta_observed_vs_source_centroid",
    "delta_observed_vs_target_centroid",
    "delta_observed_vs_target_decoy_null_rank",
    "delta_observed_vs_target_decoy_null_recall_at_1",
    "delta_observed_vs_target_decoy_null_auroc",
    "delta_vs_target_decoy_null_normalized_best_positive_rank",
    "delta_vs_target_decoy_null_recall_at_1",
    "delta_vs_target_decoy_null_auroc",
    "delta_vs_random_normalized_best_positive_rank",
    "delta_vs_centroid_baseline_normalized_best_positive_rank",
    "delta_vs_centroid_baseline_recall_at_1",
    "delta_vs_centroid_baseline_auroc",
    "delta_vs_single_signature_baseline_normalized_best_positive_rank",
    "delta_vs_single_signature_baseline_recall_at_1",
    "delta_vs_single_signature_baseline_auroc",
    "delta_vs_within_dataset_positive_control_normalized_best_positive_rank",
    *[
        f"delta_vs_{variant}_{field}"
        for variant in (
            "source_centroid",
            "target_centroid",
            "source_peer",
            "target_peer",
            "within_dataset_positive_control",
        )
        for field in ("normalized_rank", "recall_at_1", "auroc")
    ],
]
PEER_COLUMNS = [
    f"{prefix}_{field}"
    for prefix in ("source_individual", "target_individual")
    for field in (
        "n_peers",
        "mean_similarity",
        "sd_similarity",
        "n_below_observed",
        "fraction_below_observed",
        "corrected_percentile",
        "n_at_least_observed",
        "empirical_p_upper",
    )
]
DELTA_PEER_COLUMNS = [
    "delta_observed_vs_source_individual_mean",
    "delta_observed_vs_target_individual_mean",
]
RETRIEVAL_OUTPUT_COLUMNS = [
    *IDENTITY_COLUMNS,
    *COUNT_COLUMNS,
    *METRIC_COLUMNS,
    *PEER_COLUMNS,
    *DELTA_PEER_COLUMNS,
]

def _record_key(record: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        str(record[column])
        for column in (
            "dataset_a",
            "dataset_b",
            "direction",
            "cell_type",
            "time_key",
            "query_obs_id",
            "representation",
            "retrieval_variant",
            "similarity_metric",
            "scale_variant",
        )
    )


def _base_record(
    *,
    dataset_a: str,
    dataset_b: str,
    direction: str,
    query_dataset: str,
    target_dataset: str,
    cell_type: str,
    time_key: str,
    query_pool: pd.DataFrame,
    query_idx: int,
    target_pool: pd.DataFrame,
    representation: str,
    retrieval_variant: str,
    similarity_metric: str,
    scale_variant: str,
    n_shared_genes: int,
) -> dict[str, Any]:
    return {
        "dataset_a": dataset_a,
        "dataset_b": dataset_b,
        "direction": direction,
        "query_dataset": query_dataset,
        "target_dataset": target_dataset,
        "cell_type": cell_type,
        "time_key": time_key,
        "query_obs_id": str(query_pool.loc[query_idx, "obs_id"]),
        "query_pubchem_cid": str(query_pool.loc[query_idx, "pubchem_cid"]),
        "query_dose_key": str(query_pool.loc[query_idx, "dose_key"]),
        "query_dose_uM": float(query_pool.loc[query_idx, "pert_dose_uM"]),
        "representation": representation,
        "retrieval_variant": retrieval_variant,
        "similarity_metric": similarity_metric,
        "scale_variant": scale_variant,
        "target_pool_size": int(len(target_pool)),
        "n_query_side_conditions": int(len(query_pool)),
        "n_target_side_conditions": int(len(target_pool)),
        "n_query_side_unique_compounds": int(
            query_pool["pubchem_cid"].astype(str).nunique()
        ),
        "n_target_side_unique_compounds": int(
            target_pool["pubchem_cid"].astype(str).nunique()
        ),
        "n_shared_genes": int(n_shared_genes),
        "n_raw_shared_genes": int(n_shared_genes),
    }


def _retrieval_fields(
    metrics: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "n_positive_candidates": int(metrics["n_positives"]),
        "n_negative_candidates": int(metrics["n_negatives"]),
        "observed_best_positive_rank": metrics["best_rank"],
        "observed_normalized_best_positive_rank": metrics["normalized_rank"],
        "observed_recall_at_1": metrics["recall_at_1"],
        "observed_auroc": metrics["auroc"],
    }


def _injected_candidate_metrics(
    query_scores: np.ndarray,
    candidate_similarity: float,
) -> dict[str, float]:
    if not np.isfinite(candidate_similarity):
        return {
            "best_rank": float("nan"),
            "normalized_rank": float("nan"),
            "recall_at_1": float("nan"),
            "auroc": float("nan"),
        }
    scores = np.concatenate(
        [
            np.asarray(query_scores, dtype=np.float64),
            np.asarray([candidate_similarity], dtype=np.float64),
        ]
    )
    positive = np.zeros(len(scores), dtype=bool)
    positive[-1] = True
    return summarize_retrieval_scores(scores, positive)


def _strict_enriched_rows_for_direction(
    *,
    records: dict[tuple[str, ...], dict[str, Any]],
    dataset_a: str,
    dataset_b: str,
    direction: str,
    query_dataset: str,
    target_dataset: str,
    cell_type: str,
    time_key: str,
    query_pool: pd.DataFrame,
    target_pool: pd.DataFrame,
    query_matrix: np.ndarray,
    target_matrix: np.ndarray,
    strict_map: Mapping[str, set[str]],
    scale_variant: str,
    n_shared_genes: int,
    max_dose_fold: float,
    max_peers: int,
    sampling_seed: int,
    similarity_metrics: tuple[str, ...] = RETRIEVAL_SIMILARITIES,
    include_within_control: bool = True,
    representation: str = "logFC",
    centroid_query_matrix: Optional[np.ndarray] = None,
    centroid_target_matrix: Optional[np.ndarray] = None,
    precomputed_score_matrix: Optional[np.ndarray] = None,
    precomputed_valid_queries: Optional[np.ndarray] = None,
    precomputed_valid_targets: Optional[np.ndarray] = None,
    precomputed_source_score_matrix: Optional[np.ndarray] = None,
    eligibility_valid_queries: Optional[np.ndarray] = None,
    eligibility_valid_targets: Optional[np.ndarray] = None,
    population_counts: tuple[Optional[int], Optional[int]] = (None, None),
    eligibility_query_matrix: Optional[np.ndarray] = None,
    eligibility_target_matrix: Optional[np.ndarray] = None,
    progress: Optional[Callable[[str, str], None]] = None,
    dot_then_divide_peers: bool = False,
) -> None:
    # Peers and centroids use dot-then-divide cosine for logFC, sign(logFC) and moderated-t;
    # z-score peers normalize first.
    score_peers = peer_vector_scores if dot_then_divide_peers else score_vector_against_matrix
    score_pair = peer_vector_pair if dot_then_divide_peers else score_vector_pair
    query_obs_ids = query_pool["obs_id"].astype(str).to_numpy()
    query_compounds = query_pool["pubchem_cid"].astype(str).to_numpy()
    query_dose_keys = query_pool["dose_key"].astype(str).to_numpy()
    query_doses = query_pool["pert_dose_uM"].to_numpy(dtype=np.float64)
    target_obs_ids = target_pool["obs_id"].astype(str).to_numpy()
    target_compounds = target_pool["pubchem_cid"].astype(str).to_numpy()
    target_dose_keys = target_pool["dose_key"].astype(str).to_numpy()
    target_doses = target_pool["pert_dose_uM"].to_numpy(dtype=np.float64)

    for metric in similarity_metrics:
        if precomputed_score_matrix is None:
            score_matrix, valid_queries, valid_targets = similarity_matrix(
                query_matrix,
                target_matrix,
                metric,
            )
        else:
            if len(similarity_metrics) != 1:
                raise ValueError(
                    "Precomputed retrieval scores require exactly one metric"
                )
            if (
                precomputed_valid_queries is None
                or precomputed_valid_targets is None
            ):
                raise ValueError(
                    "Precomputed retrieval scores require validity masks"
                )
            score_matrix = np.asarray(
                precomputed_score_matrix,
                dtype=np.float64,
            )
            valid_queries = np.asarray(
                precomputed_valid_queries,
                dtype=bool,
            )
            valid_targets = np.asarray(
                precomputed_valid_targets,
                dtype=bool,
            )

        if (
            eligibility_valid_queries is not None
            and eligibility_valid_targets is not None
        ):
            eligibility_queries = np.asarray(
                eligibility_valid_queries,
                dtype=bool,
            )
            eligibility_targets = np.asarray(
                eligibility_valid_targets,
                dtype=bool,
            )
        elif (
            eligibility_query_matrix is not None
            and eligibility_target_matrix is not None
        ):
            _, eligibility_queries, eligibility_targets = similarity_matrix(
                eligibility_query_matrix,
                eligibility_target_matrix,
                metric,
            )
        else:
            eligibility_queries = None
            eligibility_targets = None

        if (
            eligibility_queries is not None
            and eligibility_targets is not None
        ):
            if (
                not bool(np.all(valid_queries[eligibility_queries]))
                or not bool(np.all(valid_targets[eligibility_targets]))
            ):
                raise AssertionError(
                    f"Z-score {scale_variant} created an invalid signature in "
                    f"raw-valid {metric} retrieval geometry"
                )
            valid_queries = eligibility_queries
            target_indices = np.flatnonzero(eligibility_targets)
        else:
            target_indices = np.flatnonzero(valid_targets)
        if len(target_indices) < MIN_TARGET_CANDIDATES:
            continue
        filtered_obs = target_obs_ids[target_indices]
        filtered_compounds = target_compounds[target_indices]
        filtered_dose_keys = target_dose_keys[target_indices]
        filtered_doses = target_doses[target_indices]
        filtered_matrix = target_matrix[target_indices]
        query_count = len(query_pool)
        progress_interval = max(1, (query_count + 9) // 10)
        for query_idx in range(len(query_pool)):
            if (
                progress is not None
                and query_idx % progress_interval == 0
            ):
                progress(
                    "query_progress",
                    (
                        f"scale={scale_variant} metric={metric} "
                        f"direction={direction} "
                        f"processed={query_idx}/{query_count}"
                    ),
                )
            if not valid_queries[query_idx]:
                continue
            query_scores = score_matrix[query_idx, target_indices]
            positive = strict_positive_mask(
                query_obs_ids[query_idx],
                filtered_obs,
                strict_map,
            )
            observed = summarize_retrieval_scores(query_scores, positive)
            if (
                observed["n_positives"] == 0
                or observed["n_negatives"] == 0
            ):
                continue
            positive_indices = np.flatnonzero(positive)
            positive_scores = query_scores[positive_indices]
            best_target_idx = int(
                positive_indices[int(np.argmax(positive_scores))]
            )
            observed_best_score = float(query_scores[best_target_idx])
            best_target_dose = float(filtered_doses[best_target_idx])
            best_target_dose_key = str(
                filtered_dose_keys[best_target_idx]
            )

            source_mask = (
                (query_dose_keys == query_dose_keys[query_idx])
                & (query_compounds != query_compounds[query_idx])
                & valid_queries
            )
            source_all_rows = np.flatnonzero(source_mask)
            source_offsets = select_peer_indices(
                len(source_all_rows),
                max_peers,
                "|".join(
                    [
                        str(query_dataset),
                        str(target_dataset),
                        str(cell_type),
                        str(time_key),
                        str(query_obs_ids[query_idx]),
                        str(query_dose_keys[query_idx]),
                        str(query_compounds[query_idx]),
                        "source",
                    ]
                ),
                sampling_seed=sampling_seed,
            )
            source_rows = source_all_rows[source_offsets]
            source_scores = (
                (
                    np.asarray(
                        precomputed_source_score_matrix[
                            query_idx,
                            source_rows,
                        ],
                        dtype=np.float64,
                    )
                    if precomputed_source_score_matrix is not None
                    else score_peers(
                        query_matrix[query_idx],
                        query_matrix[source_rows],
                        metric,
                    )
                )
                if len(source_rows)
                else np.empty(0, dtype=np.float64)
            )
            source_summary = retrieval_peer_summary(
                observed_best_score,
                source_scores,
                "source_individual",
            )
            source_centroid_similarity = float("nan")
            if len(source_all_rows):
                source_centroid_similarity = score_pair(
                    query_matrix[query_idx],
                    (query_matrix[source_all_rows].mean(axis=0, dtype=np.float64)
                     if centroid_query_matrix is None else
                     np.sign(centroid_query_matrix[source_all_rows].mean(axis=0, dtype=np.float64))),
                    metric,
                )

            target_mask = (
                (filtered_dose_keys == best_target_dose_key)
                & (filtered_compounds != query_compounds[query_idx])
            )
            target_all_rows = np.flatnonzero(target_mask)
            target_offsets = select_peer_indices(
                len(target_all_rows),
                max_peers,
                "|".join(
                    [
                        str(query_dataset),
                        str(target_dataset),
                        str(cell_type),
                        str(time_key),
                        str(query_obs_ids[query_idx]),
                        str(best_target_dose_key),
                        str(query_compounds[query_idx]),
                        "target",
                    ]
                ),
                sampling_seed=sampling_seed,
            )
            target_rows = target_all_rows[target_offsets]
            target_scores = query_scores[target_rows]
            target_summary = retrieval_peer_summary(
                observed_best_score,
                target_scores,
                "target_individual",
            )
            target_centroid_similarity = float("nan")
            if len(target_all_rows):
                target_centroid_similarity = score_pair(
                    query_matrix[query_idx],
                    (filtered_matrix[target_all_rows].mean(axis=0, dtype=np.float64)
                     if centroid_target_matrix is None else
                     np.sign(centroid_target_matrix[target_indices][target_all_rows].mean(axis=0, dtype=np.float64))),
                    metric,
                )
            source_centroid_metrics = _injected_candidate_metrics(
                query_scores,
                source_centroid_similarity,
            )
            target_centroid_metrics = _injected_candidate_metrics(
                query_scores,
                target_centroid_similarity,
            )
            source_peer_metrics = expected_single_signature_metrics(
                query_scores,
                source_scores,
            )
            target_peer_metrics = expected_single_signature_metrics(
                query_scores,
                target_scores,
            )

            within_metrics = {
                "best_rank": float("nan"),
                "normalized_rank": float("nan"),
                "recall_at_1": float("nan"),
                "auroc": float("nan"),
            }
            if include_within_control:
                within_mask = np.arange(len(query_pool)) != query_idx
                within_mask &= valid_queries
                within_matrix = query_matrix[within_mask]
                within_compounds = query_compounds[within_mask]
                within_doses = query_doses[within_mask]
                if len(within_matrix) >= 2:
                    within_scores = score_vector_against_matrix(
                        query_matrix[query_idx],
                        within_matrix,
                        metric,
                    )
                    within_positive = (
                        within_compounds == query_compounds[query_idx]
                    ) & (
                        raw_dose_fold_differences(
                            query_doses[query_idx],
                            within_doses,
                        )
                        <= max_dose_fold + 1e-12
                    )
                    within_metrics = summarize_retrieval_scores(
                        within_scores,
                        within_positive,
                    )

            null = exact_cross_assay_null(
                n_candidates=len(query_scores),
                n_positives=int(observed["n_positives"]),
            )
            calibration = exact_null_rank_calibration(
                n_candidates=len(query_scores),
                n_positives=int(observed["n_positives"]),
                best_rank=observed["best_rank"],
            )
            minimum_calibration = exact_null_rank_calibration(
                n_candidates=len(query_scores),
                n_positives=int(observed["n_positives"]),
                best_rank=1.0,
            )
            record = _base_record(
                dataset_a=dataset_a,
                dataset_b=dataset_b,
                direction=direction,
                query_dataset=query_dataset,
                target_dataset=target_dataset,
                cell_type=cell_type,
                time_key=time_key,
                query_pool=query_pool,
                query_idx=query_idx,
                target_pool=target_pool.iloc[target_indices].reset_index(
                    drop=True
                ),
                representation=representation,
                retrieval_variant="strict_matched_condition",
                similarity_metric=metric,
                scale_variant=scale_variant,
                n_shared_genes=n_shared_genes,
            )
            key = _record_key(record)
            if key in records:
                records[key].update(record)
                record = records[key]
            else:
                records[key] = record
            record.update(_retrieval_fields(observed))
            record.update(source_summary)
            record.update(target_summary)
            record.update(
                {
                    "target_pool_size_before_similarity_filter": int(
                        len(target_pool)
                    ),
                    "n_zero_norm_target_candidates": int(
                        len(target_pool) - len(target_indices)
                    ),
                    "centroid_baseline_peer_count": int(
                        len(source_all_rows)
                    ),
                    "single_signature_baseline_peer_count": int(
                        source_peer_metrics["n_peers"]
                    ),
                    "n_excluded_single_signature_peers": int(
                        len(source_all_rows)
                        - int(source_peer_metrics["n_peers"])
                    ),
                    "source_centroid_peer_count": int(len(source_all_rows)),
                    "target_centroid_peer_count": int(len(target_all_rows)),
                    "source_individual_total_count": int(
                        len(source_all_rows)
                    ),
                    "source_individual_scored_count": int(len(source_rows)),
                    "target_individual_total_count": int(
                        len(target_all_rows)
                    ),
                    "target_individual_scored_count": int(len(target_rows)),
                    "left_population_row_count": population_counts[0],
                    "right_population_row_count": population_counts[1],
                    "observed_best_positive_similarity": observed_best_score,
                    "observed_mean_positive_similarity": float(
                        np.mean(positive_scores)
                    ),
                    "observed_sd_positive_similarity": (
                        float(np.std(positive_scores, ddof=1))
                        if len(positive_scores) >= 2
                        else float("nan")
                    ),
                    "best_target_obs_id": str(
                        filtered_obs[best_target_idx]
                    ),
                    "best_target_dose_key": best_target_dose_key,
                    "best_target_dose_uM": best_target_dose,
                    "best_target_dose_fold_difference": float(
                        max(
                            query_doses[query_idx] / best_target_dose,
                            best_target_dose / query_doses[query_idx],
                        )
                    ),
                    "source_centroid_similarity": source_centroid_similarity,
                    "target_centroid_similarity": target_centroid_similarity,
                    "centroid_baseline_candidate_similarity": (
                        source_centroid_similarity
                    ),
                    "target_decoy_null_normalized_best_positive_rank": (
                        null["normalized_rank"]
                    ),
                    "target_decoy_null_best_positive_rank": (
                        null["best_rank"]
                    ),
                    "target_decoy_null_recall_at_1": null["recall_at_1"],
                    "target_decoy_null_auroc": null["auroc"],
                    "exact_null_mid_p": exact_null_mid_p(
                        n_candidates=len(query_scores),
                        n_positives=int(observed["n_positives"]),
                        best_rank=int(observed["best_rank"]),
                    ),
                    "null_pit": calibration["null_pit"],
                    "null_p_value": calibration["null_p_value"],
                    "min_achievable_null_p_value": (
                        minimum_calibration["null_p_value"]
                    ),
                    "null_pit_below_half": float(
                        calibration["null_pit"] < 0.5
                    ),
                    "observed_auroc_minus_half": (
                        float(observed["auroc"]) - 0.5
                    ),
                    "within_source_best_positive_rank": (
                        within_metrics["best_rank"]
                    ),
                    "within_source_normalized_best_positive_rank": (
                        within_metrics["normalized_rank"]
                    ),
                    "within_source_recall_at_1": (
                        within_metrics["recall_at_1"]
                    ),
                    "within_source_auroc": within_metrics["auroc"],
                    "centroid_baseline_best_positive_rank": (
                        source_centroid_metrics["best_rank"]
                    ),
                    "centroid_baseline_normalized_best_positive_rank": (
                        source_centroid_metrics["normalized_rank"]
                    ),
                    "centroid_baseline_recall_at_1": (
                        source_centroid_metrics["recall_at_1"]
                    ),
                    "centroid_baseline_auroc": (
                        source_centroid_metrics["auroc"]
                    ),
                    "single_signature_baseline_best_positive_rank": (
                        source_peer_metrics["best_rank"]
                    ),
                    "single_signature_baseline_normalized_best_positive_rank": (
                        source_peer_metrics["normalized_rank"]
                    ),
                    "single_signature_baseline_recall_at_1": (
                        source_peer_metrics["recall_at_1"]
                    ),
                    "single_signature_baseline_auroc": (
                        source_peer_metrics["auroc"]
                    ),
                    "within_dataset_positive_control_best_positive_rank": (
                        within_metrics["best_rank"]
                    ),
                    "within_dataset_positive_control_normalized_best_positive_rank": (
                        within_metrics["normalized_rank"]
                    ),
                    "within_dataset_positive_control_recall_at_1": (
                        within_metrics["recall_at_1"]
                    ),
                    "within_dataset_positive_control_auroc": (
                        within_metrics["auroc"]
                    ),
                    "delta_observed_vs_source_centroid": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            source_centroid_similarity,
                        )
                    ),
                    "delta_observed_vs_target_centroid": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            target_centroid_similarity,
                        )
                    ),
                    "delta_observed_vs_source_individual_mean": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            source_summary[
                                "source_individual_mean_similarity"
                            ],
                        )
                    ),
                    "delta_observed_vs_target_individual_mean": (
                        cross_source_core.difference_if_both_defined(
                            observed_best_score,
                            target_summary[
                                "target_individual_mean_similarity"
                            ],
                        )
                    ),
                    "delta_observed_vs_target_decoy_null_rank": (
                        cross_source_core.difference_if_both_defined(
                            observed["normalized_rank"],
                            null["normalized_rank"],
                        )
                    ),
                    "delta_observed_vs_target_decoy_null_recall_at_1": (
                        cross_source_core.difference_if_both_defined(
                            observed["recall_at_1"],
                            null["recall_at_1"],
                        )
                    ),
                    "delta_observed_vs_target_decoy_null_auroc": (
                        cross_source_core.difference_if_both_defined(
                            observed["auroc"],
                            null["auroc"],
                        )
                    ),
                    "delta_vs_target_decoy_null_normalized_best_positive_rank": (
                        cross_source_core.difference_if_both_defined(
                            observed["normalized_rank"],
                            null["normalized_rank"],
                        )
                    ),
                    "delta_vs_target_decoy_null_recall_at_1": (
                        cross_source_core.difference_if_both_defined(
                            observed["recall_at_1"],
                            null["recall_at_1"],
                        )
                    ),
                    "delta_vs_target_decoy_null_auroc": (
                        cross_source_core.difference_if_both_defined(
                            observed["auroc"],
                            null["auroc"],
                        )
                    ),
                    "delta_vs_random_normalized_best_positive_rank": (
                        cross_source_core.difference_if_both_defined(
                            observed["normalized_rank"],
                            null["normalized_rank"],
                        )
                    ),
                }
            )
            metric_variants = {
                "source_centroid": source_centroid_metrics,
                "target_centroid": target_centroid_metrics,
                "source_peer": source_peer_metrics,
                "target_peer": target_peer_metrics,
                "within_dataset_positive_control": within_metrics,
            }
            for variant, values in metric_variants.items():
                for field in (
                    "best_rank",
                    "normalized_rank",
                    "recall_at_1",
                    "auroc",
                ):
                    record[f"{variant}_{field}"] = values.get(
                        field,
                        float("nan"),
                    )
                for field in ("normalized_rank", "recall_at_1", "auroc"):
                    record[f"delta_vs_{variant}_{field}"] = (
                        cross_source_core.difference_if_both_defined(
                            observed[field],
                            values.get(field, float("nan")),
                        )
                    )
            for alias, values in (
                ("centroid_baseline", source_centroid_metrics),
                ("single_signature_baseline", source_peer_metrics),
                ("within_dataset_positive_control", within_metrics),
            ):
                for field, output_field in (
                    ("normalized_rank", "normalized_best_positive_rank"),
                    ("recall_at_1", "recall_at_1"),
                    ("auroc", "auroc"),
                ):
                    record[f"delta_vs_{alias}_{output_field}"] = (
                        cross_source_core.difference_if_both_defined(
                            observed[field],
                            values.get(field, float("nan")),
                        )
                    )
        if progress is not None:
            progress(
                "query_progress",
                (
                    f"scale={scale_variant} metric={metric} "
                    f"direction={direction} "
                    f"processed={query_count}/{query_count}"
                ),
            )


def _score_selected_scale(
    *,
    records: dict[tuple[str, ...], dict[str, Any]],
    dataset_a: str,
    dataset_b: str,
    cell_type: str,
    time_key: str,
    left_pool: pd.DataFrame,
    right_pool: pd.DataFrame,
    left_matrix: np.ndarray,
    right_matrix: np.ndarray,
    strict_maps: Mapping[str, Mapping[str, set[str]]],
    scale_variant: str,
    n_shared_genes: int,
    max_dose_fold: float,
    max_peers: int,
    sampling_seed: int,
    similarity_metrics: tuple[str, ...],
    population_counts: tuple[Optional[int], Optional[int]] = (None, None),
    eligibility_masks: Optional[
        Mapping[str, tuple[np.ndarray, np.ndarray]]
    ] = None,
    progress: Optional[Callable[[str, str], None]] = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    raw_validity: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for metric in similarity_metrics:
        if progress is not None:
            progress(
                "similarity",
                f"scale={scale_variant} metric={metric}",
            )
        cross_scores, valid_left, valid_right = similarity_matrix(
            left_matrix,
            right_matrix,
            metric,
        )
        left_scores, left_self_valid, _ = similarity_matrix(
            left_matrix,
            left_matrix,
            metric,
        )
        right_scores, right_self_valid, _ = similarity_matrix(
            right_matrix,
            right_matrix,
            metric,
        )
        if not np.array_equal(valid_left, left_self_valid):
            raise AssertionError(
                f"Left {metric} validity differs between cross- and "
                "within-source geometry"
            )
        if not np.array_equal(valid_right, right_self_valid):
            raise AssertionError(
                f"Right {metric} validity differs between cross- and "
                "within-source geometry"
            )
        raw_validity[metric] = (valid_left.copy(), valid_right.copy())
        eligibility_left = None
        eligibility_right = None
        if eligibility_masks is not None:
            eligibility_left, eligibility_right = eligibility_masks[metric]

        _strict_enriched_rows_for_direction(
            records=records,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            direction="A_to_B",
            query_dataset=dataset_a,
            target_dataset=dataset_b,
            cell_type=cell_type,
            time_key=time_key,
            query_pool=left_pool,
            target_pool=right_pool,
            query_matrix=left_matrix,
            target_matrix=right_matrix,
            strict_map=strict_maps["A_to_B"],
            scale_variant=scale_variant,
            n_shared_genes=n_shared_genes,
            max_dose_fold=max_dose_fold,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
            similarity_metrics=(metric,),
            include_within_control=False,
            precomputed_score_matrix=cross_scores,
            precomputed_valid_queries=valid_left,
            precomputed_valid_targets=valid_right,
            precomputed_source_score_matrix=left_scores,
            eligibility_valid_queries=eligibility_left,
            eligibility_valid_targets=eligibility_right,
            population_counts=population_counts,
            progress=progress,
        )
        _strict_enriched_rows_for_direction(
            records=records,
            dataset_a=dataset_a,
            dataset_b=dataset_b,
            direction="B_to_A",
            query_dataset=dataset_b,
            target_dataset=dataset_a,
            cell_type=cell_type,
            time_key=time_key,
            query_pool=right_pool,
            target_pool=left_pool,
            query_matrix=right_matrix,
            target_matrix=left_matrix,
            strict_map=strict_maps["B_to_A"],
            scale_variant=scale_variant,
            n_shared_genes=n_shared_genes,
            max_dose_fold=max_dose_fold,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
            similarity_metrics=(metric,),
            include_within_control=False,
            precomputed_score_matrix=cross_scores.T,
            precomputed_valid_queries=valid_right,
            precomputed_valid_targets=valid_left,
            precomputed_source_score_matrix=right_scores,
            eligibility_valid_queries=eligibility_right,
            eligibility_valid_targets=eligibility_left,
            population_counts=population_counts,
            progress=progress,
        )
    return raw_validity


def peer_cap(settings: Mapping[str, Any], computation: str, similarity: str) -> int:
    """Preserve the per-representation caps supported by saved retrieval results."""
    override = settings.get("max_baseline_peers")
    if override is not None:
        return int(override)
    capped = computation in REPRESENTATIONS or computation == "z-score-dataset"
    return 256 if capped and similarity != "negative_l2" else 0


def _peer_pool(frame, source, positions, *, moderated=False):
    pool = build_logfc_pool_matrices(frame, source, positions)
    if not moderated:
        return pool.frame, pool.logfc, pool.logfc, pool.diagnostics
    selected, effects = [], []
    for index, row in pool.frame.iterrows():
        try:
            effects.append(source.get_vector(
                str(row["obs_id"]), "t", pubchem_cid=str(row["pubchem_cid"]),
                dose_key=str(row["dose_key"]), time_key=str(row["time_key"]),
            )[positions])
            selected.append(index)
        except KeyError as exc:
            pool.diagnostics.append({"obs_id": str(row["obs_id"]), "reason": "missing_t_row", "error": str(exc)})
    return (pool.frame.iloc[selected].reset_index(drop=True), pool.logfc[selected],
            np.asarray(effects, dtype=np.float64).reshape(len(effects), len(positions)), pool.diagnostics)


def _notebook_peer_aliases(record):
    """Retain the notebook's metric labels used by the raw/sign/t CI hashes."""
    for side in ("source", "target"):
        prefix = f"{side}_individual"
        record[f"{prefix}_n"] = record[f"{prefix}_n_peers"]
        for metric, suffix in (("normalized_rank", "normalized_best_positive_rank"),
                               ("recall_at_1", "recall_at_1"), ("auroc", "auroc")):
            record[f"{prefix}_{suffix}"] = record[f"{side}_peer_{metric}"]
        record[f"delta_observed_vs_{side}_individual_rank"] = record[f"delta_vs_{side}_peer_normalized_rank"]


def _strict_notebook_rows(*, frame, left, right, left_source, right_source,
                          left_positions, right_positions, computations,
                          similarities, z_score_catalog):
    """Notebook strict geometry: finite logFC, t and adjusted p on both sides.

    These rows supply whole-stratum centroids. Individual peers have independent
    eligibility masks, so selecting t or z-scores cannot change raw peer results.
    """
    if any(
        len(pool.frame) < MIN_TARGET_CANDIDATES
        or pool.frame["pubchem_cid"].nunique() < MIN_UNIQUE_COMPOUNDS_PER_SIDE
        for pool in (left, right)
    ):
        return []
    finite = np.logical_and.reduce([
        np.isfinite(values).all(axis=0)
        for pool in (left, right) for values in (pool.logfc, pool.t, pool.adj_p)
    ])
    if finite.sum() < 2:
        return []
    a, b, cell, time = (str(frame.iloc[0][key]) for key in CONTEXT_COLUMNS)
    maps = strict_positive_maps(frame)
    rows = []
    for computation in computations:
        scale = Z_SCORE_SCALES.get(computation, RAW_SCALE_VARIANT)
        representation = REPRESENTATIONS.get(computation, "logFC")
        matrices, centroids = [], []
        for pool, source, positions in ((left, left_source, left_positions),
                                         (right, right_source, right_positions)):
            values = pool.t if computation == "moderated-t" else pool.logfc
            baseline = pool.centroid_t if computation == "moderated-t" else pool.centroid_logfc
            if computation in Z_SCORE_SCALES:
                values = z_score_catalog.standardize_aligned_matrix(source, values, positions, scale_variant=scale)
                baseline = z_score_catalog.standardize_aligned_matrix(source, baseline, positions, scale_variant=scale)
                if not np.isfinite(values[:, finite]).all():
                    raise AssertionError(f"{scale} removed notebook raw-valid genes")
            elif computation == "direction":
                values, baseline = np.sign(values), np.sign(baseline)
            matrices.append(values[:, finite])
            centroids.append(baseline[:, finite])
        for side, query, target in ((0, left, right), (1, right, left)):
            direction = "A_to_B" if side == 0 else "B_to_A"
            for metric in similarities:
                scores, valid_queries, valid_targets = similarity_matrix(matrices[side], matrices[1-side], metric)
                target_indices = np.flatnonzero(valid_targets)
                if len(target_indices) < MIN_TARGET_CANDIDATES:
                    continue
                target_ids = target.frame.iloc[target_indices]["obs_id"].astype(str).to_numpy()
                for index in np.flatnonzero(valid_queries):
                    query_scores = scores[index, target_indices]
                    positive = strict_positive_mask(str(query.frame.iloc[index]["obs_id"]), target_ids, maps[direction])
                    observed = summarize_retrieval_scores(query_scores, positive)
                    if not observed["n_positives"] or not observed["n_negatives"]:
                        continue
                    baseline_score = float("nan")
                    if query.centroid_peer_counts[index] > 0:
                        baseline_score = score_vector_pair(matrices[side][index], centroids[side][index], metric)
                    baseline = _injected_candidate_metrics(query_scores, baseline_score)
                    record = _base_record(
                        dataset_a=a, dataset_b=b, direction=direction,
                        query_dataset=a if side == 0 else b, target_dataset=b if side == 0 else a,
                        cell_type=cell, time_key=time, query_pool=query.frame, query_idx=index,
                        target_pool=target.frame.iloc[target_indices], representation=representation,
                        retrieval_variant="strict_matched_condition", similarity_metric=metric,
                        scale_variant=scale, n_shared_genes=int(finite.sum()),
                    )
                    record.update(
                        strict_available=True, strict_row_order=len(rows),
                        strict_observed_normalized_best_positive_rank=observed["normalized_rank"],
                        strict_baseline_normalized_best_positive_rank=baseline["normalized_rank"],
                        strict_delta_vs_baseline_normalized_best_positive_rank=(
                            cross_source_core.difference_if_both_defined(observed["normalized_rank"], baseline["normalized_rank"])
                        ),
                        strict_n_shared_genes=int(finite.sum()),
                    )
                    rows.append(record)
    return rows


def _context_records(frame, *, config, catalog, z_score_catalog, progress=None):
    settings = config["settings"]
    computations = tuple(settings["computations"])
    similarities = tuple(settings["similarity_metrics"])
    a, b, cell, time = (str(frame.iloc[0][key]) for key in CONTEXT_COLUMNS)
    pools = [pool_frame_for_context(read_overlap_metadata(config, name), cell, time) for name in (a, b)]
    diagnostics = []
    if any(len(pool) < MIN_TARGET_CANDIDATES or pool["pubchem_cid"].nunique() < MIN_UNIQUE_COMPOUNDS_PER_SIDE for pool in pools):
        return pd.DataFrame(columns=OUTPUT_COLUMNS), [{"analysis": ANALYSIS, "dataset_a": a, "dataset_b": b,
            "cell_type": cell, "time_key": time, "reason": "insufficient_candidates_or_unique_compounds"}]
    sources = [catalog.get_line_source(name, cell) for name in (a, b)]
    genes, left_positions, right_positions = catalog.shared_gene_positions(*sources)
    if len(genes) < 2:
        raise ValueError(f"Fewer than two shared genes for {a}/{b}/{cell}")
    positions = (left_positions, right_positions)
    raw = [_peer_pool(pool, source, pos) for pool, source, pos in zip(pools, sources, positions)]
    for values in raw:
        diagnostics.extend(values[3])
    maps = strict_positive_maps(frame)
    max_fold = float(cross_source_core.DEFAULT_MATCH_SETTINGS.max_dose_fold_difference)
    common = dict(dataset_a=a, dataset_b=b, cell_type=cell, time_key=time, max_dose_fold=max_fold,
                  sampling_seed=int(settings["peer_sampling_seed"]), progress=progress)
    records = {}
    for computation in computations:
        current = raw
        if computation == "moderated-t":
            current = [_peer_pool(pool, source, pos, moderated=True) for pool, source, pos in zip(pools, sources, positions)]
            for values in current:
                diagnostics.extend(values[3])
        if min(len(values[0]) for values in current) < MIN_TARGET_CANDIDATES:
            continue
        finite = np.logical_and.reduce([np.isfinite(values[1]).all(axis=0) for values in current])
        if computation == "moderated-t":
            finite &= np.logical_and.reduce([np.isfinite(values[2]).all(axis=0) for values in current])
        if finite.sum() < 2:
            continue
        raw_matrices = [values[1][:, finite] for values in current]
        matrices = [values[2][:, finite] for values in current]
        scale = Z_SCORE_SCALES.get(computation, RAW_SCALE_VARIANT)
        representation = REPRESENTATIONS.get(computation, "logFC")
        if computation == "direction":
            matrices = [np.sign(values) for values in matrices]
        elif computation in Z_SCORE_SCALES:
            matrices = [z_score_catalog.standardize_aligned_matrix(source, values[1], pos, scale_variant=scale)[:, finite]
                        for source, values, pos in zip(sources, current, positions)]
            if not all(np.isfinite(values).all() for values in matrices):
                raise AssertionError(f"{scale} removed reviewer raw-valid genes")
        computation_records = {}
        for metric in similarities:
            cap = peer_cap(settings, computation, metric)
            if computation in Z_SCORE_SCALES:
                _, valid_left, valid_right = similarity_matrix(*raw_matrices, metric)
                _score_selected_scale(
                    **common, records=computation_records,
                    left_pool=current[0][0], right_pool=current[1][0],
                    left_matrix=matrices[0], right_matrix=matrices[1], strict_maps=maps,
                    scale_variant=scale, n_shared_genes=int(finite.sum()), max_peers=cap,
                    similarity_metrics=(metric,), eligibility_masks={metric: (valid_left, valid_right)},
                    population_counts=tuple(z_score_catalog.get_stats(source, scale_variant=scale).population_row_count for source in sources),
                )
            else:
                for side in (0, 1):
                    direction = "A_to_B" if side == 0 else "B_to_A"
                    _strict_enriched_rows_for_direction(
                        **common, records=computation_records, direction=direction,
                        query_dataset=a if side == 0 else b, target_dataset=b if side == 0 else a,
                        query_pool=current[side][0], target_pool=current[1-side][0],
                        query_matrix=matrices[side], target_matrix=matrices[1-side], strict_map=maps[direction],
                        scale_variant=scale, n_shared_genes=int(finite.sum()), max_peers=cap,
                        similarity_metrics=(metric,), include_within_control=False, representation=representation,
                        centroid_query_matrix=raw_matrices[side] if computation == "direction" else None,
                        centroid_target_matrix=raw_matrices[1-side] if computation == "direction" else None,
                        dot_then_divide_peers=True,
                    )
        rows = list(computation_records.values())
        # Reviewer workers sort their rows by the original identity columns before merging.
        if computation in Z_SCORE_SCALES and rows:
            rows = pd.DataFrame(rows).sort_values(IDENTITY_COLUMNS).to_dict("records")
        for order, record in enumerate(rows):
            if computation in REPRESENTATIONS:
                _notebook_peer_aliases(record)
            record.update(peer_available=True, peer_row_order=order)
            records[_record_key(record)] = record

    # Load notebook geometry independently so t/adjusted-p missing values never
    # remove genes or candidates from the reviewer z-score peer computation.
    strict_pools = [build_pool_matrices(pool, source, pos, adj_layer_name=cross_source_core.first_available_layer(
        source, ADJ_PVALUE_LAYER_PREFERENCES)) for pool, source, pos in zip(pools, sources, positions)]
    for pool in strict_pools:
        diagnostics.extend(pool.diagnostics)
    for strict in _strict_notebook_rows(
        frame=frame, left=strict_pools[0], right=strict_pools[1], left_source=sources[0], right_source=sources[1],
        left_positions=left_positions, right_positions=right_positions, computations=computations,
        similarities=similarities, z_score_catalog=z_score_catalog,
    ):
        key = _record_key(strict)
        if key in records:
            records[key].update({name: value for name, value in strict.items() if name.startswith("strict_")})
        else:
            records[key] = strict
    result = pd.DataFrame(records.values()).reindex(columns=OUTPUT_COLUMNS)
    for column in ("strict_available", "peer_available"):
        result[column] = result[column].eq(True)
    return result.sort_values(IDENTITY_COLUMNS).reset_index(drop=True), diagnostics


OUTPUT_COLUMNS = [*RETRIEVAL_OUTPUT_COLUMNS,
    "strict_available", "peer_available", "strict_row_order", "peer_row_order", "strict_n_shared_genes",
    "strict_observed_normalized_best_positive_rank", "strict_baseline_normalized_best_positive_rank",
    "strict_delta_vs_baseline_normalized_best_positive_rank",
    *[f"{side}_individual_{field}" for side in ("source", "target")
      for field in ("n", "normalized_best_positive_rank", "recall_at_1", "auroc")],
    "delta_observed_vs_source_individual_rank", "delta_observed_vs_target_individual_rank",
]


def score_task(task: TaskSpec, config: Mapping[str, Any]):
    catalog = make_worker_catalog(config)
    def progress(phase, detail=""):
        report_task_progress(config, task, phase=phase, detail=detail)
    try:
        metrics, diagnostics = _context_records(
            read_task_input(config, task), config=config, catalog=catalog,
            z_score_catalog=make_worker_z_score_catalog(config, catalog) if config["settings"]["z_score_scale_variants"] else None,
            progress=progress,
        )
        # Notebooks sort contexts; the reviewer workers merge in task order.
        # Preserve both because bootstrap compound positions follow row order.
        notebook_prefix = "\x1f".join(task.key) + "\x1f"
        reviewer_prefix = f"{task.task_id:08d}:"
        for family in ("strict", "peer"):
            column = f"{family}_row_order"
            prefixes = pd.Series(notebook_prefix, index=metrics.index)
            if family == "peer":
                prefixes.loc[metrics["scale_variant"].ne(RAW_SCALE_VARIANT)] = reviewer_prefix
            metrics[column] = [
                "" if pd.isna(value) else f"{prefix}{int(value):09d}"
                for prefix, value in zip(prefixes, metrics[column])
            ]
        for record in diagnostics:
            record["task_id"] = task.task_id
        progress("finished", f"rows={len(metrics)} diagnostics={len(diagnostics)}")
        return metrics, diagnostic_frame(diagnostics)
    finally:
        catalog.close()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_arguments(parser, analysis=ANALYSIS, include_z_score_scales=False)
    add_computation_arguments(parser, computations=COMPUTATIONS)
    parser.set_defaults(datasets="l1000_phase1,l1000_phase2,tahoe,cigs_mce,cigs_tcm,sciplex",
                        max_baseline_peers=None, workers=PAPER_RUNTIME["workers"])
    parser.add_argument("--similarities", nargs="+", choices=RETRIEVAL_SIMILARITIES,
                        default=list(RETRIEVAL_SIMILARITIES), help="Similarity metrics to score (default: all three).")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.list_computations:
        print_computations(COMPUTATIONS)
        return 0
    computations = resolve_computations(args.compute, computations=COMPUTATIONS)
    args.required_layers_override = ("logFC", "t")
    args.require_adj_p_override = True
    settings = {
        "computations": list(computations), "similarity_metrics": list(dict.fromkeys(args.similarities)),
        "max_baseline_peers": args.max_baseline_peers, "peer_sampling_seed": args.peer_sampling_seed,
        "z_score_scale_variants": [Z_SCORE_SCALES[name] for name in computations if name in Z_SCORE_SCALES],
        "scorer_version": "table9-five-representations-v1",
    }
    metrics_path, _ = run_analysis(
        analysis=ANALYSIS, scorer_module="scripts.run_overlap_group_rep_retrieval_metrics", args=args,
        context_columns=CONTEXT_COLUMNS, rows_per_shard=None, final_metrics_name=FINAL_METRICS_NAME,
        settings=settings, code_paths=[Path(__file__), *[REPO_ROOT / "scripts" / name for name in (
            "cross_source_core.py", "cross_source_parallel.py", "cross_source_scoring.py",
            "population_zscore.py", "peer_baselines.py", "notebook_cache.py")]],
    )
    summarize_run(metrics_path, computations=computations)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
