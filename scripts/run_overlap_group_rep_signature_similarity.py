#!/usr/bin/env python3
"""Parallel, resumable Table 6 matched-signature scoring.

Select components with --compute: raw, z-score-dataset,
z-score-dataset-cell-type, moderated-t. Moderated-t retains the raw signature
metrics and adds t peer baselines using the same selected peers. A CI summary
is written beside the scored metrics after scoring completes.
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
    score_signature_baselines,
    select_source_peers,
)
from scripts.cross_source_strata import SignatureStratum
from scripts.helper_group_rep_deg_signature_ci import summarize_run
from scripts.population_zscore import (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
    POPULATION_SCALE_VARIANTS,
)


ANALYSIS = "signature"
FINAL_METRICS_NAME = "signature_scored_metrics.tsv"
MODERATED_T_ANALYSIS = "signature_moderated_t"
MODERATED_T_METRICS_NAME = "signature_moderated_t_scored_metrics.tsv"
EFFECT_LAYER = "t"
PEER_PREFIX = "pbt"
OBSERVED_KEY = "pbt_observed_spearman_t"
CONTEXT_COLUMNS = (
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "left_dose_key",
    "right_dose_key",
)
COMPUTATIONS = {
    "raw": "Raw matched-signature metrics and individual-peer baselines.",
    "z-score-dataset": (
        "Dataset-wide per-gene population-z-score signature metrics."
    ),
    "z-score-dataset-cell-type": (
        "Dataset-by-cell-type per-gene population-z-score signature metrics."
    ),
    "moderated-t": "Moderated-t signature peer baselines, retaining raw signature metrics.",
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


def _centroid(
    source: cross_source_core.LineSource,
    obs_id: str,
    lookup: Mapping[str, str],
    layer: str,
    positions: np.ndarray,
) -> Optional[np.ndarray]:
    value = source.get_baseline_vector(obs_id, layer, **lookup)
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64)[positions]


def _raw_signature_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    max_peers: Optional[int],
    sampling_seed: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
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
    global_genes, left_global_positions = catalog.global_gene_positions(
        left_source
    )
    _, right_global_positions = catalog.global_gene_positions(right_source)

    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
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
    left_t_full = left_source.get_vector(left_obs_id, "t", **left_lookup)
    right_t_full = right_source.get_vector(right_obs_id, "t", **right_lookup)
    left_logfc = left_logfc_full[left_positions]
    right_logfc = right_logfc_full[right_positions]
    left_t = left_t_full[left_positions]
    right_t = right_t_full[right_positions]
    observed = cross_source_core.score_signature_pair(
        left_logfc=left_logfc,
        right_logfc=right_logfc,
        left_t=left_t,
        right_t=right_t,
    )
    global_observed = cross_source_core.empty_score_dict()
    if global_genes.size >= 2:
        global_observed = cross_source_core.score_signature_pair(
            left_logfc=left_logfc_full[left_global_positions],
            right_logfc=right_logfc_full[right_global_positions],
            left_t=left_t_full[left_global_positions],
            right_t=right_t_full[right_global_positions],
        )

    left_centroid_logfc = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "logFC",
        left_positions,
    )
    right_centroid_logfc = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "logFC",
        right_positions,
    )
    left_centroid_t = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "t",
        left_positions,
    )
    right_centroid_t = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "t",
        right_positions,
    )
    left_centroid_logfc_global = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "logFC",
        left_global_positions,
    )
    right_centroid_logfc_global = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "logFC",
        right_global_positions,
    )
    left_centroid_t_global = _centroid(
        left_source,
        left_obs_id,
        left_lookup,
        "t",
        left_global_positions,
    )
    right_centroid_t_global = _centroid(
        right_source,
        right_obs_id,
        right_lookup,
        "t",
        right_global_positions,
    )
    left_baseline = cross_source_core.empty_score_dict()
    if left_centroid_logfc is not None and left_centroid_t is not None:
        left_baseline = cross_source_core.score_signature_pair(
            left_logfc=left_logfc,
            right_logfc=left_centroid_logfc,
            left_t=left_t,
            right_t=left_centroid_t,
        )
    right_baseline = cross_source_core.empty_score_dict()
    if right_centroid_logfc is not None and right_centroid_t is not None:
        right_baseline = cross_source_core.score_signature_pair(
            left_logfc=right_logfc,
            right_logfc=right_centroid_logfc,
            left_t=right_t,
            right_t=right_centroid_t,
        )
    left_baseline_global = cross_source_core.empty_score_dict()
    if (
        global_genes.size >= 2
        and left_centroid_logfc_global is not None
        and left_centroid_t_global is not None
    ):
        left_baseline_global = cross_source_core.score_signature_pair(
            left_logfc=left_logfc_full[left_global_positions],
            right_logfc=left_centroid_logfc_global,
            left_t=left_t_full[left_global_positions],
            right_t=left_centroid_t_global,
        )
    right_baseline_global = cross_source_core.empty_score_dict()
    if (
        global_genes.size >= 2
        and right_centroid_logfc_global is not None
        and right_centroid_t_global is not None
    ):
        right_baseline_global = cross_source_core.score_signature_pair(
            left_logfc=right_logfc_full[right_global_positions],
            right_logfc=right_centroid_logfc_global,
            left_t=right_t_full[right_global_positions],
            right_t=right_centroid_t_global,
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
    expected_left = left_source.baseline_peer_count(
        left_obs_id,
        **left_lookup,
    )
    expected_right = right_source.baseline_peer_count(
        right_obs_id,
        **right_lookup,
    )
    if left_selected.total_count != expected_left:
        raise AssertionError(
            f"Left peer set has {left_selected.total_count} rows; "
            f"centroid used {expected_left}"
        )
    if right_selected.total_count != expected_right:
        raise AssertionError(
            f"Right peer set has {right_selected.total_count} rows; "
            f"centroid used {expected_right}"
        )
    left_peers = left_selected.stratum.values[
        left_selected.row_indices
    ][:, left_positions]
    right_peers = right_selected.stratum.values[
        right_selected.row_indices
    ][:, right_positions]
    overlap_name = f"signed_overlap_t_top{cross_source_core.TOP_K}"
    left_mean_abs_t = cross_source_core.mean_available(np.abs(left_t).tolist())
    right_mean_abs_t = cross_source_core.mean_available(np.abs(right_t).tolist())
    left_mean_abs_t_global = (
        cross_source_core.mean_available(
            np.abs(left_t_full[left_global_positions]).tolist()
        )
        if global_genes.size >= 2
        else float("nan")
    )
    right_mean_abs_t_global = (
        cross_source_core.mean_available(
            np.abs(right_t_full[right_global_positions]).tolist()
        )
        if global_genes.size >= 2
        else float("nan")
    )
    record: dict[str, Any] = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "n_global_common_genes": int(global_genes.size),
        "left_mean_abs_t": left_mean_abs_t,
        "right_mean_abs_t": right_mean_abs_t,
        "pair_mean_abs_t": cross_source_core.mean_available(
            [left_mean_abs_t, right_mean_abs_t]
        ),
        "left_mean_abs_t_global": left_mean_abs_t_global,
        "right_mean_abs_t_global": right_mean_abs_t_global,
        "pair_mean_abs_t_global": cross_source_core.mean_available(
            [left_mean_abs_t_global, right_mean_abs_t_global]
        ),
        "observed_spearman_logfc": observed["spearman_logfc"],
        "observed_spearman_logfc_global": global_observed["spearman_logfc"],
        "observed_spearman_t": observed["spearman_t"],
        "observed_spearman_t_global": global_observed["spearman_t"],
        f"observed_{overlap_name}": observed[overlap_name],
        f"observed_{overlap_name}_global": global_observed[overlap_name],
        "left_baseline_peer_count": expected_left,
        "right_baseline_peer_count": expected_right,
        "left_baseline_spearman_logfc": left_baseline["spearman_logfc"],
        "right_baseline_spearman_logfc": right_baseline["spearman_logfc"],
        "left_baseline_spearman_logfc_global": (
            left_baseline_global["spearman_logfc"]
        ),
        "right_baseline_spearman_logfc_global": (
            right_baseline_global["spearman_logfc"]
        ),
        "left_baseline_spearman_t": left_baseline["spearman_t"],
        "right_baseline_spearman_t": right_baseline["spearman_t"],
        "left_baseline_spearman_t_global": left_baseline_global["spearman_t"],
        "right_baseline_spearman_t_global": (
            right_baseline_global["spearman_t"]
        ),
        f"left_baseline_{overlap_name}": left_baseline[overlap_name],
        f"right_baseline_{overlap_name}": right_baseline[overlap_name],
        f"left_baseline_{overlap_name}_global": (
            left_baseline_global[overlap_name]
        ),
        f"right_baseline_{overlap_name}_global": (
            right_baseline_global[overlap_name]
        ),
        "baseline_pair_mean_spearman_logfc": (
            cross_source_core.mean_available(
                [
                    left_baseline["spearman_logfc"],
                    right_baseline["spearman_logfc"],
                ]
            )
        ),
        "baseline_pair_mean_spearman_t": cross_source_core.mean_available(
            [
                left_baseline["spearman_t"],
                right_baseline["spearman_t"],
            ]
        ),
        "baseline_pair_mean_spearman_logfc_global": (
            cross_source_core.mean_available(
                [
                    left_baseline_global["spearman_logfc"],
                    right_baseline_global["spearman_logfc"],
                ]
            )
        ),
        "baseline_pair_mean_spearman_t_global": (
            cross_source_core.mean_available(
                [
                    left_baseline_global["spearman_t"],
                    right_baseline_global["spearman_t"],
                ]
            )
        ),
        f"baseline_pair_mean_{overlap_name}": (
            cross_source_core.mean_available(
                [
                    left_baseline[overlap_name],
                    right_baseline[overlap_name],
                ]
            )
        ),
        f"baseline_pair_mean_{overlap_name}_global": (
            cross_source_core.mean_available(
                [
                    left_baseline_global[overlap_name],
                    right_baseline_global[overlap_name],
                ]
            )
        ),
        "left_peer_total_count": left_selected.total_count,
        "right_peer_total_count": right_selected.total_count,
        "left_peer_scored_count": left_selected.scored_count,
        "right_peer_scored_count": right_selected.scored_count,
        "peer_sampling_seed": sampling_seed,
        "max_baseline_peers": max_peers,
        "pb_observed_spearman_logfc": observed["spearman_logfc"],
    }
    record.update(
        score_signature_baselines(
            observed=observed["spearman_logfc"],
            left_sample=left_logfc,
            right_sample=right_logfc,
            left_centroid=left_centroid_logfc,
            right_centroid=right_centroid_logfc,
            left_peer_matrix=left_peers,
            right_peer_matrix=right_peers,
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
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _signature_z_score_base_record(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    max_peers: Optional[int],
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
    expected_left = left_source.baseline_peer_count(
        left_obs_id,
        **left_lookup,
    )
    expected_right = right_source.baseline_peer_count(
        right_obs_id,
        **right_lookup,
    )
    if left_selected.total_count != expected_left:
        raise AssertionError(
            f"Left peer set has {left_selected.total_count} rows; "
            f"centroid used {expected_left}"
        )
    if right_selected.total_count != expected_right:
        raise AssertionError(
            f"Right peer set has {right_selected.total_count} rows; "
            f"centroid used {expected_right}"
        )
    record = {
        **row.to_dict(),
        "n_common_genes": int(shared_genes.size),
        "left_baseline_peer_count": expected_left,
        "right_baseline_peer_count": expected_right,
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
        "left_selected": left_selected,
        "right_selected": right_selected,
    }
    return record, state


def _z_score_signature_columns(
    *,
    state: Mapping[str, Any],
    z_score_catalog,
    scale_variant: str,
    prepared_strata: dict[tuple[Any, ...], SignatureStratum],
) -> dict[str, Any]:
    left_source = state["left_source"]
    right_source = state["right_source"]
    left_positions = state["left_positions"]
    right_positions = state["right_positions"]
    left_selected = state["left_selected"]
    right_selected = state["right_selected"]
    left_full = state["left_logfc_full"]
    right_full = state["right_logfc_full"]
    pubchem_cid = str(state["pubchem_cid"])

    left_sample = z_score_catalog.standardize_aligned_vector(
        left_source,
        np.asarray(left_full, dtype=np.float64)[left_positions],
        left_positions,
        scale_variant=scale_variant,
    )
    right_sample = z_score_catalog.standardize_aligned_vector(
        right_source,
        np.asarray(right_full, dtype=np.float64)[right_positions],
        right_positions,
        scale_variant=scale_variant,
    )
    observed = cross_source_core.signed_spearman(left_sample, right_sample)

    def prepared_stratum(source, selected, positions) -> SignatureStratum:
        raw_stratum = selected.stratum
        cache_key = (
            id(raw_stratum),
            str(scale_variant),
            np.asarray(positions, dtype=np.int64).tobytes(),
        )
        prepared = prepared_strata.get(cache_key)
        if prepared is None:
            selected_values = np.asarray(
                raw_stratum.values[:, positions],
                dtype=np.float64,
            )
            standardized = z_score_catalog.standardize_aligned_matrix(
                source,
                selected_values,
                np.asarray(positions, dtype=np.int64),
                scale_variant=scale_variant,
            )
            prepared = SignatureStratum(
                raw_stratum.compounds,
                standardized,
                gene_keys=np.asarray(raw_stratum.gene_keys)[positions],
                max_prepared_cache_bytes=0,
            )
            prepared_strata[cache_key] = prepared
        return prepared

    left_prepared = prepared_stratum(
        left_source,
        left_selected,
        left_positions,
    )
    right_prepared = prepared_stratum(
        right_source,
        right_selected,
        right_positions,
    )
    left_centroid = left_prepared.different_compound_centroid(
        pubchem_cid,
        require_all_finite=True,
    )
    right_centroid = right_prepared.different_compound_centroid(
        pubchem_cid,
        require_all_finite=True,
    )
    left_peers = left_prepared.values[left_selected.row_indices]
    right_peers = right_prepared.values[right_selected.row_indices]
    prefix = f"{scale_variant}__w4"
    values: dict[str, Any] = {
        f"{scale_variant}__n_common_genes": int(len(left_positions)),
        f"{scale_variant}__n_w4_common_genes": int(
            np.sum(np.isfinite(left_sample) & np.isfinite(right_sample))
        ),
        f"{scale_variant}__left_population_row_count": (
            z_score_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        f"{scale_variant}__right_population_row_count": (
            z_score_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).population_row_count
        ),
        f"{scale_variant}__left_population_valid_genes": (
            z_score_catalog.get_stats(
                left_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        f"{scale_variant}__right_population_valid_genes": (
            z_score_catalog.get_stats(
                right_source,
                scale_variant=scale_variant,
            ).n_valid_genes
        ),
        f"{scale_variant}__left_peer_total_count": left_selected.total_count,
        f"{scale_variant}__right_peer_total_count": right_selected.total_count,
        f"{scale_variant}__left_peer_scored_count": left_selected.scored_count,
        f"{scale_variant}__right_peer_scored_count": right_selected.scored_count,
        f"{scale_variant}__w4_observed_spearman_logfc": observed,
    }
    values.update(
        score_signature_baselines(
            observed=observed,
            left_sample=left_sample,
            right_sample=right_sample,
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peer_matrix=left_peers,
            right_peer_matrix=right_peers,
            prefix=prefix,
        )
    )
    return values


def assert_aligned(logfc_stratum, effect_stratum, source, dose_key: str, time_key: str) -> None:
    """Pin the assumption that lets us reuse logFC peer indices on the t stratum.

    Both strata are selected by (dose_key, time_key) and cached per layer, so their row
    order is the same today. If that ever stopped holding, reusing row_indices would
    silently pair each peer's identity with another peer's moderated t: every column
    would still be finite and plausible, and the error would be invisible downstream.
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


MISLEADING_METRIC_NAME = "spearman_logfc"
EFFECT_METRIC_NAME = "spearman_t"


def relabel_effect_layer(record: dict[str, Any]) -> dict[str, Any]:
    """Rename the peer columns so they name the layer they actually measure.

    score_signature_baselines is layer-agnostic in its maths but not in its naming: it
    hardcodes "spearman_logfc" and only varies the prefix. Called on the t layer it
    therefore emits pbt_source_peer_spearman_logfc_pair and friends -- moderated-t values
    under a column name that says logFC. Only the pbt_ prefix distinguishes them from the
    genuine logFC family in the same row, which is far too subtle to survive being read by
    someone else, or by us in six months.

    Renaming here rather than upstream keeps score_signature_baselines untouched, since it
    is shared with the running capfix jobs. Guarded against collisions because the logFC
    family (prefix "pb") lives in this same record.
    """
    renamed: dict[str, Any] = {}
    for key, value in record.items():
        if key.startswith(f"{PEER_PREFIX}_") and MISLEADING_METRIC_NAME in key:
            new_key = key.replace(MISLEADING_METRIC_NAME, EFFECT_METRIC_NAME)
            if new_key in record:
                raise AssertionError(
                    f"renaming {key!r} to {new_key!r} would overwrite an existing column"
                )
            renamed[new_key] = value
        else:
            renamed[key] = value
    return renamed


def _moderated_t_signature_columns(
    row: pd.Series,
    *,
    state: Mapping[str, Any],
    observed_reference: Optional[float],
) -> dict[str, Any]:
    """Add t peer baselines using the raw run's gene positions and sampled peers."""
    left_source = state["left_source"]
    right_source = state["right_source"]
    left_positions = state["left_positions"]
    right_positions = state["right_positions"]
    left_lookup = _lookup(row, "left")
    right_lookup = _lookup(row, "right")
    left_obs_id = str(row["left_obs_id"])
    right_obs_id = str(row["right_obs_id"])

    # get_vector caches on (layer, row_pos), and the raw record already pulled these, so
    # this is a cache hit rather than re-reading the layer.
    left_effect = left_source.get_vector(left_obs_id, EFFECT_LAYER, **left_lookup)[
        left_positions
    ]
    right_effect = right_source.get_vector(right_obs_id, EFFECT_LAYER, **right_lookup)[
        right_positions
    ]
    # Same helper the driver uses, so the centroid convention (different-compound,
    # all-finite) is the published one rather than a re-derivation.
    left_centroid = _centroid(
        left_source, left_obs_id, left_lookup, EFFECT_LAYER, left_positions
    )
    right_centroid = _centroid(
        right_source, right_obs_id, right_lookup, EFFECT_LAYER, right_positions
    )
    left_peers = effect_peer_matrix(
        left_source, state["left_selected"], left_lookup, left_positions
    )
    right_peers = effect_peer_matrix(
        right_source, state["right_selected"], right_lookup, right_positions
    )

    # Recomputed from the same arrays handed to the peer scoring below, so the observed
    # value and its comparators cannot drift apart. Cross-checked against the driver's
    # own observed_spearman_t, which comes from score_signature_pair.
    observed = cross_source_core.signed_spearman(left_effect, right_effect)
    driver_observed = observed_reference
    if driver_observed is not None and np.isfinite(driver_observed) and np.isfinite(observed):
        if not np.isclose(observed, driver_observed, rtol=0, atol=1e-12):
            raise AssertionError(
                f"observed spearman_t {observed!r} disagrees with the driver's "
                f"{driver_observed!r}; the t vectors here are not the driver's"
            )

    record = {OBSERVED_KEY: observed}
    record.update(
        score_signature_baselines(
            observed=observed,
            left_sample=left_effect,
            right_sample=right_effect,
            left_centroid=left_centroid,
            right_centroid=right_centroid,
            left_peer_matrix=left_peers,
            right_peer_matrix=right_peers,
            prefix=PEER_PREFIX,
        )
    )
    return relabel_effect_layer(record)


def score_signature_row(
    row: pd.Series,
    *,
    catalog: cross_source_core.LineSourceCatalog,
    z_score_catalog,
    max_peers: Optional[int],
    sampling_seed: int,
    include_raw_metrics: bool = True,
    z_score_scale_variants: tuple[str, ...] = POPULATION_SCALE_VARIANTS,
    prepared_strata: Optional[
        dict[tuple[Any, ...], SignatureStratum]
    ] = None,
    include_moderated_t: bool = False,
) -> dict[str, Any]:
    if prepared_strata is None:
        prepared_strata = {}
    if include_raw_metrics or include_moderated_t:
        record, state = _raw_signature_record(
            row,
            catalog=catalog,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    else:
        record, state = _signature_z_score_base_record(
            row,
            catalog=catalog,
            max_peers=max_peers,
            sampling_seed=sampling_seed,
        )
    if include_moderated_t:
        record.update(
            _moderated_t_signature_columns(
                row,
                state=state,
                observed_reference=record.get("observed_spearman_t"),
            )
        )
    for scale_variant in z_score_scale_variants:
        record.update(
            _z_score_signature_columns(
                state=state,
                z_score_catalog=z_score_catalog,
                scale_variant=scale_variant,
                prepared_strata=prepared_strata,
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
    prepared_strata: dict[tuple[Any, ...], SignatureStratum] = {}
    try:
        z_score_catalog = (
            make_worker_z_score_catalog(config, catalog)
            if settings["z_score_scale_variants"]
            else None
        )
        for _, row in frame.iterrows():
            try:
                records.append(
                    score_signature_row(
                        row,
                        catalog=catalog,
                        z_score_catalog=z_score_catalog,
                        max_peers=int(settings["max_baseline_peers"]),
                        sampling_seed=int(settings["peer_sampling_seed"]),
                        include_raw_metrics=bool(
                            settings["include_raw_metrics"]
                        ),
                        z_score_scale_variants=tuple(
                            settings["z_score_scale_variants"]
                        ),
                        prepared_strata=prepared_strata,
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
        description=(
            "Parallel, resumable matched-pair signature-similarity scoring."
        ),
        epilog=(
            "Without --compute, run raw plus the z-score scopes selected by --z-score-scales. "
            "Moderated-t is opt-in and retains raw signature metrics."
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
            ("logFC", "t")
            if "raw" in selected_computations or "moderated-t" in selected_computations
            else ("logFC",)
        )
        args.require_adj_p_override = False
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
        "top_k": cross_source_core.TOP_K,
        "max_baseline_peers": args.max_baseline_peers,
        "peer_sampling_seed": args.peer_sampling_seed,
        "computations": list(selected_computations),
        "include_raw_metrics": "raw" in selected_computations or include_moderated_t,
        "include_moderated_t": include_moderated_t,
        "z_score_scale_variants": list(z_score_scale_variants),
        "scorer_version": "parallel-signature-v5",
    }
    metrics_path, _ = run_analysis(
        analysis=ANALYSIS,
        scorer_module="scripts.run_overlap_group_rep_signature_similarity",
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
