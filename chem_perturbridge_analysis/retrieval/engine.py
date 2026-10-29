from __future__ import annotations

import time
from dataclasses import dataclass
from os import PathLike
from typing import Iterable, Optional

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse

from .config import RetrievalSettings
from .data import (
    CellTypeData,
    DatasetStore,
    best_row_for_group,
    load_cell_type_data,
)
from .metrics import (
    compute_signed_adjusted,
    compute_signed_and_z,
    cosine_scores,
    mrrmse_scores,
    pearson_scores,
    rank_center_norm_rows,
    spearman_scores_precomputed,
)
from .zscore import (
    PopulationZScoreContext,
    build_dataset_population_context,
    build_population_context,
    has_cell_type_population_cache,
    has_dataset_population_cache,
    has_population_inputs,
    population_valid_gene_mask,
    standardize_selected,
)

DERIVED_SIGNED_REP = "signed_-log10(p)*sign(logFC)"
DERIVED_SIGNED_ADJP_REP = "signed_-log10(adj.p)*sign(logFC)"
DERIVED_Z_REP = "z=sign(logFC)*Phi^-1(1-p/2)"
DERIVED_DIRECTION_REP = "sign(logFC)"
DERIVED_POPULATION_ZSCORE_REP = "per_gene_population_zscore(logFC)"
DERIVED_POPULATION_ZSCORE_DATASET_REP = "per_gene_population_zscore_dataset"

ALL_METRICS = ("cosine", "pearson", "spearman", "mrrmse")
SIMILARITY_METRICS = {"cosine", "pearson", "spearman"}
DISTANCE_METRICS = {"mrrmse"}


@dataclass
class PairContext:
    db_cell_type: str
    db_data: CellTypeData
    representations: set[str]
    query_var_idx: np.ndarray
    db_var_idx: np.ndarray
    n_genes: int


def _dense_layer_subset(adata: ad.AnnData, layer: str, var_idx: np.ndarray) -> np.ndarray:
    matrix = adata.layers[layer]
    if issparse(matrix):
        dense = matrix[:, var_idx].toarray()
    else:
        dense = np.asarray(matrix[:, var_idx])
    # NumPy 2.x disallows forcing no-copy in cases where a cast is required.
    return np.asarray(dense, dtype=np.float32)


def _resolve_adj_pvalue_layer(
    query_adata: ad.AnnData, db_adata: ad.AnnData, settings: RetrievalSettings
) -> Optional[str]:
    """First preferred adjusted p-value layer present on both sides, else None."""
    for layer in settings.adj_pvalue_layers:
        if layer in query_adata.layers and layer in db_adata.layers:
            return layer
    return None


def _available_representations(
    query_adata: ad.AnnData,
    db_adata: ad.AnnData,
    settings: RetrievalSettings,
    query_dataset: str = "",
    db_dataset: str = "",
    query_cell_type: str = "",
    db_cell_type: str = "",
) -> list[str]:
    common_layers = sorted(
        (set(query_adata.layers.keys()) & set(db_adata.layers.keys()))
        - set(settings.excluded_layers)
    )
    reps = list(common_layers)
    has_derived = all(
        [
            settings.logfc_layer in query_adata.layers,
            settings.pvalue_layer in query_adata.layers,
            settings.logfc_layer in db_adata.layers,
            settings.pvalue_layer in db_adata.layers,
        ]
    )
    if settings.logfc_layer in query_adata.layers and settings.logfc_layer in db_adata.layers:
        reps.append(DERIVED_DIRECTION_REP)
    if has_derived:
        reps.append(DERIVED_SIGNED_REP)
        reps.append(DERIVED_Z_REP)
    has_logfc = settings.logfc_layer in query_adata.layers and settings.logfc_layer in db_adata.layers
    if has_logfc and _resolve_adj_pvalue_layer(query_adata, db_adata, settings) is not None:
        reps.append(DERIVED_SIGNED_ADJP_REP)
    if has_population_inputs(query_adata, settings.logfc_layer) and has_population_inputs(
        db_adata, settings.logfc_layer
    ):
        if (
            query_dataset
            and db_dataset
            and query_cell_type
            and db_cell_type
            and has_cell_type_population_cache(query_dataset, query_cell_type)
            and has_cell_type_population_cache(db_dataset, db_cell_type)
        ):
            reps.append(DERIVED_POPULATION_ZSCORE_REP)
        if (
            query_dataset
            and db_dataset
            and has_dataset_population_cache(query_dataset)
            and has_dataset_population_cache(db_dataset)
        ):
            reps.append(DERIVED_POPULATION_ZSCORE_DATASET_REP)
    return reps


_POPULATION_CONTEXT_CACHE: dict[tuple[str, str], PopulationZScoreContext] = {}
_DATASET_POPULATION_CONTEXT_CACHE: dict[tuple[str, str], PopulationZScoreContext] = {}


def _population_context(
    cell_data: CellTypeData, settings: RetrievalSettings
) -> PopulationZScoreContext:
    key = (cell_data.dataset_name, cell_data.cell_type)
    cached = _POPULATION_CONTEXT_CACHE.get(key)
    if cached is None:
        cached = build_population_context(
            cell_data.adata,
            dataset_name=cell_data.dataset_name,
            cell_type=cell_data.cell_type,
            layer_name=settings.logfc_layer,
        )
        _POPULATION_CONTEXT_CACHE[key] = cached
    return cached


def _dataset_population_context(
    cell_data: CellTypeData, settings: RetrievalSettings
) -> PopulationZScoreContext:
    key = (cell_data.dataset_name, cell_data.cell_type)
    cached = _DATASET_POPULATION_CONTEXT_CACHE.get(key)
    if cached is None:
        cached = build_dataset_population_context(
            cell_data.adata,
            dataset_name=cell_data.dataset_name,
            layer_name=settings.logfc_layer,
        )
        _DATASET_POPULATION_CONTEXT_CACHE[key] = cached
    return cached


def _materialize_representation(
    query_data: CellTypeData,
    context: PairContext,
    representation: str,
    settings: RetrievalSettings,
) -> tuple[np.ndarray, np.ndarray]:
    query_adata = query_data.adata
    db_adata = context.db_data.adata
    q_idx = context.query_var_idx
    d_idx = context.db_var_idx

    if representation in query_adata.layers and representation in db_adata.layers:
        x_query = _dense_layer_subset(query_adata, representation, q_idx)
        x_db = _dense_layer_subset(db_adata, representation, d_idx)
        return x_query, x_db

    if representation == DERIVED_DIRECTION_REP:
        q_logfc = _dense_layer_subset(query_adata, settings.logfc_layer, q_idx)
        d_logfc = _dense_layer_subset(db_adata, settings.logfc_layer, d_idx)
        return np.sign(q_logfc).astype(np.float32, copy=False), np.sign(d_logfc).astype(
            np.float32, copy=False
        )

    if representation in (
        DERIVED_POPULATION_ZSCORE_REP,
        DERIVED_POPULATION_ZSCORE_DATASET_REP,
    ):
        if representation == DERIVED_POPULATION_ZSCORE_DATASET_REP:
            query_ctx = _dataset_population_context(query_data, settings)
            db_ctx = _dataset_population_context(context.db_data, settings)
        else:
            query_ctx = _population_context(query_data, settings)
            db_ctx = _population_context(context.db_data, settings)
        q_stat_idx, q_valid = population_valid_gene_mask(query_ctx, q_idx)
        d_stat_idx, d_valid = population_valid_gene_mask(db_ctx, d_idx)

        q_logfc = _dense_layer_subset(query_adata, settings.logfc_layer, q_idx)
        d_logfc = _dense_layer_subset(db_adata, settings.logfc_layer, d_idx)
        # Same gene eligibility as the overlap notebooks: finite on both sides, then
        # restricted to genes whose population transform is defined on both sides.
        finite_gene_mask = np.isfinite(q_logfc).all(axis=0) & np.isfinite(d_logfc).all(axis=0)
        gene_mask = finite_gene_mask & q_valid & d_valid
        if int(gene_mask.sum()) < 2:
            raise ValueError(
                "Population z-score left fewer than 2 usable genes for "
                f"{query_data.dataset_name}:{query_data.cell_type} -> "
                f"{context.db_data.dataset_name}:{context.db_cell_type}"
            )
        return (
            standardize_selected(q_logfc, query_ctx, q_stat_idx, gene_mask),
            standardize_selected(d_logfc, db_ctx, d_stat_idx, gene_mask),
        )

    if representation == DERIVED_SIGNED_ADJP_REP:
        adj_layer = _resolve_adj_pvalue_layer(query_adata, db_adata, settings)
        if adj_layer is None:
            raise ValueError(
                f"None of {settings.adj_pvalue_layers} are present in both "
                f"{query_data.dataset_name} and {context.db_data.dataset_name}."
            )
        q_logfc = _dense_layer_subset(query_adata, settings.logfc_layer, q_idx)
        q_adj_p = _dense_layer_subset(query_adata, adj_layer, q_idx)
        d_logfc = _dense_layer_subset(db_adata, settings.logfc_layer, d_idx)
        d_adj_p = _dense_layer_subset(db_adata, adj_layer, d_idx)
        return (
            compute_signed_adjusted(q_logfc, q_adj_p, p_floor=settings.p_floor),
            compute_signed_adjusted(d_logfc, d_adj_p, p_floor=settings.p_floor),
        )

    if representation in (DERIVED_SIGNED_REP, DERIVED_Z_REP):
        q_logfc = _dense_layer_subset(query_adata, settings.logfc_layer, q_idx)
        q_pvals = _dense_layer_subset(query_adata, settings.pvalue_layer, q_idx)
        d_logfc = _dense_layer_subset(db_adata, settings.logfc_layer, d_idx)
        d_pvals = _dense_layer_subset(db_adata, settings.pvalue_layer, d_idx)

        signed_query, z_query = compute_signed_and_z(
            q_logfc, q_pvals, p_floor=settings.p_floor, p_clip_high=settings.p_clip_high
        )
        signed_db, z_db = compute_signed_and_z(
            d_logfc, d_pvals, p_floor=settings.p_floor, p_clip_high=settings.p_clip_high
        )
        if representation == DERIVED_SIGNED_REP:
            return signed_query, signed_db
        return z_query, z_db

    raise ValueError(f"Unknown representation: {representation}")


def _score_candidates_for_query(
    x: np.ndarray,
    y: np.ndarray,
    selected_metrics: tuple[str, ...],
    x_ranked: Optional[np.ndarray] = None,
    x_rank_norm: Optional[float] = None,
    y_ranked: Optional[np.ndarray] = None,
    y_rank_norm: Optional[np.ndarray] = None,
) -> dict[str, np.ndarray]:
    scores: dict[str, np.ndarray] = {}
    if "cosine" in selected_metrics:
        scores["cosine"] = cosine_scores(x, y)
    if "pearson" in selected_metrics:
        scores["pearson"] = pearson_scores(x, y)
    if "spearman" in selected_metrics:
        if x_ranked is None or x_rank_norm is None or y_ranked is None or y_rank_norm is None:
            raise ValueError("Spearman requested but ranked inputs were not provided.")
        scores["spearman"] = spearman_scores_precomputed(
            x_ranked, x_rank_norm, y_ranked, y_rank_norm
        )
    if "mrrmse" in selected_metrics:
        scores["mrrmse"] = mrrmse_scores(x, y)
    return scores


def _source_layers_for_representations(
    query_adata: ad.AnnData,
    db_adata: ad.AnnData,
    representations: set[str],
    settings: RetrievalSettings,
) -> list[str]:
    """Layers that the selected representations read from."""
    layers: set[str] = set()
    for representation in representations:
        if representation in query_adata.layers and representation in db_adata.layers:
            layers.add(representation)
        elif representation in (
            DERIVED_DIRECTION_REP,
            DERIVED_POPULATION_ZSCORE_REP,
            DERIVED_POPULATION_ZSCORE_DATASET_REP,
        ):
            layers.add(settings.logfc_layer)
        elif representation in (DERIVED_SIGNED_REP, DERIVED_Z_REP):
            layers.update({settings.logfc_layer, settings.pvalue_layer})
        elif representation == DERIVED_SIGNED_ADJP_REP:
            layers.add(settings.logfc_layer)
            adj_layer = _resolve_adj_pvalue_layer(query_adata, db_adata, settings)
            if adj_layer is not None:
                layers.add(adj_layer)
    return sorted(layers)


def _finite_gene_mask(
    query_adata: ad.AnnData,
    db_adata: ad.AnnData,
    query_var_idx: np.ndarray,
    db_var_idx: np.ndarray,
    representations: set[str],
    settings: RetrievalSettings,
) -> np.ndarray:
    mask = np.ones(query_var_idx.size, dtype=bool)
    for layer in _source_layers_for_representations(
        query_adata, db_adata, representations, settings
    ):
        mask &= np.isfinite(_dense_layer_subset(query_adata, layer, query_var_idx)).all(axis=0)
        mask &= np.isfinite(_dense_layer_subset(db_adata, layer, db_var_idx)).all(axis=0)
    return mask


def _build_pair_contexts(
    query_data: CellTypeData,
    db_store: DatasetStore,
    db_cell_types: Iterable[str],
    cache: dict[tuple[str, str], Optional[CellTypeData]],
    settings: RetrievalSettings,
) -> list[PairContext]:
    query_var_names = query_data.adata.var_names.values

    contexts: list[PairContext] = []
    for db_cell_type in db_cell_types:
        db_data = load_cell_type_data(db_store, db_cell_type, cache)
        if db_data is None or db_data.adata.n_obs == 0:
            continue

        common_genes = np.intersect1d(
            query_var_names,
            db_data.adata.var_names.values,
            assume_unique=False,
        )
        if common_genes.size == 0:
            continue

        available_reps = set(
            _available_representations(
                query_data.adata,
                db_data.adata,
                settings,
                query_dataset=query_data.dataset_name,
                db_dataset=db_data.dataset_name,
                query_cell_type=query_data.cell_type,
                db_cell_type=db_data.cell_type,
            )
        )
        if settings.include_representations:
            requested_z = settings.include_representations & {
                DERIVED_POPULATION_ZSCORE_REP,
                DERIVED_POPULATION_ZSCORE_DATASET_REP,
            }
            missing_z = requested_z - available_reps
            if missing_z:
                raise FileNotFoundError(
                    "Requested population z-score representation(s) "
                    f"{sorted(missing_z)} but caches are missing for "
                    f"{query_data.dataset_name}:{query_data.cell_type} -> "
                    f"{db_data.dataset_name}:{db_cell_type}. "
                    "Run scripts/slurm/run_precompute_population_zscore.sbatch first."
                )
            pair_reps = available_reps & settings.include_representations
        else:
            pair_reps = available_reps
        if settings.skip_representations:
            pair_reps -= settings.skip_representations
        if not pair_reps:
            continue

        query_var_idx = query_data.adata.var_names.get_indexer(common_genes)
        db_var_idx = db_data.adata.var_names.get_indexer(common_genes)

        # Keep only genes that are finite on both sides in every scored source layer.
        # Without this a single all-NaN gene makes every row non-finite, and the pair
        # yields no scores at all.
        finite_gene_mask = _finite_gene_mask(
            query_data.adata, db_data.adata, query_var_idx, db_var_idx, pair_reps, settings
        )
        n_dropped = int(finite_gene_mask.size - np.count_nonzero(finite_gene_mask))
        if n_dropped:
            query_var_idx = query_var_idx[finite_gene_mask]
            db_var_idx = db_var_idx[finite_gene_mask]
        if query_var_idx.size < 2:
            continue

        contexts.append(
            PairContext(
                db_cell_type=db_cell_type,
                db_data=db_data,
                representations=pair_reps,
                query_var_idx=query_var_idx,
                db_var_idx=db_var_idx,
                n_genes=int(query_var_idx.size),
            )
        )
    return contexts


def evaluate_query_cell_type(
    query_data: CellTypeData,
    db_store: DatasetStore,
    db_cell_types: Iterable[str],
    cache: dict[tuple[str, str], Optional[CellTypeData]],
    settings: RetrievalSettings,
    verbose: bool = False,
    log_prefix: str = "",
) -> list[dict]:
    contexts = _build_pair_contexts(query_data, db_store, db_cell_types, cache, settings)
    if not contexts:
        if verbose:
            print(
                f"{log_prefix} no pair contexts for query_cell_type={query_data.cell_type}; skipping",
                flush=True,
            )
        return []

    # Ground-truth comes from the matching DB cell type; process it first so
    # rank counting can run in one pass over candidate contexts.
    ordered_contexts = sorted(contexts, key=lambda ctx: ctx.db_cell_type != query_data.cell_type)

    selected_metrics = tuple(metric for metric in ALL_METRICS if metric in settings.include_metrics)
    needs_spearman = "spearman" in selected_metrics
    if not selected_metrics:
        if verbose:
            print(f"{log_prefix} no metrics selected; skipping query_cell_type={query_data.cell_type}", flush=True)
        return []

    all_representations = sorted({rep for ctx in contexts for rep in ctx.representations})
    if settings.include_representations:
        all_representations = [
            rep for rep in all_representations if rep in settings.include_representations
        ]
    if settings.skip_representations:
        all_representations = [
            rep for rep in all_representations if rep not in settings.skip_representations
        ]
    if not all_representations:
        if verbose:
            print(
                f"{log_prefix} no representations left after filtering for query_cell_type={query_data.cell_type}",
                flush=True,
            )
        return []

    n_queries = query_data.adata.n_obs
    query_pubchem_cids = query_data.obs["pubchem_cid"].astype("string").fillna("").to_numpy(dtype=str)
    query_cid_valid = query_data.obs["pubchem_cid"].notna().to_numpy(dtype=bool)
    query_times = query_data.obs["pert_time_h"].to_numpy(dtype=np.float64)
    query_doses = query_data.obs["pert_dose_uM"].to_numpy(dtype=np.float64)
    query_index = query_data.adata.obs_names.astype(str).to_numpy()
    valid_query_mask = query_cid_valid & np.isfinite(query_times) & np.isfinite(query_doses)
    valid_query_rows = np.flatnonzero(valid_query_mask)

    # GT row index depends only on (query cell type, matching DB cell type, time, dose, CID),
    # so precompute it once instead of recomputing per representation/context.
    gt_rows = np.full(n_queries, -1, dtype=np.int32)
    gt_context = next((ctx for ctx in ordered_contexts if ctx.db_cell_type == query_data.cell_type), None)
    if gt_context is not None and valid_query_rows.size:
        gt_groups = gt_context.db_data.pubchem_cid_groups
        for q_idx in valid_query_rows:
            group = gt_groups.get(query_pubchem_cids[q_idx])
            if group is None:
                continue
            gt_rows[q_idx] = best_row_for_group(
                group,
                time_h=float(query_times[q_idx]),
                dose_um=float(query_doses[q_idx]),
            )

    out_rows: list[dict] = []
    for rep_idx, representation in enumerate(all_representations, start=1):
        rep_start_rows = len(out_rows)
        rep_t0 = time.perf_counter()
        if verbose:
            print(
                f"{log_prefix} representation {rep_idx}/{len(all_representations)} "
                f"start rep={representation}",
                flush=True,
            )

        candidate_counts = {
            metric: np.zeros(n_queries, dtype=np.int64) for metric in selected_metrics
        }
        better_counts = {
            metric: np.zeros(n_queries, dtype=np.int64) for metric in selected_metrics
        }
        gt_scores = {
            metric: np.full(n_queries, np.nan, dtype=np.float64) for metric in selected_metrics
        }
        gt_genes = np.full(n_queries, np.nan, dtype=np.float64)

        for context in ordered_contexts:
            if representation not in context.representations:
                continue

            is_gt_context = context.db_cell_type == query_data.cell_type
            x_query, x_db = _materialize_representation(query_data, context, representation, settings)
            q_finite = np.isfinite(x_query).all(axis=1)
            d_finite = np.isfinite(x_db).all(axis=1)

            # All finite DB rows are background candidates (all dose/time variants).
            valid_db_rows = np.flatnonzero(d_finite)
            if valid_db_rows.size == 0:
                continue

            context_query_rows = valid_query_rows[q_finite[valid_query_rows]]
            if context_query_rows.size == 0:
                continue

            x_query64 = np.asarray(x_query, dtype=np.float64)
            y_all = np.asarray(x_db[valid_db_rows], dtype=np.float64)

            if is_gt_context:
                row_to_valid_pos = np.full(x_db.shape[0], -1, dtype=np.int32)
                row_to_valid_pos[valid_db_rows] = np.arange(valid_db_rows.size, dtype=np.int32)
            else:
                row_to_valid_pos = None

            if needs_spearman:
                q_ranked, q_rank_norm, q_rank_valid = rank_center_norm_rows(x_query)
                d_ranked, d_rank_norm, _ = rank_center_norm_rows(x_db)
                y_all_ranked = np.asarray(d_ranked[valid_db_rows], dtype=np.float64)
                y_all_rank_norm = np.asarray(d_rank_norm[valid_db_rows], dtype=np.float64)
            else:
                q_ranked = None
                q_rank_norm = None
                q_rank_valid = None
                d_ranked = None
                d_rank_norm = None
                y_all_ranked = None
                y_all_rank_norm = None

            for q_idx in context_query_rows:
                x = x_query64[q_idx]

                # Score query against ALL valid DB rows in this cell-type context.
                metric_scores = _score_candidates_for_query(
                    x=x,
                    y=y_all,
                    selected_metrics=selected_metrics,
                    x_ranked=q_ranked[q_idx] if needs_spearman else None,
                    x_rank_norm=(
                        float(q_rank_norm[q_idx]) if (needs_spearman and q_rank_valid[q_idx]) else np.nan
                    ),
                    y_ranked=y_all_ranked if needs_spearman else None,
                    y_rank_norm=y_all_rank_norm if needs_spearman else None,
                )

                # Ground-truth: same cell type + same pubchem_cid, closest time/dose.
                if is_gt_context and row_to_valid_pos is not None:
                    gt_row = int(gt_rows[q_idx])
                    if gt_row >= 0 and gt_row < d_finite.size and d_finite[gt_row]:
                        gt_pos = int(row_to_valid_pos[gt_row])
                        if gt_pos >= 0:
                            gt_genes[q_idx] = context.n_genes
                            for metric_name, values in metric_scores.items():
                                value = float(values[gt_pos])
                                if np.isfinite(value):
                                    gt_scores[metric_name][q_idx] = value

                for metric_name, values in metric_scores.items():
                    finite_mask = np.isfinite(values)
                    finite_count = int(np.count_nonzero(finite_mask))
                    if finite_count == 0:
                        continue

                    candidate_counts[metric_name][q_idx] += finite_count
                    gt_score = float(gt_scores[metric_name][q_idx])
                    if not np.isfinite(gt_score):
                        continue

                    compare_values = values if finite_count == values.size else values[finite_mask]
                    if metric_name in SIMILARITY_METRICS:
                        better_counts[metric_name][q_idx] += int(np.sum(compare_values > gt_score))
                    elif metric_name in DISTANCE_METRICS:
                        better_counts[metric_name][q_idx] += int(np.sum(compare_values < gt_score))
                    else:
                        raise ValueError(f"Unexpected metric: {metric_name}")

        for metric_name in selected_metrics:
            metric_gt_scores = gt_scores[metric_name]
            metric_candidates = candidate_counts[metric_name]
            metric_better = better_counts[metric_name]
            output_query_rows = np.flatnonzero(np.isfinite(metric_gt_scores) & (metric_candidates > 0))
            for q_idx in output_query_rows:
                n_candidates = int(metric_candidates[q_idx])
                rank = float(1 + int(metric_better[q_idx]))

                if n_candidates <= 1:
                    rank_normalized = 0.0
                    retrieval_score = 1.0
                else:
                    rank_normalized = float((rank - 1.0) / (n_candidates - 1.0))
                    retrieval_score = float(1.0 - rank_normalized)

                out_rows.append(
                    {
                        "query_dataset": query_data.dataset_name,
                        "db_dataset": db_store.dataset_name,
                        "query_cell_type": query_data.cell_type,
                        "query_obs_id": query_index[q_idx],
                        "pubchem_cid": query_pubchem_cids[q_idx],
                        "pert_time_h": float(query_times[q_idx]),
                        "pert_dose_uM": float(query_doses[q_idx]),
                        "representation": representation,
                        "metric": metric_name,
                        "n_candidates": n_candidates,
                        "rank": rank,
                        "rank_over_n": float(rank / n_candidates),
                        "rank_normalized": rank_normalized,
                        "retrieval_score": retrieval_score,
                        "n_genes_gt_pair": float(gt_genes[q_idx])
                    }
                )

        if verbose:
            elapsed = time.perf_counter() - rep_t0
            print(
                f"{log_prefix} representation {rep_idx}/{len(all_representations)} "
                f"done rep={representation} rows_added={len(out_rows) - rep_start_rows} "
                f"elapsed_s={elapsed:.1f}",
                flush=True,
            )

    return out_rows


def evaluate_dataset_pair(
    query_store: DatasetStore,
    db_store: DatasetStore,
    settings: RetrievalSettings,
    cell_type_filter: Optional[set[str]] = None,
    cache: Optional[dict[tuple[str, str], Optional[CellTypeData]]] = None,
    verbose: bool = False,
) -> list[dict]:
    pair_t0 = time.perf_counter()
    shared_cache: dict[tuple[str, str], Optional[CellTypeData]] = cache if cache is not None else {}
    query_cell_types = query_store.list_cell_types()
    if cell_type_filter is not None:
        query_cell_types = [ct for ct in query_cell_types if ct in cell_type_filter]
    all_db_cell_types = db_store.list_cell_types()
    if cell_type_filter is not None:
        all_db_cell_types = [ct for ct in all_db_cell_types if ct in cell_type_filter]

    # Query cell types must exist in the DB for ground-truth to be possible.
    eligible_query_cell_types = sorted(set(query_cell_types) & set(all_db_cell_types))
    if not eligible_query_cell_types:
        if verbose:
            print(
                f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                "no overlapping cell types; skipping pair"
            )
        return []

    if verbose:
        print(
            f"[{query_store.dataset_name}->{db_store.dataset_name}] "
            f"start pair: eligible_query_cell_types={len(eligible_query_cell_types)} "
            f"db_cell_types={len(all_db_cell_types)}",
            flush=True,
        )

    detail_rows: list[dict] = []
    for idx, query_cell_type in enumerate(eligible_query_cell_types, start=1):
        cell_t0 = time.perf_counter()
        query_data = load_cell_type_data(query_store, query_cell_type, shared_cache)
        if query_data is None or query_data.adata.n_obs == 0:
            if verbose:
                print(
                    f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                    f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type} "
                    "query data missing or empty; skipping",
                    flush=True,
                )
            continue

        # Ensure the matching db cell type exists and has overlapping pubchem_cids.
        matching_db_data = load_cell_type_data(db_store, query_cell_type, shared_cache)
        if matching_db_data is None or matching_db_data.adata.n_obs == 0:
            if verbose:
                print(
                    f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                    f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type} "
                    "matching db data missing or empty; skipping",
                    flush=True,
                )
            continue
        shared_pubchem_cids = set(query_data.pubchem_cid_groups) & set(matching_db_data.pubchem_cid_groups)
        if not shared_pubchem_cids:
            if verbose:
                print(
                    f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                    f"cell_type={query_cell_type} has no overlapping pubchem_cids; skipping"
                )
            continue

        # All DB cell types serve as background candidates; GT comes from the matching one.
        rows = evaluate_query_cell_type(
            query_data=query_data,
            db_store=db_store,
            db_cell_types=all_db_cell_types,
            cache=shared_cache,
            settings=settings,
            verbose=verbose,
            log_prefix=(
                f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type}"
            ),
        )
        detail_rows.extend(rows)

        if verbose:
            cell_elapsed = time.perf_counter() - cell_t0
            print(
                f"[{query_store.dataset_name}->{db_store.dataset_name}] "
                f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type} "
                f"rows={len(rows)} cumulative_rows={len(detail_rows)} elapsed_s={cell_elapsed:.1f}",
                flush=True,
            )

    if verbose:
        pair_elapsed = time.perf_counter() - pair_t0
        print(
            f"[{query_store.dataset_name}->{db_store.dataset_name}] "
            f"done pair rows={len(detail_rows)} elapsed_s={pair_elapsed:.1f}",
            flush=True,
        )
    return detail_rows


def summarize_retrieval(detail_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if detail_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    agg = {
        "n_queries": ("rank", "count"),
        "n_candidates_mean": ("n_candidates", "mean"),
        "n_candidates_median": ("n_candidates", "median"),
        "rank_mean": ("rank", "mean"),
        "rank_median": ("rank", "median"),
        "rank_over_n_mean": ("rank_over_n", "mean"),
        "rank_normalized_mean": ("rank_normalized", "mean"),
        "retrieval_score_mean": ("retrieval_score", "mean"),
        "retrieval_score_median": ("retrieval_score", "median"),
    }

    by_cell = (
        detail_df.groupby(
            ["query_dataset", "db_dataset", "query_cell_type", "representation", "metric"],
            dropna=False,
        )
        .agg(**agg)
        .reset_index()
        .sort_values(
            ["query_dataset", "db_dataset", "query_cell_type", "representation", "metric"]
        )
        .reset_index(drop=True)
    )

    overall = (
        detail_df.groupby(["query_dataset", "db_dataset", "representation", "metric"], dropna=False)
        .agg(**agg)
        .reset_index()
        .sort_values(["query_dataset", "db_dataset", "representation", "metric"])
        .reset_index(drop=True)
    )
    return by_cell, overall


def run_cross_dataset_retrieval(
    dataset_paths: dict[str, str | PathLike[str]],
    query_datasets: list[str],
    db_datasets: list[str],
    settings: Optional[RetrievalSettings] = None,
    include_self_dataset: bool = False,
    cell_type_filter: Optional[set[str]] = None,
    cache_cell_types: bool = True,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_t0 = time.perf_counter()
    settings = settings or RetrievalSettings()
    unknown_metrics = sorted(set(settings.include_metrics) - set(ALL_METRICS))
    if unknown_metrics:
        raise ValueError(f"Unknown metrics requested: {unknown_metrics}. Allowed: {list(ALL_METRICS)}")
    if not settings.include_metrics:
        raise ValueError("At least one metric must be selected.")

    stores = {
        name: DatasetStore(dataset_name=name, dataset_path=path, cache_enabled=cache_cell_types)
        for name, path in dataset_paths.items()
    }

    cache: dict[tuple[str, str], Optional[CellTypeData]] = {}
    detail_rows: list[dict] = []

    for query_dataset in query_datasets:
        for db_dataset in db_datasets:
            if not include_self_dataset and query_dataset == db_dataset:
                continue
            if query_dataset not in stores:
                raise KeyError(f"Unknown query dataset: {query_dataset}")
            if db_dataset not in stores:
                raise KeyError(f"Unknown db dataset: {db_dataset}")

            if verbose:
                print(f"[run] start pair {query_dataset}->{db_dataset}", flush=True)

            pair_t0 = time.perf_counter()
            pair_rows = evaluate_dataset_pair(
                query_store=stores[query_dataset],
                db_store=stores[db_dataset],
                settings=settings,
                cell_type_filter=cell_type_filter,
                cache=cache,
                verbose=verbose,
            )
            detail_rows.extend(pair_rows)
            if verbose:
                print(
                    f"[run] done pair {query_dataset}->{db_dataset} rows={len(pair_rows)} "
                    f"cumulative_rows={len(detail_rows)} elapsed_s={time.perf_counter() - pair_t0:.1f}",
                    flush=True,
                )

    detail_df = pd.DataFrame(detail_rows)
    if not detail_df.empty:
        detail_df = detail_df.sort_values(
            [
                "query_dataset",
                "db_dataset",
                "query_cell_type",
                "representation",
                "metric",
                "query_obs_id",
            ]
        ).reset_index(drop=True)

    summary_by_cell, summary_overall = summarize_retrieval(detail_df)
    if verbose:
        print(
            f"[run] done retrieval total_rows={len(detail_df)} "
            f"elapsed_s={time.perf_counter() - run_t0:.1f}",
            flush=True,
        )
    return detail_df, summary_by_cell, summary_overall
