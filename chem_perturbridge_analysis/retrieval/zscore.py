"""Per-gene population z-score representations for the retrieval CLI.

Both scales are applied from caches written by
``scripts/precompute_population_zscore.py``, using ``scripts/population_zscore.py``
from the [fork](https://github.com/elmella/Chem-PerturBridge_analysis).

- ``per_gene_population_zscore(logFC)`` — ``dataset_cell_type`` line caches
- ``per_gene_population_zscore_dataset`` — ``dataset_wide`` caches
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

from scripts.population_zscore import (
    DATASET_CELL_TYPE_SCOPE,
    DATASET_SCOPE,
    DATASET_WIDE_CELL_TYPE,
    PopulationGeneStats,
    dataset_stats_cache_path,
    load_population_stats_cache,
    stats_cache_path,
    unique_gene_index,
)

DEFAULT_Z_SCORE_STATS_ROOT = Path("results/population_zscore_stats")

POPULATION_OBS_COLUMNS = ("is_control", "pubchem_cid", "pert_time_h", "pert_dose_uM")


@dataclass(frozen=True)
class PopulationZScoreContext:
    """Population statistics plus a map from var positions to statistic positions."""

    stats: PopulationGeneStats
    position_to_stat: np.ndarray


def z_score_stats_root() -> Path:
    return Path(os.environ.get("Z_SCORE_STATS_ROOT", DEFAULT_Z_SCORE_STATS_ROOT))


def cell_type_population_cache_path(dataset_name: str, cell_type: str) -> Path:
    return stats_cache_path(z_score_stats_root(), dataset_name, cell_type)


def dataset_population_cache_path(dataset_name: str) -> Path:
    return dataset_stats_cache_path(z_score_stats_root(), dataset_name)


def _cache_ready(cache_path: Path) -> bool:
    return cache_path.is_file() and cache_path.with_suffix(".cache.json").is_file()


def has_cell_type_population_cache(dataset_name: str, cell_type: str) -> bool:
    return _cache_ready(cell_type_population_cache_path(dataset_name, cell_type))


def has_dataset_population_cache(dataset_name: str) -> bool:
    return _cache_ready(dataset_population_cache_path(dataset_name))


def has_population_inputs(adata: ad.AnnData, logfc_layer: str) -> bool:
    if logfc_layer not in adata.layers:
        return False
    return all(column in adata.obs.columns for column in POPULATION_OBS_COLUMNS)


def _context_from_cached_stats(
    adata: ad.AnnData, stats: PopulationGeneStats
) -> PopulationZScoreContext:
    gene_keys, gene_positions = unique_gene_index(adata.var, adata.var_names)
    stat_by_key = {str(key): index for index, key in enumerate(stats.gene_keys)}
    position_to_stat = np.full(adata.n_vars, -1, dtype=np.int64)
    for key, position in zip(gene_keys, gene_positions):
        stat_index = stat_by_key.get(str(key))
        if stat_index is not None:
            position_to_stat[int(position)] = int(stat_index)
    return PopulationZScoreContext(stats=stats, position_to_stat=position_to_stat)


def build_population_context(
    adata: ad.AnnData,
    *,
    dataset_name: str,
    cell_type: str,
    layer_name: str = "logFC",
) -> PopulationZScoreContext:
    """Load the fork's cell-line / cell-type cache and map it onto this line's var axis."""
    del layer_name
    cache_path = cell_type_population_cache_path(dataset_name, cell_type)
    stats = load_population_stats_cache(
        cache_path=cache_path,
        dataset_name=dataset_name,
        cell_type=cell_type,
        expected_scope=DATASET_CELL_TYPE_SCOPE,
    )
    return _context_from_cached_stats(adata, stats)


def build_dataset_population_context(
    adata: ad.AnnData,
    *,
    dataset_name: str,
    layer_name: str = "logFC",
) -> PopulationZScoreContext:
    """Load the fork's dataset-wide cache and map it onto this line's var axis."""
    del layer_name
    cache_path = dataset_population_cache_path(dataset_name)
    stats = load_population_stats_cache(
        cache_path=cache_path,
        dataset_name=dataset_name,
        cell_type=DATASET_WIDE_CELL_TYPE,
        expected_scope=DATASET_SCOPE,
    )
    return _context_from_cached_stats(adata, stats)


def population_valid_gene_mask(
    context: PopulationZScoreContext, var_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the statistic positions for ``var_idx`` and which of them are usable."""
    stat_idx = context.position_to_stat[np.asarray(var_idx, dtype=np.int64)]
    mapped = stat_idx >= 0
    valid = np.zeros(stat_idx.shape, dtype=bool)
    valid[mapped] = context.stats.valid_mask[stat_idx[mapped]]
    return stat_idx, valid


def standardize_selected(
    values: np.ndarray,
    context: PopulationZScoreContext,
    stat_idx: np.ndarray,
    gene_mask: np.ndarray,
) -> np.ndarray:
    """Standardize an already selected shared-gene matrix without changing row identity."""
    gene_mask = np.asarray(gene_mask, dtype=bool)
    positions = np.asarray(stat_idx, dtype=np.int64)[gene_mask]
    selected = np.asarray(values, dtype=np.float64)[:, gene_mask]
    if positions.size == 0:
        return np.empty((selected.shape[0], 0), dtype=np.float32)
    if not np.all(context.stats.valid_mask[positions]):
        raise AssertionError("Population z-score retained an invalid population gene")
    with np.errstate(invalid="ignore", divide="ignore"):
        standardized = (
            selected - context.stats.means[positions][None, :]
        ) / context.stats.population_sds[positions][None, :]
    return standardized.astype(np.float32, copy=False)
