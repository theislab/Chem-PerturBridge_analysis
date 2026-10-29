from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

DEFAULT_DATASET_PATHS: dict[str, Path] = {
    # Mirrors paths used in notebooks/load_data.ipynb
    "sciplex": Path(
        "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/sciplex/deg_data/group_rep/full/qc_false/filter_min_cells_10/results"
    ),
    "tahoe": Path(
        "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/tahoe/deg_data/group_rep/full/qc_false/filter_min_cells_50/results"
    ),
    "l1000_phase1": Path(
        "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/l1000_phase1/deg_data/group_rep/full/qc_false/filter_min_cells_0/results"
    ),
    "l1000_phase2": Path(
        "/lustre/groups/ml01/workspace/olga.novitskaia/data_updated/l1000_phase2/deg_data/group_rep/full/qc_false/filter_min_cells_0/results"
    ),
}

# Optional fallback paths for merged h5ad files.
FALLBACK_DATASET_PATHS: dict[str, Path] = {
    "l1000_phase1": Path(
        "/lustre/groups/ml01/workspace/artur.szalata/data_updated/l1000_phase1/deg_data/group_rep/full/qc_false/filter_min_cells_0/l1000_phase1.h5ad"
    ),
    "l1000_phase2": Path(
        "/lustre/groups/ml01/workspace/artur.szalata/data_updated/l1000_phase2/deg_data/group_rep/full/qc_false/filter_min_cells_0/l1000_phase2.h5ad"
    ),
}


@dataclass(frozen=True)
class RetrievalSettings:
    include_metrics: frozenset[str] = frozenset({"cosine", "pearson", "spearman", "mrrmse"})
    include_representations: frozenset[str] = frozenset()
    skip_representations: frozenset[str] = frozenset()
    excluded_layers: frozenset[str] = frozenset(
        {
            "CI.L",
            "CI.R",
            "stdev.scaled",
            "stdev.unscaled",
            "AveExpr",
            "adj.P.Value.across_all_contrasts",
            "adj.P.Value.within_one_contrast",
        }
    )
    logfc_layer: str = "logFC"
    pvalue_layer: str = "P.Value"
    # Preference order mirrors the overlap retrieval notebook.
    adj_pvalue_layers: tuple[str, ...] = (
        "adj.P.Value.within_one_contrast",
        "adj.P.Value.across_all_contrasts",
    )
    p_floor: float = 1e-300
    p_clip_high: float = 1.0 - 1e-16


def resolve_dataset_paths(overrides: Mapping[str, Path] | None = None) -> dict[str, Path]:
    """Resolve dataset paths from defaults + caller overrides + optional fallbacks."""
    merged = dict(DEFAULT_DATASET_PATHS)
    if overrides:
        merged.update({name: Path(path) for name, path in overrides.items()})

    resolved: dict[str, Path] = {}
    for name, path in merged.items():
        if path.exists():
            resolved[name] = path
            continue
        fallback = FALLBACK_DATASET_PATHS.get(name)
        if fallback is not None and fallback.exists():
            resolved[name] = fallback
            continue
        raise FileNotFoundError(
            f"Dataset path for '{name}' does not exist: {path}"
            + (f" (fallback checked: {fallback})" if fallback else "")
        )
    return resolved
