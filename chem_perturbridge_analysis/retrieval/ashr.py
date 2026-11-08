"""Adaptive-shrinkage (ashr) representation layers, stored beside the DE files.

The source differential expression h5ads on lustre are read-only for this
pipeline, so the shrunken logFC layers live in separate sidecar files and are
attached to the AnnData in memory at load time. Once attached they are ordinary
layers, so ``_available_representations`` picks them up with no other change.

Two scopes are produced, differing only in which rows share a fitted prior:

``ashr_posterior_mean(logFC)_cell_line``
    One prior per (dataset, cell line), pooling every batch.
``ashr_posterior_mean(logFC)_batch``
    One prior per (dataset, cell line, processing_batch).

For a cell line with a single batch the two scopes select the same rows and so
must produce identical values; that equality is a useful check that the
sidecars were built correctly.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional

import anndata as ad
import h5py
import numpy as np

CELL_LINE_LAYER = "ashr_posterior_mean(logFC)_cell_line"
BATCH_LAYER = "ashr_posterior_mean(logFC)_batch"
ASHR_LAYERS = (CELL_LINE_LAYER, BATCH_LAYER)

#: Set to a sidecar root to switch the layers on. Unset means the retrieval
#: pipeline behaves exactly as it did before ashr existed.
SIDECAR_ROOT_ENV = "ASHR_SIDECAR_ROOT"

_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def sidecar_root_from_env() -> Optional[Path]:
    raw = os.environ.get(SIDECAR_ROOT_ENV, "").strip()
    return Path(raw) if raw else None


def sidecar_path(root: Path, dataset_name: str, cell_type: str) -> Path:
    return Path(root) / dataset_name / f"{cell_type}.h5"


def write_sidecar(
    path: Path,
    layers: Mapping[str, np.ndarray],
    obs_names: np.ndarray,
    var_names: np.ndarray,
    provenance: Mapping[str, str],
    aliases: Optional[Mapping[str, str]] = None,
) -> None:
    """Write sidecar layers, optionally aliasing one layer name onto another.

    Single-batch cell lines produce identical values for both scopes, and tahoe
    alone is 48 such lines at ~23k genes. Recording an alias instead of a second
    copy roughly halves both the sidecar size and the per-load read for those.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(tmp, "w") as handle:
        handle.create_dataset("obs_names", data=np.asarray(obs_names, dtype=object), dtype=_STRING_DTYPE)
        handle.create_dataset("var_names", data=np.asarray(var_names, dtype=object), dtype=_STRING_DTYPE)
        group = handle.create_group("layers")
        for name, matrix in layers.items():
            group.create_dataset(
                name,
                data=np.asarray(matrix, dtype=np.float32),
                compression="gzip",
                compression_opts=4,
                chunks=True,
            )
        for key, value in provenance.items():
            handle.attrs[key] = str(value)
        if aliases:
            handle.attrs["alias_names"] = list(aliases.keys())
            handle.attrs["alias_targets"] = list(aliases.values())
    tmp.replace(path)


def _read_names(handle: h5py.File, key: str) -> np.ndarray:
    raw = handle[key][:]
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in raw])


def attach_layers(
    adata: ad.AnnData, dataset_name: str, cell_type: str, root: Path
) -> list[str]:
    """Attach sidecar ashr layers to ``adata`` in place; returns layer names added.

    A missing sidecar attaches nothing; retrieval raises later only if an ASHR
    representation was explicitly requested. A sidecar that disagrees with the
    AnnData is an error, because silently misaligned rows would corrupt every
    downstream score.
    """
    path = sidecar_path(root, dataset_name, cell_type)
    if not path.exists():
        return []

    with h5py.File(path, "r") as handle:
        obs_names = _read_names(handle, "obs_names")
        var_names = _read_names(handle, "var_names")
        n_obs, n_var = adata.shape
        if obs_names.shape[0] != n_obs or var_names.shape[0] != n_var:
            raise ValueError(
                f"{path}: shape {obs_names.shape[0]}x{var_names.shape[0]} does not match "
                f"{dataset_name}:{cell_type} {n_obs}x{n_var}"
            )
        if not np.array_equal(obs_names, adata.obs_names.to_numpy().astype(str)):
            raise ValueError(f"{path}: obs_names differ from {dataset_name}:{cell_type}")
        if not np.array_equal(var_names, adata.var_names.to_numpy().astype(str)):
            raise ValueError(f"{path}: var_names differ from {dataset_name}:{cell_type}")

        added = []
        for name in handle["layers"]:
            adata.layers[name] = handle["layers"][name][:]
            added.append(name)

        alias_names = [str(v) for v in handle.attrs.get("alias_names", [])]
        alias_targets = [str(v) for v in handle.attrs.get("alias_targets", [])]
        for name, target in zip(alias_names, alias_targets):
            if target not in adata.layers:
                raise ValueError(f"{path}: alias {name} points at missing layer {target}")
            adata.layers[name] = adata.layers[target]
            added.append(name)
    return added


def standard_errors_from_moderated_t(
    logfc: np.ndarray, t_stat: np.ndarray, min_abs_t: float = 1e-6
) -> np.ndarray:
    """Moderated standard errors, recovered as ``|logFC / t|``.

    limma stores ``stdev.unscaled`` but not ``sigma``, so the usual
    ``stdev.unscaled * sigma`` route is unavailable and this ratio is the only
    way to the posterior residual scale. Cells whose ``|t|`` falls under
    ``min_abs_t`` are returned as NaN rather than as a huge standard error.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.abs(logfc / t_stat)
    se[np.abs(t_stat) < min_abs_t] = np.nan
    se[~np.isfinite(se)] = np.nan
    se[se <= 0.0] = np.nan
    return se
