"""Per-gene population z-score standardization.

Population standardization tests whether heterogeneous source scales explain weak
cross-source agreement, using two explicit populations:

``dataset_cell_type``
    One mean and population standard deviation per gene within a line-level source file.
``dataset``
    One mean and population standard deviation per gene pooled across all supplied
    line-level files for a dataset.

Both use all eligible non-control grouped-condition logFC signatures.  Cross-source match
labels are never used while fitting.  Dataset-wide statistics are merged from the
line-level sufficient statistics, so source matrices are not scanned a second time.

Statistics are cached independently of notebook outputs and shared by the signature, DEG,
and retrieval notebooks.  The historical dataset/cell-type cache layout is retained for
backward compatibility; dataset-wide caches use a separate namespace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Mapping, Optional, Sequence, Union

import anndata as ad
import fcntl
import numpy as np
import pandas as pd


ENGINE_VERSION = 1
DEFAULT_ROW_CHUNK_SIZE = 256
MIN_FINITE_OBSERVATIONS = 2
INVALID_STRING_VALUES = {"", "nan", "none", "<na>"}
DATASET_CELL_TYPE_SCOPE = "dataset_cell_type"
DATASET_SCOPE = "dataset"
DATASET_WIDE_CELL_TYPE = "__all_cell_types__"
DATASET_WIDE_CACHE_NAMESPACE = "dataset_wide"
PER_GENE_DATASET_CELL_TYPE_VARIANT = (
    "per_gene_population_zscore_dataset_cell_type"
)
PER_GENE_DATASET_VARIANT = "per_gene_population_zscore_dataset"
POPULATION_SCALE_VARIANTS = (
    PER_GENE_DATASET_CELL_TYPE_VARIANT,
    PER_GENE_DATASET_VARIANT,
)


def infer_line_cell_type(path: Union[str, Path]) -> str:
    """Infer the notebook cell-type key from a line-level H5AD filename."""
    stem = Path(path).stem
    return stem[:-3] if stem.endswith("_de") else stem


def discover_dataset_population_sources(
    dataset_dir: Union[str, Path],
    *,
    recursive: bool = False,
) -> dict[str, Path]:
    """Discover every line-level H5AD contributing to a dataset-wide population.

    The precompute command and all three notebooks call this same function so the
    meaning and fingerprint of the dataset-wide population cannot silently differ.
    """
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset population directory does not exist: {dataset_dir}"
        )
    paths = sorted(
        dataset_dir.rglob("*.h5ad") if recursive else dataset_dir.glob("*.h5ad")
    )
    if not paths:
        raise FileNotFoundError(f"No .h5ad files found in {dataset_dir}")

    sources: dict[str, Path] = {}
    for path in paths:
        cell_type = infer_line_cell_type(path)
        existing = sources.get(cell_type)
        if existing is not None and existing.resolve() != path.resolve():
            raise ValueError(
                f"Conflicting population sources for cell type {cell_type}: "
                f"{existing} and {path}"
            )
        sources[cell_type] = path
    return dict(sorted(sources.items()))


@dataclass(frozen=True)
class PopulationGeneStats:
    dataset_name: str
    cell_type: str
    gene_keys: np.ndarray
    finite_counts: np.ndarray
    means: np.ndarray
    population_sds: np.ndarray
    valid_mask: np.ndarray
    population_row_count: int
    fingerprint: str
    cache_path: Path
    scope: str = DATASET_CELL_TYPE_SCOPE

    def __post_init__(self) -> None:
        gene_keys = np.asarray(self.gene_keys).astype(str)
        finite_counts = np.asarray(self.finite_counts, dtype=np.int64)
        means = np.asarray(self.means, dtype=np.float64)
        population_sds = np.asarray(self.population_sds, dtype=np.float64)
        valid_mask = np.asarray(self.valid_mask, dtype=bool)
        lengths = {
            len(gene_keys),
            len(finite_counts),
            len(means),
            len(population_sds),
            len(valid_mask),
        }
        if len(lengths) != 1:
            raise ValueError("Population-gene statistic arrays have inconsistent lengths")
        if len(gene_keys) and len(set(gene_keys.tolist())) != len(gene_keys):
            raise ValueError("PopulationGeneStats.gene_keys must be unique")
        if self.scope not in {DATASET_CELL_TYPE_SCOPE, DATASET_SCOPE}:
            raise ValueError(f"Unsupported population scope: {self.scope!r}")
        expected_valid = (
            (finite_counts >= MIN_FINITE_OBSERVATIONS)
            & np.isfinite(means)
            & np.isfinite(population_sds)
            & (population_sds > 0.0)
        )
        if not np.array_equal(valid_mask, expected_valid):
            raise ValueError("PopulationGeneStats.valid_mask is inconsistent with counts/SDs")
        object.__setattr__(self, "gene_keys", gene_keys)
        object.__setattr__(self, "finite_counts", finite_counts)
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "population_sds", population_sds)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "cache_path", Path(self.cache_path))

    @property
    def n_genes(self) -> int:
        return int(len(self.gene_keys))

    @property
    def n_valid_genes(self) -> int:
        return int(self.valid_mask.sum())


@dataclass(frozen=True)
class _PopulationSourceInspection:
    gene_keys: np.ndarray
    gene_positions: np.ndarray
    eligible_mask: np.ndarray
    fingerprint: str
    source_identity: tuple[int, int]


@dataclass(frozen=True)
class PopulationCacheReadiness:
    dataset_name: str
    ready_source_count: int
    total_source_count: int
    dataset_cache_ready: bool
    pending_sources: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return (
            self.ready_source_count == self.total_source_count
            and self.dataset_cache_ready
        )

    def summary(self) -> str:
        dataset_state = "ready" if self.dataset_cache_ready else "pending"
        pending_preview = ", ".join(self.pending_sources[:3])
        if len(self.pending_sources) > 3:
            pending_preview += ", ..."
        pending_suffix = (
            f"; pending lines: {pending_preview}"
            if pending_preview
            else ""
        )
        return (
            f"{self.dataset_name}: line caches "
            f"{self.ready_source_count}/{self.total_source_count}, "
            f"dataset cache {dataset_state}{pending_suffix}"
        )


def _sanitized_strings(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").fillna("").astype(str).str.strip()
    normalized.loc[normalized.str.lower().isin(INVALID_STRING_VALUES)] = ""
    return normalized


def _control_mask(values: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).to_numpy(dtype=bool)
    normalized = values.astype("string").fillna("").astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes"}).to_numpy(dtype=bool)


def eligible_population_mask(obs: pd.DataFrame, cell_type: str) -> np.ndarray:
    """Match the grouped-condition eligibility used by the cross-source notebooks."""
    required = {"is_control", "pubchem_cid", "pert_time_h", "pert_dose_uM"}
    missing = required - set(obs.columns)
    if missing:
        raise KeyError(f"Population source is missing obs columns: {sorted(missing)}")

    pubchem_cid = _sanitized_strings(obs["pubchem_cid"])
    time = pd.to_numeric(obs["pert_time_h"], errors="coerce").to_numpy(dtype=np.float64)
    dose = pd.to_numeric(obs["pert_dose_uM"], errors="coerce").to_numpy(dtype=np.float64)
    eligible = (
        ~_control_mask(obs["is_control"])
        & (pubchem_cid.to_numpy(dtype=str) != "")
        & np.isfinite(time)
        & np.isfinite(dose)
        & (dose > 0.0)
    )
    if "cell_type" in obs.columns:
        observed_cell_type = _sanitized_strings(obs["cell_type"]).to_numpy(dtype=str)
        eligible &= observed_cell_type == str(cell_type)
    return eligible


def unique_gene_index(var: pd.DataFrame, var_names: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    """Return first-occurrence gene keys and positions, matching ``LineSource``."""
    if "symbol" in var.columns:
        gene_keys = _sanitized_strings(var["symbol"]).to_numpy(dtype=str)
    else:
        gene_keys = np.asarray(pd.Index(var_names).astype(str).str.strip(), dtype=str)
    keep = (gene_keys != "") & ~pd.Series(gene_keys).duplicated(keep="first").to_numpy()
    positions = np.flatnonzero(keep).astype(np.int64)
    return gene_keys[positions], positions


def _safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    if value in {"", ".", ".."}:
        raise ValueError(f"Unsafe empty cache component derived from {value!r}")
    return value


def stats_cache_path(cache_root: Path, dataset_name: str, cell_type: str) -> Path:
    """Historical dataset/cell-type cache path (kept backward-compatible)."""
    return (
        Path(cache_root)
        / _safe_component(dataset_name)
        / f"{_safe_component(cell_type)}.npz"
    )


def dataset_stats_cache_path(cache_root: Path, dataset_name: str) -> Path:
    """Return the cache path for a dataset-wide pooled population."""
    return (
        Path(cache_root)
        / DATASET_WIDE_CACHE_NAMESPACE
        / f"{_safe_component(dataset_name)}.npz"
    )


def _metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(".cache.json")


@contextmanager
def _exclusive_cache_lock(
    cache_path: Path,
    *,
    label: Optional[str] = None,
    verbose: bool = False,
) -> Iterator[None]:
    """Serialize first-time fits of one source-context cache.

    Atomic replacement protects readers from partial files, but without a lock two
    notebooks that miss the same cache can both scan a large source before either one
    publishes it.  The persistent, tiny ``.lock`` file is intentional; the kernel lock
    itself is released automatically when the file handle closes.
    """
    lock_path = Path(cache_path).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        started_at = time.monotonic()
        if verbose:
            print(
                f"[z_score_stats] waiting for cache lock: {label or cache_path}",
                flush=True,
            )
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        waited = time.monotonic() - started_at
        if verbose:
            print(
                f"[z_score_stats] acquired cache lock after {waited:.1f}s: "
                f"{label or cache_path}",
                flush=True,
            )
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _source_file_identity(source_path: Path) -> tuple[int, int]:
    stat = Path(source_path).stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _source_fingerprint(
    *,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    shape: tuple[int, int],
    gene_keys: np.ndarray,
    eligible_mask: np.ndarray,
    layer_name: str,
    source_identity: Optional[tuple[int, int]] = None,
) -> str:
    source_size, source_mtime_ns = (
        _source_file_identity(source_path)
        if source_identity is None
        else source_identity
    )
    gene_hash = hashlib.sha256(
        "\0".join(np.asarray(gene_keys).astype(str).tolist()).encode("utf-8")
    ).hexdigest()
    eligible_hash = hashlib.sha256(
        np.packbits(np.asarray(eligible_mask, dtype=np.uint8)).tobytes()
    ).hexdigest()
    payload = {
        "engine_version": ENGINE_VERSION,
        "dataset_name": str(dataset_name),
        "cell_type": str(cell_type),
        "source_path": str(source_path.resolve()),
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "shape": [int(shape[0]), int(shape[1])],
        "layer_name": str(layer_name),
        "gene_hash": gene_hash,
        "eligible_hash": eligible_hash,
        "ddof": 0,
        "minimum_finite_observations": MIN_FINITE_OBSERVATIONS,
        "eligibility_policy": (
            "non_control+valid_pubchem+finite_time+finite_positive_dose+matching_cell_type"
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _inspect_population_source(
    *,
    adata: ad.AnnData,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    layer_name: str,
) -> _PopulationSourceInspection:
    if layer_name not in adata.layers:
        raise KeyError(f"{source_path} has no {layer_name!r} layer")
    gene_keys, gene_positions = unique_gene_index(adata.var, adata.var_names)
    eligible_mask = eligible_population_mask(adata.obs, cell_type)
    source_identity = _source_file_identity(source_path)
    fingerprint = _source_fingerprint(
        source_path=source_path,
        dataset_name=dataset_name,
        cell_type=cell_type,
        shape=adata.shape,
        gene_keys=gene_keys,
        eligible_mask=eligible_mask,
        layer_name=layer_name,
        source_identity=source_identity,
    )
    return _PopulationSourceInspection(
        gene_keys=gene_keys,
        gene_positions=gene_positions,
        eligible_mask=eligible_mask,
        fingerprint=fingerprint,
        source_identity=source_identity,
    )


def _dataset_fingerprint(
    *,
    dataset_name: str,
    source_stats: Sequence[PopulationGeneStats],
) -> str:
    return _dataset_fingerprint_from_source_records(
        dataset_name=dataset_name,
        source_records=[
            (stats.cell_type, stats.fingerprint)
            for stats in source_stats
        ],
    )


def _dataset_fingerprint_from_source_records(
    *,
    dataset_name: str,
    source_records: Sequence[tuple[str, str]],
) -> str:
    source_records = sorted(
        (
            (str(cell_type), str(fingerprint))
            for cell_type, fingerprint in source_records
        ),
        key=lambda item: (item[0], item[1]),
    )
    payload = {
        "engine_version": ENGINE_VERSION,
        "scope": DATASET_SCOPE,
        "dataset_name": str(dataset_name),
        "ddof": 0,
        "minimum_finite_observations": MIN_FINITE_OBSERVATIONS,
        "sources": [
            {
                "cell_type": cell_type,
                "fingerprint": fingerprint,
            }
            for cell_type, fingerprint in source_records
        ],
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _merge_batch_statistics(
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    batch: np.ndarray,
) -> None:
    """In-place Chan/Welford merge for a finite-aware row batch."""
    finite = np.isfinite(batch)
    batch_counts = finite.sum(axis=0, dtype=np.int64)
    if not np.any(batch_counts):
        return
    batch_sums = np.where(finite, batch, 0.0).sum(axis=0, dtype=np.float64)
    batch_means = np.zeros(batch.shape[1], dtype=np.float64)
    present = batch_counts > 0
    batch_means[present] = batch_sums[present] / batch_counts[present]
    centered = np.where(finite, batch - batch_means[None, :], 0.0)
    batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)

    previous_counts = counts.copy()
    combined_counts = previous_counts + batch_counts
    both = (previous_counts > 0) & present
    new_only = (previous_counts == 0) & present

    means[new_only] = batch_means[new_only]
    m2[new_only] = batch_m2[new_only]
    if np.any(both):
        delta = batch_means[both] - means[both]
        means[both] += delta * batch_counts[both] / combined_counts[both]
        m2[both] += (
            batch_m2[both]
            + delta
            * delta
            * previous_counts[both]
            * batch_counts[both]
            / combined_counts[both]
        )
    counts[:] = combined_counts


def _merge_summary_statistics(
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    *,
    incoming_counts: np.ndarray,
    incoming_means: np.ndarray,
    incoming_m2: np.ndarray,
) -> None:
    """In-place finite-aware Chan merge of already summarized gene populations."""
    incoming_counts = np.asarray(incoming_counts, dtype=np.int64)
    incoming_means = np.asarray(incoming_means, dtype=np.float64)
    incoming_m2 = np.asarray(incoming_m2, dtype=np.float64)
    if not (
        counts.shape
        == means.shape
        == m2.shape
        == incoming_counts.shape
        == incoming_means.shape
        == incoming_m2.shape
    ):
        raise ValueError("Summary-statistic arrays must have identical shapes")

    present = incoming_counts > 0
    if not np.any(present):
        return
    previous_counts = counts.copy()
    combined_counts = previous_counts + incoming_counts
    both = (previous_counts > 0) & present
    new_only = (previous_counts == 0) & present

    means[new_only] = incoming_means[new_only]
    m2[new_only] = incoming_m2[new_only]
    if np.any(both):
        delta = incoming_means[both] - means[both]
        means[both] += (
            delta * incoming_counts[both] / combined_counts[both]
        )
        m2[both] += (
            incoming_m2[both]
            + delta
            * delta
            * previous_counts[both]
            * incoming_counts[both]
            / combined_counts[both]
        )
    counts[:] = combined_counts


def _stats_from_accumulators(
    *,
    dataset_name: str,
    cell_type: str,
    gene_keys: np.ndarray,
    counts: np.ndarray,
    means: np.ndarray,
    m2: np.ndarray,
    population_row_count: int,
    fingerprint: str,
    cache_path: Path,
    scope: str,
) -> PopulationGeneStats:
    counts = np.asarray(counts, dtype=np.int64)
    means = np.asarray(means, dtype=np.float64).copy()
    m2 = np.asarray(m2, dtype=np.float64)
    population_sds = np.full(len(gene_keys), np.nan, dtype=np.float64)
    observed = counts > 0
    population_sds[observed] = np.sqrt(
        np.maximum(m2[observed], 0.0) / counts[observed]
    )
    means[~observed] = np.nan
    valid_mask = (
        (counts >= MIN_FINITE_OBSERVATIONS)
        & np.isfinite(means)
        & np.isfinite(population_sds)
        & (population_sds > 0.0)
    )
    return PopulationGeneStats(
        dataset_name=str(dataset_name),
        cell_type=str(cell_type),
        gene_keys=gene_keys,
        finite_counts=counts,
        means=means,
        population_sds=population_sds,
        valid_mask=valid_mask,
        population_row_count=int(population_row_count),
        fingerprint=fingerprint,
        cache_path=cache_path,
        scope=scope,
    )


def _fit_population_stats_from_open_adata(
    *,
    adata: ad.AnnData,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    cache_path: Path,
    inspection: _PopulationSourceInspection,
    layer_name: str = "logFC",
    row_chunk_size: int = DEFAULT_ROW_CHUNK_SIZE,
    verbose: bool = False,
    progress_interval_seconds: float = 60.0,
) -> PopulationGeneStats:
    if row_chunk_size < 1:
        raise ValueError("row_chunk_size must be positive")

    gene_keys = inspection.gene_keys
    gene_positions = inspection.gene_positions
    eligible_mask = inspection.eligible_mask
    n_genes = len(gene_keys)
    counts = np.zeros(n_genes, dtype=np.int64)
    means = np.zeros(n_genes, dtype=np.float64)
    m2 = np.zeros(n_genes, dtype=np.float64)
    started_at = time.monotonic()
    last_report_at = started_at
    if verbose:
        print(
            f"[z_score_stats] fitting {dataset_name}/{cell_type}: "
            f"{adata.n_obs:,} source rows in chunks of {row_chunk_size:,}",
            flush=True,
        )
    for start in range(0, adata.n_obs, row_chunk_size):
        stop = min(start + row_chunk_size, adata.n_obs)
        local_eligible = eligible_mask[start:stop]
        if np.any(local_eligible):
            # Preserve the layer's stored dtype during I/O, then promote only the
            # eligible unique-gene batch used by the stable float64 accumulator.
            raw = np.asarray(adata.layers[layer_name][start:stop, :])
            batch = np.asarray(
                raw[local_eligible][:, gene_positions],
                dtype=np.float64,
            )
            _merge_batch_statistics(counts, means, m2, batch)
        now = time.monotonic()
        if verbose and (
            stop == adata.n_obs
            or now - last_report_at >= float(progress_interval_seconds)
        ):
            elapsed = max(now - started_at, 1e-12)
            rows_per_second = stop / elapsed
            remaining_seconds = (
                (adata.n_obs - stop) / rows_per_second
                if rows_per_second > 0.0
                else float("nan")
            )
            eta = (
                f"{remaining_seconds / 60.0:.1f}m"
                if np.isfinite(remaining_seconds)
                else "unknown"
            )
            print(
                f"[z_score_stats] fitting {dataset_name}/{cell_type}: "
                f"{stop:,}/{adata.n_obs:,} rows "
                f"({100.0 * stop / max(adata.n_obs, 1):.1f}%), "
                f"{rows_per_second:,.1f} rows/s, ETA {eta}",
                flush=True,
            )
            last_report_at = now

    return _stats_from_accumulators(
        dataset_name=str(dataset_name),
        cell_type=str(cell_type),
        gene_keys=gene_keys,
        counts=counts,
        means=means,
        m2=m2,
        population_row_count=int(eligible_mask.sum()),
        fingerprint=inspection.fingerprint,
        cache_path=cache_path,
        scope=DATASET_CELL_TYPE_SCOPE,
    )


def fit_population_stats(
    *,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    cache_path: Path,
    layer_name: str = "logFC",
    row_chunk_size: int = DEFAULT_ROW_CHUNK_SIZE,
    verbose: bool = False,
    progress_interval_seconds: float = 60.0,
) -> PopulationGeneStats:
    """Fit one source with a single backed-H5AD open."""
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    adata = ad.read_h5ad(source_path, backed="r")
    try:
        inspection = _inspect_population_source(
            adata=adata,
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            layer_name=layer_name,
        )
        stats = _fit_population_stats_from_open_adata(
            adata=adata,
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            cache_path=cache_path,
            inspection=inspection,
            layer_name=layer_name,
            row_chunk_size=row_chunk_size,
            verbose=verbose,
            progress_interval_seconds=progress_interval_seconds,
        )
    finally:
        adata.file.close()
    if _source_file_identity(source_path) != inspection.source_identity:
        raise AssertionError(
            "Source changed while population statistics were being fitted"
        )
    return stats


def _cache_metadata(
    stats: PopulationGeneStats,
    *,
    source_path: Optional[Path] = None,
    source_identity: Optional[tuple[int, int]] = None,
    layer_name: Optional[str] = None,
) -> dict[str, object]:
    metadata = {
        "engine_version": ENGINE_VERSION,
        "scope": stats.scope,
        "dataset_name": stats.dataset_name,
        "cell_type": stats.cell_type,
        "population_row_count": stats.population_row_count,
        "n_genes": stats.n_genes,
        "n_valid_genes": stats.n_valid_genes,
        "fingerprint": stats.fingerprint,
    }
    if source_path is not None:
        source_path = Path(source_path)
        source_size, source_mtime_ns = (
            _source_file_identity(source_path)
            if source_identity is None
            else source_identity
        )
        metadata.update(
            {
                "source_path": str(source_path.resolve()),
                "source_size": source_size,
                "source_mtime_ns": source_mtime_ns,
                "layer_name": str(layer_name or "logFC"),
            }
        )
    return metadata


def _write_cache_metadata(
    stats: PopulationGeneStats,
    *,
    source_path: Optional[Path] = None,
    source_identity: Optional[tuple[int, int]] = None,
    layer_name: Optional[str] = None,
) -> None:
    metadata_path = _metadata_path(stats.cache_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_metadata_path = metadata_path.with_name(
        f".{metadata_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        temporary_metadata_path.write_text(
            json.dumps(
                _cache_metadata(
                    stats,
                    source_path=source_path,
                    source_identity=source_identity,
                    layer_name=layer_name,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        os.replace(temporary_metadata_path, metadata_path)
    finally:
        if temporary_metadata_path.exists():
            temporary_metadata_path.unlink()


def _write_cache(
    stats: PopulationGeneStats,
    *,
    source_path: Optional[Path] = None,
    source_identity: Optional[tuple[int, int]] = None,
    layer_name: Optional[str] = None,
) -> None:
    cache_path = stats.cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.tmp-{os.getpid()}-{threading.get_ident()}"
    )
    try:
        with temporary_path.open("wb") as stream:
            np.savez_compressed(
                stream,
                gene_keys=stats.gene_keys.astype(str),
                finite_counts=stats.finite_counts,
                means=stats.means,
                population_sds=stats.population_sds,
                valid_mask=stats.valid_mask,
            )
        os.replace(temporary_path, cache_path)
        _write_cache_metadata(
            stats,
            source_path=source_path,
            source_identity=source_identity,
            layer_name=layer_name,
        )
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _load_cache(
    *,
    cache_path: Path,
    expected_fingerprint: str,
    dataset_name: str,
    cell_type: str,
    expected_scope: str = DATASET_CELL_TYPE_SCOPE,
) -> Optional[PopulationGeneStats]:
    metadata_path = _metadata_path(cache_path)
    try:
        metadata = json.loads(metadata_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    if metadata.get("fingerprint") != expected_fingerprint or not cache_path.exists():
        return None
    if metadata.get("scope", DATASET_CELL_TYPE_SCOPE) != expected_scope:
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as values:
            stats = PopulationGeneStats(
                dataset_name=str(dataset_name),
                cell_type=str(cell_type),
                gene_keys=values["gene_keys"].astype(str),
                finite_counts=values["finite_counts"],
                means=values["means"],
                population_sds=values["population_sds"],
                valid_mask=values["valid_mask"],
                population_row_count=int(metadata["population_row_count"]),
                fingerprint=str(expected_fingerprint),
                cache_path=cache_path,
                scope=expected_scope,
            )
    except (KeyError, OSError, ValueError):
        return None
    return stats


def load_population_stats_cache(
    *,
    cache_path: Path,
    dataset_name: str,
    cell_type: str,
    expected_scope: str,
) -> PopulationGeneStats:
    """Load an existing population-statistics cache without fitting anything.

    This is the read-only entry point for downstream scorers that consume caches
    produced by ``precompute_population_zscore.py``.  Compatibility is still
    checked against the fingerprint and scope recorded beside the NPZ.
    """
    cache_path = Path(cache_path)
    metadata_path = _metadata_path(cache_path)
    try:
        metadata = json.loads(metadata_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Population-statistics metadata not found: {metadata_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Population-statistics metadata is invalid JSON: {metadata_path}"
        ) from exc
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(
            f"Population-statistics metadata has no fingerprint: {metadata_path}"
        )
    stats = _load_cache(
        cache_path=cache_path,
        expected_fingerprint=fingerprint,
        dataset_name=str(dataset_name),
        cell_type=str(cell_type),
        expected_scope=str(expected_scope),
    )
    if stats is None:
        raise ValueError(
            f"Population-statistics cache is missing or incompatible: {cache_path}"
        )
    return stats


def _unchanged_source_cache_metadata(
    *,
    cache_path: Path,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    layer_name: str,
) -> Optional[dict[str, object]]:
    """Return compatible source-cache metadata without opening the H5AD."""
    try:
        metadata = json.loads(_metadata_path(cache_path).read_text())
        source_size, source_mtime_ns = _source_file_identity(source_path)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    expected_inventory = {
        "engine_version": ENGINE_VERSION,
        "scope": DATASET_CELL_TYPE_SCOPE,
        "dataset_name": str(dataset_name),
        "cell_type": str(cell_type),
        "source_path": str(Path(source_path).resolve()),
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "layer_name": str(layer_name),
    }
    if any(metadata.get(key) != value for key, value in expected_inventory.items()):
        return None
    if not cache_path.exists():
        return None
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        return None
    return metadata


def _load_unchanged_source_cache(
    *,
    cache_path: Path,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    layer_name: str,
) -> Optional[PopulationGeneStats]:
    """Fast-path a line cache using immutable source inventory metadata."""
    metadata = _unchanged_source_cache_metadata(
        cache_path=cache_path,
        source_path=source_path,
        dataset_name=dataset_name,
        cell_type=cell_type,
        layer_name=layer_name,
    )
    if metadata is None:
        return None
    return _load_cache(
        cache_path=cache_path,
        expected_fingerprint=str(metadata["fingerprint"]),
        dataset_name=dataset_name,
        cell_type=cell_type,
    )


def load_or_fit_population_stats(
    *,
    source_path: Path,
    dataset_name: str,
    cell_type: str,
    cache_root: Path,
    layer_name: str = "logFC",
    row_chunk_size: int = DEFAULT_ROW_CHUNK_SIZE,
    force: bool = False,
    verbose: bool = True,
) -> PopulationGeneStats:
    """Load a compatible source-context cache or fit and publish it atomically."""
    source_path = Path(source_path)
    cache_path = stats_cache_path(cache_root, dataset_name, cell_type)
    inspect_started_at = time.monotonic()
    if not force:
        cached = _load_unchanged_source_cache(
            cache_path=cache_path,
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            layer_name=layer_name,
        )
        if cached is not None:
            if verbose:
                print(
                    f"[z_score_stats] fast-reloaded {dataset_name}/{cell_type}: "
                    f"{cached.n_valid_genes:,}/{cached.n_genes:,} valid genes",
                    flush=True,
                )
            return cached
    if verbose:
        print(
            f"[z_score_stats] inspecting {dataset_name}/{cell_type}: {source_path}",
            flush=True,
        )
    adata = ad.read_h5ad(source_path, backed="r")
    try:
        inspection = _inspect_population_source(
            adata=adata,
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            layer_name=layer_name,
        )
        if not force:
            cached = _load_cache(
                cache_path=cache_path,
                expected_fingerprint=inspection.fingerprint,
                dataset_name=dataset_name,
                cell_type=cell_type,
            )
            if cached is not None:
                if _source_file_identity(source_path) != inspection.source_identity:
                    raise AssertionError(
                        "Source changed while its population cache was validated"
                    )
                if verbose:
                    print(
                        f"[z_score_stats] reloaded {dataset_name}/{cell_type} after "
                        f"{time.monotonic() - inspect_started_at:.1f}s: "
                        f"{cached.n_valid_genes:,}/{cached.n_genes:,} valid genes"
                    )
                _write_cache_metadata(
                    cached,
                    source_path=source_path,
                    source_identity=inspection.source_identity,
                    layer_name=layer_name,
                )
                return cached

        with _exclusive_cache_lock(
            cache_path,
            label=f"{dataset_name}/{cell_type}",
            verbose=verbose,
        ):
            # Another process may have completed this source while this process
            # waited. Recheck before scanning the already-open H5AD layer.
            if not force:
                cached = _load_cache(
                    cache_path=cache_path,
                    expected_fingerprint=inspection.fingerprint,
                    dataset_name=dataset_name,
                    cell_type=cell_type,
                )
                if cached is not None:
                    if (
                        _source_file_identity(source_path)
                        != inspection.source_identity
                    ):
                        raise AssertionError(
                            "Source changed while waiting for its population cache"
                        )
                    if verbose:
                        print(
                            f"[z_score_stats] reloaded after waiting "
                            f"{dataset_name}/{cell_type}: "
                            f"{cached.n_valid_genes:,}/{cached.n_genes:,} "
                            "valid genes",
                            flush=True,
                        )
                    _write_cache_metadata(
                        cached,
                        source_path=source_path,
                        source_identity=inspection.source_identity,
                        layer_name=layer_name,
                    )
                    return cached

            stats = _fit_population_stats_from_open_adata(
                adata=adata,
                source_path=source_path,
                dataset_name=dataset_name,
                cell_type=cell_type,
                cache_path=cache_path,
                inspection=inspection,
                layer_name=layer_name,
                row_chunk_size=row_chunk_size,
                verbose=verbose,
            )
            if _source_file_identity(source_path) != inspection.source_identity:
                raise AssertionError(
                    "Source changed while population statistics were being fitted"
                )
            _write_cache(
                stats,
                source_path=source_path,
                source_identity=inspection.source_identity,
                layer_name=layer_name,
            )
    finally:
        adata.file.close()
    if stats.fingerprint != inspection.fingerprint:
        # This is defensive: both values derive from the same immutable inspection.
        # Keeping the assertion makes accidental future divergence explicit.
        raise AssertionError(
            "Fitted population statistics fingerprint changed unexpectedly"
        )
    if _source_file_identity(source_path) != inspection.source_identity:
        # Catch a source replacement between cache publication and file close.
        raise AssertionError(
            "Source changed while population statistics were being fitted"
        )
    if verbose:
        print(
            f"[z_score_stats] computed {dataset_name}/{cell_type} in "
            f"{time.monotonic() - inspect_started_at:.1f}s: "
            f"{stats.population_row_count:,} rows, "
            f"{stats.n_valid_genes:,}/{stats.n_genes:,} valid genes",
            flush=True,
        )
    return stats


def _canonical_source_paths(
    source_paths: Union[Mapping[str, Path], Sequence[tuple[str, Path]]],
) -> list[tuple[str, Path]]:
    items = (
        list(source_paths.items())
        if isinstance(source_paths, Mapping)
        else list(source_paths)
    )
    if not items:
        raise ValueError("At least one dataset line source is required")
    normalized = sorted(
        ((str(cell_type), Path(path)) for cell_type, path in items),
        key=lambda item: (item[0], str(item[1].resolve())),
    )
    cell_types = [cell_type for cell_type, _ in normalized]
    if len(set(cell_types)) != len(cell_types):
        raise ValueError("Dataset-wide source cell types must be unique")
    return normalized


def check_dataset_population_cache_readiness(
    *,
    source_paths: Union[Mapping[str, Path], Sequence[tuple[str, Path]]],
    dataset_name: str,
    cache_root: Path,
    layer_name: str = "logFC",
) -> PopulationCacheReadiness:
    """Check both z-score cache scopes using metadata and file inventory only."""
    canonical_sources = _canonical_source_paths(source_paths)
    source_records: list[tuple[str, str]] = []
    pending_sources: list[str] = []
    for cell_type, source_path in canonical_sources:
        cache_path = stats_cache_path(cache_root, dataset_name, cell_type)
        metadata = _unchanged_source_cache_metadata(
            cache_path=cache_path,
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            layer_name=layer_name,
        )
        if metadata is None:
            pending_sources.append(cell_type)
            continue
        source_records.append((cell_type, str(metadata["fingerprint"])))

    dataset_cache_ready = False
    if len(source_records) == len(canonical_sources):
        expected_fingerprint = _dataset_fingerprint_from_source_records(
            dataset_name=dataset_name,
            source_records=source_records,
        )
        dataset_cache_path = dataset_stats_cache_path(cache_root, dataset_name)
        try:
            dataset_metadata = json.loads(
                _metadata_path(dataset_cache_path).read_text()
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            dataset_metadata = {}
        dataset_cache_ready = (
            dataset_cache_path.exists()
            and dataset_metadata.get("engine_version") == ENGINE_VERSION
            and dataset_metadata.get("scope") == DATASET_SCOPE
            and dataset_metadata.get("dataset_name") == str(dataset_name)
            and dataset_metadata.get("cell_type") == DATASET_WIDE_CELL_TYPE
            and dataset_metadata.get("fingerprint") == expected_fingerprint
        )

    return PopulationCacheReadiness(
        dataset_name=str(dataset_name),
        ready_source_count=len(source_records),
        total_source_count=len(canonical_sources),
        dataset_cache_ready=dataset_cache_ready,
        pending_sources=tuple(pending_sources),
    )


def check_population_caches_ready(
    *,
    dataset_sources: Mapping[
        str,
        Union[Mapping[str, Path], Sequence[tuple[str, Path]]],
    ],
    cache_root: Path,
    layer_name: str = "logFC",
) -> list[PopulationCacheReadiness]:
    """Return deterministic readiness records for every requested dataset."""
    if not dataset_sources:
        raise ValueError("At least one dataset is required for z-score readiness")
    return [
        check_dataset_population_cache_readiness(
            source_paths=dataset_sources[dataset_name],
            dataset_name=dataset_name,
            cache_root=cache_root,
            layer_name=layer_name,
        )
        for dataset_name in sorted(dataset_sources)
    ]


def ensure_population_caches_ready(
    *,
    dataset_sources: Mapping[
        str,
        Union[Mapping[str, Path], Sequence[tuple[str, Path]]],
    ],
    cache_root: Path,
    layer_name: str = "logFC",
    wait: bool = False,
    poll_seconds: float = 30.0,
    timeout_seconds: Optional[float] = None,
    verbose: bool = True,
) -> list[PopulationCacheReadiness]:
    """Require completed line and dataset caches, optionally polling until ready."""
    if poll_seconds <= 0.0:
        raise ValueError("poll_seconds must be positive")
    if timeout_seconds is not None and timeout_seconds < 0.0:
        raise ValueError("timeout_seconds cannot be negative")

    started_at = time.monotonic()
    while True:
        readiness = check_population_caches_ready(
            dataset_sources=dataset_sources,
            cache_root=cache_root,
            layer_name=layer_name,
        )
        pending = [record for record in readiness if not record.ready]
        if not pending:
            if verbose:
                print(
                    f"[z_score_precompute] all {len(readiness):,} required dataset "
                    "caches are ready",
                    flush=True,
                )
            return readiness

        details = " | ".join(record.summary() for record in pending)
        if not wait:
            raise RuntimeError(
                "z-score population precompute is incomplete. "
                f"{details}. Finish scripts/precompute_population_zscore.py, "
                "then rerun this z-score setup cell and the cells below it."
            )

        elapsed = time.monotonic() - started_at
        if timeout_seconds is not None and elapsed >= timeout_seconds:
            raise TimeoutError(
                "Timed out waiting for z-score population precompute. "
                f"{details}"
            )
        if verbose:
            print(
                f"[z_score_precompute] waiting {elapsed / 60.0:.1f}m: {details}",
                flush=True,
            )
        sleep_seconds = poll_seconds
        if timeout_seconds is not None:
            sleep_seconds = min(
                sleep_seconds,
                max(timeout_seconds - elapsed, 0.0),
            )
        time.sleep(sleep_seconds)


def fit_dataset_population_stats(
    *,
    dataset_name: str,
    source_stats: Sequence[PopulationGeneStats],
    cache_path: Path,
    fingerprint: Optional[str] = None,
) -> PopulationGeneStats:
    """Pool line-level sufficient statistics with gene-key-aware Chan merges."""
    source_stats = sorted(
        list(source_stats),
        key=lambda stats: (stats.cell_type, stats.fingerprint),
    )
    if not source_stats:
        raise ValueError("At least one line-level PopulationGeneStats is required")
    for stats in source_stats:
        if stats.dataset_name != str(dataset_name):
            raise ValueError(
                f"Cannot pool {stats.dataset_name!r} into dataset {dataset_name!r}"
            )
        if stats.scope != DATASET_CELL_TYPE_SCOPE:
            raise ValueError("Dataset-wide pooling requires line-level statistics")

    # Sorted union makes the pooled gene order independent of line-file input order.
    gene_keys = np.asarray(
        sorted(
            {
                str(gene_key)
                for stats in source_stats
                for gene_key in stats.gene_keys
            }
        ),
        dtype=str,
    )
    gene_to_position = {
        str(gene_key): position for position, gene_key in enumerate(gene_keys)
    }
    counts = np.zeros(len(gene_keys), dtype=np.int64)
    means = np.zeros(len(gene_keys), dtype=np.float64)
    m2 = np.zeros(len(gene_keys), dtype=np.float64)

    for stats in source_stats:
        positions = np.fromiter(
            (gene_to_position[str(gene_key)] for gene_key in stats.gene_keys),
            dtype=np.int64,
            count=stats.n_genes,
        )
        incoming_counts = np.zeros(len(gene_keys), dtype=np.int64)
        incoming_means = np.zeros(len(gene_keys), dtype=np.float64)
        incoming_m2 = np.zeros(len(gene_keys), dtype=np.float64)
        present = stats.finite_counts > 0
        present_positions = positions[present]
        incoming_counts[present_positions] = stats.finite_counts[present]
        incoming_means[present_positions] = stats.means[present]
        incoming_m2[present_positions] = (
            np.square(stats.population_sds[present])
            * stats.finite_counts[present]
        )
        _merge_summary_statistics(
            counts,
            means,
            m2,
            incoming_counts=incoming_counts,
            incoming_means=incoming_means,
            incoming_m2=incoming_m2,
        )

    expected_fingerprint = fingerprint or _dataset_fingerprint(
        dataset_name=dataset_name,
        source_stats=source_stats,
    )
    return _stats_from_accumulators(
        dataset_name=dataset_name,
        cell_type=DATASET_WIDE_CELL_TYPE,
        gene_keys=gene_keys,
        counts=counts,
        means=means,
        m2=m2,
        population_row_count=sum(
            stats.population_row_count for stats in source_stats
        ),
        fingerprint=expected_fingerprint,
        cache_path=cache_path,
        scope=DATASET_SCOPE,
    )


def load_or_fit_dataset_population_stats_from_source_stats(
    *,
    source_stats: Sequence[PopulationGeneStats],
    dataset_name: str,
    cache_root: Path,
    force: bool = False,
    verbose: bool = True,
) -> PopulationGeneStats:
    """Load or pool dataset-wide statistics from already available line stats.

    This is the zero-I/O aggregation path used by the precompute command's
    ``--scope both`` mode. It prevents reopening every source solely to recover the
    line statistics that were fitted or reloaded moments earlier.
    """
    source_stats = sorted(
        list(source_stats),
        key=lambda stats: (stats.cell_type, stats.fingerprint),
    )
    if not source_stats:
        raise ValueError("At least one line-level PopulationGeneStats is required")
    fingerprint = _dataset_fingerprint(
        dataset_name=dataset_name,
        source_stats=source_stats,
    )
    cache_path = dataset_stats_cache_path(cache_root, dataset_name)

    if not force:
        cached = _load_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
            dataset_name=dataset_name,
            cell_type=DATASET_WIDE_CELL_TYPE,
            expected_scope=DATASET_SCOPE,
        )
        if cached is not None:
            if verbose:
                print(
                    f"[z_score_stats:dataset] reloaded {dataset_name}: "
                    f"{cached.population_row_count:,} rows across "
                    f"{len(source_stats):,} cell types, "
                    f"{cached.n_valid_genes:,}/{cached.n_genes:,} valid genes",
                    flush=True,
                )
            return cached

    with _exclusive_cache_lock(
        cache_path,
        label=f"dataset-wide/{dataset_name}",
        verbose=verbose,
    ):
        if not force:
            cached = _load_cache(
                cache_path=cache_path,
                expected_fingerprint=fingerprint,
                dataset_name=dataset_name,
                cell_type=DATASET_WIDE_CELL_TYPE,
                expected_scope=DATASET_SCOPE,
            )
            if cached is not None:
                if verbose:
                    print(
                        f"[z_score_stats:dataset] reloaded after waiting "
                        f"{dataset_name}: "
                        f"{cached.n_valid_genes:,}/{cached.n_genes:,} valid genes",
                        flush=True,
                    )
                return cached

        started_at = time.monotonic()
        if verbose:
            print(
                f"[z_score_stats:dataset] pooling {dataset_name}: "
                f"{len(source_stats):,} line-stat caches",
                flush=True,
            )
        stats = fit_dataset_population_stats(
            dataset_name=dataset_name,
            source_stats=source_stats,
            cache_path=cache_path,
            fingerprint=fingerprint,
        )
        _write_cache(stats)

    if verbose:
        print(
            f"[z_score_stats:dataset] computed {dataset_name} in "
            f"{time.monotonic() - started_at:.1f}s: "
            f"{stats.population_row_count:,} rows across "
            f"{len(source_stats):,} cell types, "
            f"{stats.n_valid_genes:,}/{stats.n_genes:,} valid genes",
            flush=True,
        )
    return stats


def load_or_fit_dataset_population_stats(
    *,
    source_paths: Union[Mapping[str, Path], Sequence[tuple[str, Path]]],
    dataset_name: str,
    cache_root: Path,
    layer_name: str = "logFC",
    row_chunk_size: int = DEFAULT_ROW_CHUNK_SIZE,
    force: bool = False,
    force_source_stats: bool = False,
    verbose: bool = True,
) -> PopulationGeneStats:
    """Load or fit per-gene statistics pooled across a dataset's line files.

    ``source_paths`` maps each cell type to its line-level ``.h5ad``.  Line-level
    caches are loaded or fitted first, then their sufficient statistics are aligned by
    gene key and merged.  Source matrices are therefore scanned at most once.
    """
    canonical_sources = _canonical_source_paths(source_paths)
    source_stats = [
        load_or_fit_population_stats(
            source_path=source_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            cache_root=cache_root,
            layer_name=layer_name,
            row_chunk_size=row_chunk_size,
            force=force_source_stats,
            verbose=verbose,
        )
        for cell_type, source_path in canonical_sources
    ]
    return load_or_fit_dataset_population_stats_from_source_stats(
        source_stats=source_stats,
        dataset_name=dataset_name,
        cache_root=cache_root,
        force=force,
        verbose=verbose,
    )


def align_population_stats(
    stats: PopulationGeneStats,
    gene_keys: np.ndarray,
) -> PopulationGeneStats:
    """Return statistics in an exact requested gene order.

    This is primarily useful for applying dataset-wide union statistics to one
    line-level matrix.  Missing or duplicate requested keys are errors; genes are never
    silently dropped or reordered.
    """
    requested = np.asarray(gene_keys).astype(str)
    if requested.ndim != 1:
        raise ValueError("gene_keys must be one-dimensional")
    if len(set(requested.tolist())) != len(requested):
        raise ValueError("Requested gene_keys must be unique")
    source_positions = {
        str(gene_key): position
        for position, gene_key in enumerate(stats.gene_keys)
    }
    missing = [
        str(gene_key)
        for gene_key in requested
        if str(gene_key) not in source_positions
    ]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        raise ValueError(
            f"{len(missing)} requested genes are absent from cached "
            f"{stats.scope} statistics: {preview}{suffix}"
        )
    positions = np.fromiter(
        (source_positions[str(gene_key)] for gene_key in requested),
        dtype=np.int64,
        count=len(requested),
    )
    return PopulationGeneStats(
        dataset_name=stats.dataset_name,
        cell_type=stats.cell_type,
        gene_keys=requested,
        finite_counts=stats.finite_counts[positions],
        means=stats.means[positions],
        population_sds=stats.population_sds[positions],
        valid_mask=stats.valid_mask[positions],
        population_row_count=stats.population_row_count,
        fingerprint=stats.fingerprint,
        cache_path=stats.cache_path,
        scope=stats.scope,
    )


def _validate_gene_alignment(gene_keys: np.ndarray, stats: PopulationGeneStats) -> None:
    observed = np.asarray(gene_keys).astype(str)
    if observed.shape != stats.gene_keys.shape or not np.array_equal(
        observed, stats.gene_keys
    ):
        raise ValueError(
            f"Gene keys do not match cached z-score statistics for "
            f"{stats.dataset_name}/{stats.cell_type}"
        )


def standardize_vector(
    values: np.ndarray,
    *,
    gene_keys: np.ndarray,
    stats: PopulationGeneStats,
) -> np.ndarray:
    _validate_gene_alignment(gene_keys, stats)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size != stats.n_genes:
        raise ValueError("Vector length does not match population-gene statistics")
    standardized = np.full(values.size, np.nan, dtype=np.float64)
    valid = stats.valid_mask & np.isfinite(values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized[valid] = (
            values[valid] - stats.means[valid]
        ) / stats.population_sds[valid]
    standardized[~np.isfinite(standardized)] = np.nan
    return standardized


def standardize_matrix(
    values: np.ndarray,
    *,
    gene_keys: np.ndarray,
    stats: PopulationGeneStats,
) -> np.ndarray:
    _validate_gene_alignment(gene_keys, stats)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != stats.n_genes:
        raise ValueError("Matrix shape does not match population-gene statistics")
    standardized = np.full(values.shape, np.nan, dtype=np.float64)
    valid_genes = stats.valid_mask
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        standardized[:, valid_genes] = (
            values[:, valid_genes] - stats.means[None, valid_genes]
        ) / stats.population_sds[None, valid_genes]
    standardized[
        ~np.isfinite(values) | ~np.isfinite(standardized)
    ] = np.nan
    return standardized


def stats_qc_record(stats: PopulationGeneStats) -> dict[str, object]:
    finite_counts = stats.finite_counts[stats.valid_mask]
    return {
        "dataset_name": stats.dataset_name,
        "cell_type": stats.cell_type,
        "scope": stats.scope,
        "population_row_count": stats.population_row_count,
        "n_genes": stats.n_genes,
        "n_valid_genes": stats.n_valid_genes,
        "min_finite_count": int(finite_counts.min()) if finite_counts.size else 0,
        "median_finite_count": (
            float(np.median(finite_counts)) if finite_counts.size else float("nan")
        ),
        "max_finite_count": int(finite_counts.max()) if finite_counts.size else 0,
        "fingerprint": stats.fingerprint,
        "cache_path": str(stats.cache_path),
    }


class PopulationStatsCatalog:
    """Own z-score population policy, readiness, fitted statistics, and alignment.

    All three cross-source notebooks use this catalog. Metric-specific notebooks
    retain their own query construction and scoring, while this module owns the
    definition and lifecycle of the population standardization itself.
    """

    def __init__(
        self,
        *,
        source_dataset_dirs: Mapping[str, Path],
        matched_lines: Mapping[str, Sequence[str]],
        matched_dataset_names: Sequence[str],
        resolve_line_path: Callable[[str, str], Path],
        cache_root: Path,
        smoke_comparison_population: bool = False,
        precompute_mode: str = "wait",
        poll_seconds: float = 30.0,
        timeout_seconds: float = 12.0 * 60.0 * 60.0,
    ) -> None:
        precompute_mode = str(precompute_mode).strip().lower()
        if precompute_mode not in {"stop", "wait", "lazy"}:
            raise ValueError("precompute_mode must be stop, wait, or lazy")
        if float(poll_seconds) <= 0.0:
            raise ValueError("poll_seconds must be positive")
        if float(timeout_seconds) <= 0.0:
            raise ValueError("timeout_seconds must be positive")

        self.source_dataset_dirs = {
            str(dataset_name): Path(path)
            for dataset_name, path in source_dataset_dirs.items()
        }
        self.matched_lines = {
            str(dataset_name): sorted(
                {str(cell_type) for cell_type in cell_types}
            )
            for dataset_name, cell_types in matched_lines.items()
        }
        self.matched_dataset_names = [
            str(dataset_name) for dataset_name in matched_dataset_names
        ]
        self.resolve_line_path = resolve_line_path
        self.cache_root = Path(cache_root)
        self.smoke_comparison_population = bool(
            smoke_comparison_population
        )
        self.precompute_mode = precompute_mode
        self.poll_seconds = float(poll_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.stats_by_source: dict[
            tuple[str, str, str],
            PopulationGeneStats,
        ] = {}
        self.pooled_stats_by_dataset: dict[str, PopulationGeneStats] = {}

        unknown = sorted(
            set(self.matched_dataset_names) - set(self.source_dataset_dirs)
        )
        if unknown:
            raise KeyError(
                f"Matched datasets have no population source directory: {unknown}"
            )
        if self.smoke_comparison_population:
            print(
                "WARNING: z-score smoke population enabled; results are execution "
                "checks, not production reviewer estimates.",
                flush=True,
            )
        elif self.precompute_mode != "lazy":
            ensure_population_caches_ready(
                dataset_sources={
                    dataset_name: self.dataset_population_source_paths(
                        dataset_name
                    )
                    for dataset_name in self.matched_dataset_names
                },
                cache_root=self.cache_root,
                wait=self.precompute_mode == "wait",
                poll_seconds=self.poll_seconds,
                timeout_seconds=self.timeout_seconds,
            )

    @property
    def population_policy(self) -> str:
        return (
            "comparison_retained_lines_smoke"
            if self.smoke_comparison_population
            else "all_dataset_lines"
        )

    def dataset_population_source_paths(
        self,
        dataset_name: str,
    ) -> dict[str, Path]:
        dataset_name = str(dataset_name)
        if dataset_name not in self.source_dataset_dirs:
            raise KeyError(
                f"No population source directory for dataset {dataset_name!r}"
            )
        if not self.smoke_comparison_population:
            return discover_dataset_population_sources(
                self.source_dataset_dirs[dataset_name]
            )
        cell_types = self.matched_lines.get(dataset_name, [])
        return {
            cell_type: Path(
                self.resolve_line_path(dataset_name, cell_type)
            )
            for cell_type in cell_types
        }

    def get_stats(
        self,
        source,
        *,
        scale_variant: str,
    ) -> PopulationGeneStats:
        scale_variant = str(scale_variant)
        if scale_variant not in POPULATION_SCALE_VARIANTS:
            raise ValueError(
                f"Unknown z-score scale variant: {scale_variant!r}; expected "
                f"one of {POPULATION_SCALE_VARIANTS}"
            )
        key = (
            scale_variant,
            str(source.dataset_name),
            str(source.cell_type),
        )
        if key not in self.stats_by_source:
            if scale_variant == PER_GENE_DATASET_CELL_TYPE_VARIANT:
                stats = load_or_fit_population_stats(
                    source_path=source.path,
                    dataset_name=source.dataset_name,
                    cell_type=source.cell_type,
                    cache_root=self.cache_root,
                )
            else:
                dataset_name = str(source.dataset_name)
                if dataset_name not in self.pooled_stats_by_dataset:
                    self.pooled_stats_by_dataset[dataset_name] = (
                        load_or_fit_dataset_population_stats(
                            source_paths=self.dataset_population_source_paths(
                                dataset_name
                            ),
                            dataset_name=dataset_name,
                            cache_root=self.cache_root,
                        )
                    )
                stats = align_population_stats(
                    self.pooled_stats_by_dataset[dataset_name],
                    np.asarray(source.unique_gene_keys).astype(str),
                )
            expected_gene_keys = np.asarray(
                source.unique_gene_keys
            ).astype(str)
            if not np.array_equal(stats.gene_keys, expected_gene_keys):
                raise AssertionError(
                    f"z-score gene order disagrees with LineSource for {key}"
                )
            self.stats_by_source[key] = stats
        return self.stats_by_source[key]

    def standardize_vector(
        self,
        source,
        values: np.ndarray,
        *,
        scale_variant: str,
    ) -> np.ndarray:
        return standardize_vector(
            values,
            gene_keys=source.unique_gene_keys,
            stats=self.get_stats(source, scale_variant=scale_variant),
        )

    def standardize_matrix(
        self,
        source,
        values: np.ndarray,
        *,
        scale_variant: str,
    ) -> np.ndarray:
        return standardize_matrix(
            values,
            gene_keys=source.unique_gene_keys,
            stats=self.get_stats(source, scale_variant=scale_variant),
        )

    def standardize_aligned_vector(
        self,
        source,
        values: np.ndarray,
        source_gene_positions: np.ndarray,
        *,
        scale_variant: str,
    ) -> np.ndarray:
        stats = self.get_stats(source, scale_variant=scale_variant)
        positions = np.asarray(source_gene_positions, dtype=np.int64)
        selected = np.asarray(values, dtype=np.float64).reshape(-1)
        if selected.size != positions.size:
            raise ValueError(
                "Aligned vector width does not match source_gene_positions"
            )
        standardized = np.full(selected.shape, np.nan, dtype=np.float64)
        valid = stats.valid_mask[positions] & np.isfinite(selected)
        with np.errstate(
            invalid="ignore",
            divide="ignore",
            over="ignore",
        ):
            standardized[valid] = (
                selected[valid] - stats.means[positions][valid]
            ) / stats.population_sds[positions][valid]
        standardized[~np.isfinite(standardized)] = np.nan
        return standardized

    def standardize_aligned_matrix(
        self,
        source,
        values: np.ndarray,
        source_gene_positions: np.ndarray,
        *,
        scale_variant: str,
    ) -> np.ndarray:
        stats = self.get_stats(source, scale_variant=scale_variant)
        positions = np.asarray(source_gene_positions, dtype=np.int64)
        selected = np.atleast_2d(np.asarray(values, dtype=np.float64))
        if selected.shape[1] != positions.size:
            raise ValueError(
                "Aligned matrix width does not match source_gene_positions"
            )
        standardized = np.full(selected.shape, np.nan, dtype=np.float64)
        valid_genes = stats.valid_mask[positions]
        with np.errstate(
            invalid="ignore",
            divide="ignore",
            over="ignore",
        ):
            standardized[:, valid_genes] = (
                selected[:, valid_genes]
                - stats.means[positions][None, valid_genes]
            ) / stats.population_sds[positions][None, valid_genes]
        standardized[
            ~np.isfinite(selected) | ~np.isfinite(standardized)
        ] = np.nan
        return standardized
