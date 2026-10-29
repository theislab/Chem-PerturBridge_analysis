"""Reusable in-memory strata for cross-source centroid and peer baselines.

The reviewer-addition notebooks repeatedly score signatures within one already
filtered source/context stratum.  Reloading that stratum or recomputing its
different-compound centroid for every matched row is unnecessarily expensive for
large CIGS sources.  :class:`SignatureStratum` keeps the unique-gene matrix, finite
column totals, and row identities together so callers can:

* derive the exact finite-mean centroid after excluding every row for one compound;
* obtain deterministic absolute row indices for capped individual-peer scoring; and
* reuse prepared Spearman ranks for repeated queries on the same gene positions.

This module deliberately does not load ``.h5ad`` files or decide which rows belong in
a stratum.  The notebooks retain those policy decisions and pass the resulting
matrix here.  That keeps the optimization small and makes parity straightforward to
test.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

from scripts.peer_baselines import (
    DEFAULT_PEER_SAMPLING_SEED,
    PreparedSpearmanRows,
    exact_mean_excluding_row_mask,
    finite_column_totals,
    prepare_spearman_rows,
    select_peer_indices,
)

__all__ = [
    "DEFAULT_PREPARED_CACHE_BYTES",
    "PeerSelection",
    "SignatureStratum",
]


DEFAULT_PREPARED_CACHE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class PeerSelection:
    """Absolute matrix row indices selected from a different-compound peer set."""

    total_count: int
    row_indices: np.ndarray

    @property
    def selected_count(self) -> int:
        return int(self.row_indices.size)


class _ByteLimitedLRU:
    """Tiny byte-aware LRU used for the potentially large prepared-rank matrices."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max(0, int(max_bytes))
        self.current_bytes = 0
        self._entries = OrderedDict()

    def get(self, key: bytes) -> Optional[PreparedSpearmanRows]:
        entry = self._entries.pop(key, None)
        if entry is None:
            return None
        self._entries[key] = entry
        return entry[0]

    def put(
        self,
        key: bytes,
        value: PreparedSpearmanRows,
        *,
        size_bytes: int,
    ) -> None:
        size_bytes = max(0, int(size_bytes))
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.current_bytes -= int(previous[1])
        if self.max_bytes == 0 or size_bytes > self.max_bytes:
            return
        while self._entries and self.current_bytes + size_bytes > self.max_bytes:
            _, (_, removed_bytes) = self._entries.popitem(last=False)
            self.current_bytes -= int(removed_bytes)
        self._entries[key] = (value, size_bytes)
        self.current_bytes += size_bytes

    @property
    def entry_count(self) -> int:
        return len(self._entries)


class SignatureStratum:
    """One source/context population aligned to a unique-gene column order.

    Parameters
    ----------
    compounds
        Compound identifier for every matrix row. Duplicate identifiers are expected:
        excluding a query compound removes all of its rows.
    values
        Raw condition-level values with shape ``(rows, unique genes)``.
    gene_keys
        Optional unique gene identifiers documenting the matrix column order.
    max_prepared_cache_bytes
        Per-stratum bound for prepared Spearman ranks. An entry larger than the bound
        is returned to the caller but not retained. Set to zero to disable reuse.

    Notes
    -----
    The class avoids copying floating-point ``values``. Callers must therefore treat
    the supplied array as immutable for the lifetime of the stratum; changing it would
    invalidate the stored finite sums and ranks.
    """

    def __init__(
        self,
        compounds: np.ndarray,
        values: np.ndarray,
        *,
        gene_keys: Optional[np.ndarray] = None,
        max_prepared_cache_bytes: int = DEFAULT_PREPARED_CACHE_BYTES,
    ):
        normalized_values = np.asarray(values)
        if normalized_values.ndim != 2:
            raise ValueError("values must be a two-dimensional row-by-gene matrix")
        if not np.issubdtype(normalized_values.dtype, np.floating):
            normalized_values = normalized_values.astype(np.float32)
        normalized_compounds = np.asarray(compounds).astype(str).reshape(-1)
        if normalized_compounds.size != normalized_values.shape[0]:
            raise ValueError("compounds length does not match values rows")

        if gene_keys is None:
            normalized_gene_keys = None
        else:
            normalized_gene_keys = np.asarray(gene_keys).astype(str).reshape(-1)
            if normalized_gene_keys.size != normalized_values.shape[1]:
                raise ValueError("gene_keys length does not match values columns")
            if np.unique(normalized_gene_keys).size != normalized_gene_keys.size:
                raise ValueError("gene_keys must already be unique")

        self.compounds = normalized_compounds
        self.values = normalized_values
        self.gene_keys = normalized_gene_keys
        self.finite_sums, self.finite_counts = finite_column_totals(self.values)
        self._compound_rows = {}
        self._prepared_cache = _ByteLimitedLRU(max_prepared_cache_bytes)

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_genes(self) -> int:
        return int(self.values.shape[1])

    def compound_row_indices(self, compound: str) -> np.ndarray:
        """Return absolute row indices for every occurrence of ``compound``."""
        compound = str(compound)
        cached = self._compound_rows.get(compound)
        if cached is None:
            cached = np.flatnonzero(self.compounds == compound).astype(np.int64)
            self._compound_rows[compound] = cached
        return cached

    def different_compound_peer_indices(
        self,
        excluded_compound: str,
    ) -> np.ndarray:
        """Return all absolute row indices not belonging to ``excluded_compound``."""
        excluded_rows = self.compound_row_indices(excluded_compound)
        if excluded_rows.size == 0:
            return np.arange(self.n_rows, dtype=np.int64)
        keep = np.ones(self.n_rows, dtype=bool)
        keep[excluded_rows] = False
        return np.flatnonzero(keep).astype(np.int64)

    def different_compound_centroid(
        self,
        excluded_compound: str,
        *,
        require_all_finite: bool = False,
    ) -> Optional[np.ndarray]:
        """Gene-wise mean after excluding all rows for one compound.

        By default this has the missing-value semantics of ``numpy.nanmean``: genes
        with no finite remaining values are NaN. With ``require_all_finite=True``, a
        gene is NaN when any remaining row is non-finite, matching the historical
        notebook's ``numpy.mean`` centroid exactly. ``None`` is returned when no gene
        has a finite value after exclusion.
        """
        excluded_rows = self.compound_row_indices(excluded_compound)
        excluded_mask = np.zeros(self.n_rows, dtype=bool)
        excluded_mask[excluded_rows] = True
        centroid = exact_mean_excluding_row_mask(
            self.values,
            excluded_mask,
            total_sums=self.finite_sums,
            total_counts=self.finite_counts,
        )
        n_remaining = self.n_rows - int(excluded_rows.size)
        if centroid is None and n_remaining > 0:
            centroid = np.full(self.n_genes, np.nan, dtype=np.float64)
        if centroid is None or not require_all_finite:
            return centroid

        excluded_values = self.values[excluded_rows]
        excluded_counts = np.isfinite(excluded_values).sum(axis=0, dtype=np.int64)
        remaining_counts = self.finite_counts - excluded_counts
        centroid = np.asarray(centroid, dtype=np.float64)
        centroid[remaining_counts != n_remaining] = np.nan
        return centroid

    def select_different_compound_peers(
        self,
        excluded_compound: str,
        *,
        max_peers: Optional[int],
        seed_key: str,
        sampling_seed: int = DEFAULT_PEER_SAMPLING_SEED,
    ) -> PeerSelection:
        """Select deterministic peers and report both total and selected counts.

        ``seed_key`` is forwarded unchanged to the established
        :func:`select_peer_indices` helper so existing notebook seed conventions retain
        exactly the same draws.
        """
        eligible_rows = self.different_compound_peer_indices(excluded_compound)
        selected_offsets = select_peer_indices(
            int(eligible_rows.size),
            max_peers,
            seed_key,
            sampling_seed=sampling_seed,
        )
        return PeerSelection(
            total_count=int(eligible_rows.size),
            row_indices=eligible_rows[selected_offsets],
        )

    def prepared_spearman(
        self,
        gene_positions: Optional[np.ndarray] = None,
    ) -> PreparedSpearmanRows:
        """Return reusable row ranks for the requested ordered gene positions.

        The cache key preserves both position membership and order. Passing every
        column in natural order shares the all-gene entry with ``gene_positions=None``.
        """
        positions = self._normalize_gene_positions(gene_positions)
        if positions is None:
            cache_key = b"all"
            selected_values = self.values
        else:
            position_bytes = positions.astype("<i8", copy=False).tobytes()
            cache_key = b"positions:" + position_bytes
            selected_values = self.values[:, positions]

        cached = self._prepared_cache.get(cache_key)
        if cached is not None:
            return cached

        prepared = prepare_spearman_rows(selected_values)
        incremental_bytes = (
            int(prepared.normalized_ranks.nbytes)
            + int(prepared.finite_rows.nbytes)
            + int(prepared.valid_rows.nbytes)
        )
        if not np.shares_memory(prepared.values, self.values):
            incremental_bytes += int(prepared.values.nbytes)
        self._prepared_cache.put(
            cache_key,
            prepared,
            size_bytes=incremental_bytes,
        )
        return prepared

    @property
    def prepared_cache_entry_count(self) -> int:
        return self._prepared_cache.entry_count

    @property
    def prepared_cache_bytes(self) -> int:
        return int(self._prepared_cache.current_bytes)

    def _normalize_gene_positions(
        self,
        gene_positions: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if gene_positions is None:
            return None
        positions = np.asarray(gene_positions)
        if positions.dtype == bool:
            positions = positions.reshape(-1)
            if positions.size != self.n_genes:
                raise ValueError("boolean gene_positions must match the number of genes")
            positions = np.flatnonzero(positions)
        else:
            if positions.ndim != 1:
                raise ValueError("gene_positions must be one-dimensional")
            if not np.issubdtype(positions.dtype, np.integer):
                raise TypeError("gene_positions must contain integer positions")
            positions = positions.astype(np.int64, copy=False)
        if positions.size:
            if int(positions.min()) < 0 or int(positions.max()) >= self.n_genes:
                raise IndexError("gene_positions contains an out-of-range column")
            if np.unique(positions).size != positions.size:
                raise ValueError("gene_positions must not contain duplicates")
        if np.array_equal(positions, np.arange(self.n_genes, dtype=np.int64)):
            return None
        return positions
