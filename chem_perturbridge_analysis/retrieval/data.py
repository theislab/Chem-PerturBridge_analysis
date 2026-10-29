from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from .ashr import attach_layers as attach_ashr_layers
from .ashr import sidecar_root_from_env

REQUIRED_OBS_COLUMNS = ("cell_type", "pubchem_cid", "pert_time_h", "pert_dose_uM")


@dataclass
class PubchemCIDGroup:
    rows: np.ndarray
    times: np.ndarray
    doses: np.ndarray


@dataclass
class CellTypeData:
    dataset_name: str
    cell_type: str
    adata: ad.AnnData
    obs: pd.DataFrame
    pubchem_cid_groups: dict[str, PubchemCIDGroup]


class DatasetStore:
    """Loads per-cell-type differential expression AnnData from file-dir or merged h5ad."""

    def __init__(self, dataset_name: str, dataset_path: Path, cache_enabled: bool = True):
        self.dataset_name = dataset_name
        self.dataset_path = Path(dataset_path)
        self.cache_enabled = cache_enabled
        self.kind = self._detect_kind(self.dataset_path)
        self._cell_cache: dict[str, ad.AnnData] = {}
        self._cell_type_cache: Optional[list[str]] = None
        self._combined_adata: Optional[ad.AnnData] = None

    @staticmethod
    def _detect_kind(path: Path) -> str:
        if path.is_dir():
            return "files"
        if path.is_file() and path.suffix == ".h5ad":
            return "combined_h5ad"
        raise FileNotFoundError(f"Unsupported dataset path: {path}")

    @staticmethod
    def _cell_type_from_filename(path: Path) -> str:
        return re.sub(r"_de\.h5ad$", "", path.name)

    def _load_combined(self) -> ad.AnnData:
        if self._combined_adata is None:
            self._combined_adata = ad.read_h5ad(self.dataset_path)
        return self._combined_adata

    def list_cell_types(self) -> list[str]:
        if self._cell_type_cache is not None:
            return self._cell_type_cache

        if self.kind == "files":
            cell_types = sorted(
                {self._cell_type_from_filename(p) for p in self.dataset_path.glob("*_de.h5ad")}
            )
            if not cell_types:
                cell_types = sorted(
                    {self._cell_type_from_filename(p) for p in self.dataset_path.glob("*.h5ad")}
                )
        else:
            adata = ad.read_h5ad(self.dataset_path, backed="r")
            try:
                if "cell_type" not in adata.obs.columns:
                    raise KeyError(
                        f"{self.dataset_name}: expected obs['cell_type'] in {self.dataset_path}"
                    )
                cell_types = sorted(pd.Index(adata.obs["cell_type"].astype(str)).unique().tolist())
            finally:
                adata.file.close()

        self._cell_type_cache = cell_types
        return cell_types

    def load_cell_type(self, cell_type: str) -> Optional[ad.AnnData]:
        if cell_type in self._cell_cache:
            return self._cell_cache[cell_type]

        loaded: Optional[ad.AnnData]
        if self.kind == "files":
            file_path = self.dataset_path / f"{cell_type}_de.h5ad"
            if not file_path.exists():
                loaded = None
            else:
                loaded = ad.read_h5ad(file_path)
        else:
            full = self._load_combined()
            if "cell_type" not in full.obs.columns:
                raise KeyError(
                    f"{self.dataset_name}: expected obs['cell_type'] in merged {self.dataset_path}"
                )
            mask = full.obs["cell_type"].astype(str) == str(cell_type)
            if int(mask.sum()) == 0:
                loaded = None
            else:
                loaded = full[mask].copy()

        ashr_root = sidecar_root_from_env()
        if loaded is not None and ashr_root is not None:
            attach_ashr_layers(loaded, self.dataset_name, cell_type, ashr_root)

        if self.cache_enabled:
            self._cell_cache[cell_type] = loaded
        return loaded


def build_obs_table(adata: ad.AnnData, dataset_name: str, cell_type: str) -> pd.DataFrame:
    missing = [col for col in REQUIRED_OBS_COLUMNS if col not in adata.obs.columns]
    if missing:
        raise KeyError(
            f"{dataset_name}:{cell_type} missing required obs columns {missing}. "
            f"Available: {list(adata.obs.columns)}"
        )

    obs = adata.obs.copy()
    obs["_row"] = np.arange(adata.n_obs, dtype=np.int32)
    obs["cell_type"] = obs["cell_type"].astype(str)
    obs["pubchem_cid"] = obs["pubchem_cid"].astype("string")
    obs["pert_time_h"] = pd.to_numeric(obs["pert_time_h"], errors="coerce")
    obs["pert_dose_uM"] = pd.to_numeric(obs["pert_dose_uM"], errors="coerce")
    return obs


def build_pubchem_cid_groups(obs: pd.DataFrame) -> dict[str, PubchemCIDGroup]:
    valid = (
        obs["pubchem_cid"].notna()
        & np.isfinite(obs["pert_time_h"].to_numpy(dtype=float))
        & np.isfinite(obs["pert_dose_uM"].to_numpy(dtype=float))
    )
    grouped: dict[str, PubchemCIDGroup] = {}
    for pubchem_cid, grp in obs.loc[valid].groupby("pubchem_cid", sort=False):
        grouped[str(pubchem_cid)] = PubchemCIDGroup(
            rows=grp["_row"].to_numpy(dtype=np.int32),
            times=grp["pert_time_h"].to_numpy(dtype=np.float64),
            doses=grp["pert_dose_uM"].to_numpy(dtype=np.float64),
        )
    return grouped


def load_cell_type_data(
    store: DatasetStore, cell_type: str, cache: dict[tuple[str, str], Optional[CellTypeData]]
) -> Optional[CellTypeData]:
    cache_key = (store.dataset_name, cell_type)
    if cache_key in cache:
        return cache[cache_key]

    adata = store.load_cell_type(cell_type)
    if adata is None:
        cache[cache_key] = None
        return None

    obs = build_obs_table(adata, store.dataset_name, cell_type)
    grouped = build_pubchem_cid_groups(obs)
    cell_data = CellTypeData(
        dataset_name=store.dataset_name,
        cell_type=cell_type,
        adata=adata,
        obs=obs,
        pubchem_cid_groups=grouped,
    )
    cache[cache_key] = cell_data
    return cell_data


def best_row_for_group(group: PubchemCIDGroup, time_h: float, dose_um: float) -> int:
    dt = np.abs(group.times - time_h)
    min_dt = np.min(dt)
    time_ties = np.flatnonzero(dt == min_dt)
    if time_ties.size == 1:
        return int(group.rows[time_ties[0]])
    tied_doses = group.doses[time_ties]
    dd = np.abs(tied_doses - dose_um)
    if np.isfinite(dose_um) and dose_um > 0.0:
        positive_mask = np.isfinite(tied_doses) & (tied_doses > 0.0)
        if np.any(positive_mask):
            # Prefer log-dose distance because dose-response effects are typically scale-based.
            dd[positive_mask] = np.abs(
                np.log(tied_doses[positive_mask]) - np.log(dose_um)
            )
    winner = time_ties[int(np.argmin(dd))]
    return int(group.rows[winner])


def candidate_rows_for_queries(
    pubchem_cid_groups: dict[str, PubchemCIDGroup],
    query_times: np.ndarray,
    query_doses: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """
    Returns pubchem_cid order + candidate row matrix with shape [n_query, n_pubchem_cids].
    Missing query time/dose rows are marked as -1.
    """
    pubchem_cids = sorted(pubchem_cid_groups.keys())
    n_queries = query_times.shape[0]
    n_pubchem_cids = len(pubchem_cids)
    out = np.full((n_queries, n_pubchem_cids), -1, dtype=np.int32)

    if n_pubchem_cids == 0:
        return pubchem_cids, out

    for j, cid in enumerate(pubchem_cids):
        group = pubchem_cid_groups[cid]
        for i in range(n_queries):
            t = float(query_times[i])
            d = float(query_doses[i])
            if not (np.isfinite(t) and np.isfinite(d)):
                continue
            out[i, j] = best_row_for_group(group, time_h=t, dose_um=d)

    return pubchem_cids, out
