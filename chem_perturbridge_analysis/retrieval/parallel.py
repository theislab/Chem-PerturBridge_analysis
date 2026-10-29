from __future__ import annotations

import re
import time
from os import PathLike
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd

from .config import RetrievalSettings
from .data import DatasetStore, best_row_for_group, load_cell_type_data
from .engine import (
    DERIVED_POPULATION_ZSCORE_DATASET_REP,
    DERIVED_POPULATION_ZSCORE_REP,
    _available_representations,
    run_cross_dataset_retrieval,
    summarize_retrieval,
)

TRUTH_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "query_obs_id",
    "query_row",
    "pubchem_cid",
    "pert_time_h",
    "pert_dose_uM",
    "gt_db_cell_type",
    "gt_db_obs_id",
    "gt_db_row",
    "n_genes_gt_pair",
]

TRUTH_SUMMARY_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "n_query_obs",
    "n_shared_pubchem_cids",
    "n_truth_queries",
    "n_genes_gt_pair",
]

TASK_COLUMNS = [
    "task_id",
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "representation",
    "representation_slug",
    "n_truth_queries",
    "n_truth_cell_types",
    "n_eligible_query_cell_types",
    "task_detail_file",
]

DETAIL_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "query_obs_id",
    "pubchem_cid",
    "pert_time_h",
    "pert_dose_uM",
    "representation",
    "metric",
    "n_candidates",
    "rank",
    "rank_over_n",
    "rank_normalized",
    "retrieval_score",
    "n_genes_gt_pair",
]

SUMMARY_BASE_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "representation",
    "metric",
    "n_queries",
    "n_candidates_mean",
    "n_candidates_median",
    "rank_mean",
    "rank_median",
    "rank_over_n_mean",
    "rank_normalized_mean",
    "retrieval_score_mean",
    "retrieval_score_median",
]

SUMMARY_BY_CELL_COLUMNS = [
    "query_dataset",
    "db_dataset",
    "query_cell_type",
    "representation",
    "metric",
    "n_queries",
    "n_candidates_mean",
    "n_candidates_median",
    "rank_mean",
    "rank_median",
    "rank_over_n_mean",
    "rank_normalized_mean",
    "retrieval_score_mean",
    "retrieval_score_median",
]

SUMMARY_OVERALL_COLUMNS = list(SUMMARY_BASE_COLUMNS)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def slugify_representation(representation: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", representation).strip("_").lower()
    return slug or "rep"


def output_filename(output_prefix: str, name: str) -> str:
    return f"{output_prefix}_{name}" if output_prefix else name


def task_detail_filename(
    output_prefix: str,
    task_id: int,
    query_dataset: str,
    db_dataset: str,
    representation_slug: str,
    query_cell_type: str = "",
) -> str:
    cell_suffix = ""
    if query_cell_type:
        cell_suffix = f"_{slugify_representation(query_cell_type)}"
    return output_filename(
        output_prefix,
        f"task{int(task_id):04d}_"
        f"{query_dataset}_to_{db_dataset}_{representation_slug}{cell_suffix}_detail.csv",
    )


def _task_query_cell_type(task: pd.Series) -> str:
    if "query_cell_type" not in task.index:
        return ""
    value = task["query_cell_type"]
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def build_truth_matches_and_tasks(
    dataset_paths: dict[str, str | PathLike[str]],
    query_datasets: list[str],
    db_datasets: list[str],
    settings: Optional[RetrievalSettings] = None,
    include_self_dataset: bool = False,
    cell_type_filter: Optional[set[str]] = None,
    cache_cell_types: bool = True,
    split_by_cell_type: bool = False,
    output_prefix: str = "cross_dataset_retrieval_parallel",
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    settings = settings or RetrievalSettings()
    stores = {
        name: DatasetStore(dataset_name=name, dataset_path=Path(path), cache_enabled=cache_cell_types)
        for name, path in dataset_paths.items()
    }
    cache: dict[tuple[str, str], object] = {}

    truth_rows: list[dict] = []
    summary_rows: list[dict] = []
    task_rows: list[dict] = []
    next_task_id = 1
    run_t0 = time.perf_counter()

    for query_dataset in query_datasets:
        for db_dataset in db_datasets:
            if not include_self_dataset and query_dataset == db_dataset:
                continue
            if query_dataset not in stores:
                raise KeyError(f"Unknown query dataset: {query_dataset}")
            if db_dataset not in stores:
                raise KeyError(f"Unknown db dataset: {db_dataset}")

            pair_t0 = time.perf_counter()
            query_store = stores[query_dataset]
            db_store = stores[db_dataset]
            query_cell_types = query_store.list_cell_types()
            db_cell_types = db_store.list_cell_types()
            if cell_type_filter is not None:
                query_cell_types = [ct for ct in query_cell_types if ct in cell_type_filter]
                db_cell_types = [ct for ct in db_cell_types if ct in cell_type_filter]

            eligible_query_cell_types = sorted(set(query_cell_types) & set(db_cell_types))
            if not eligible_query_cell_types:
                _log(
                    verbose,
                    f"[precompute] pair {query_dataset}->{db_dataset}: no overlapping cell types; skipping",
                )
                continue

            _log(
                verbose,
                f"[precompute] pair {query_dataset}->{db_dataset}: "
                f"eligible_query_cell_types={len(eligible_query_cell_types)}",
            )

            pair_representations: set[str] = set()
            cell_type_representations: dict[str, set[str]] = {}
            cell_type_truth_queries: dict[str, int] = {}
            pair_truth_queries = 0
            pair_truth_cell_types = 0

            for idx, query_cell_type in enumerate(eligible_query_cell_types, start=1):
                query_data = load_cell_type_data(query_store, query_cell_type, cache)
                db_data = load_cell_type_data(db_store, query_cell_type, cache)

                if query_data is None or query_data.adata.n_obs == 0:
                    _log(
                        verbose,
                        f"[precompute] pair {query_dataset}->{db_dataset} "
                        f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type}: "
                        "query data missing or empty",
                    )
                    continue
                if db_data is None or db_data.adata.n_obs == 0:
                    _log(
                        verbose,
                        f"[precompute] pair {query_dataset}->{db_dataset} "
                        f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type}: "
                        "db data missing or empty",
                    )
                    continue

                shared_pubchem_cids = set(query_data.pubchem_cid_groups) & set(db_data.pubchem_cid_groups)
                if not shared_pubchem_cids:
                    _log(
                        verbose,
                        f"[precompute] pair {query_dataset}->{db_dataset} "
                        f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type}: "
                        "no shared pubchem_cids",
                    )
                    continue

                cell_representations = set(
                    _available_representations(
                        query_data.adata,
                        db_data.adata,
                        settings,
                        query_dataset=query_dataset,
                        db_dataset=db_dataset,
                        query_cell_type=query_data.cell_type,
                        db_cell_type=db_data.cell_type,
                    )
                )
                if settings.include_representations:
                    cell_representations &= set(settings.include_representations)
                if settings.skip_representations:
                    cell_representations -= set(settings.skip_representations)
                pair_representations |= cell_representations
                n_genes = int(
                    np.intersect1d(
                        query_data.adata.var_names.values,
                        db_data.adata.var_names.values,
                        assume_unique=False,
                    ).size
                )

                query_pubchem_cids = query_data.obs["pubchem_cid"].to_numpy(dtype=object)
                query_cid_valid = query_data.obs["pubchem_cid"].notna().to_numpy(dtype=bool)
                query_times = query_data.obs["pert_time_h"].to_numpy(dtype=np.float64)
                query_doses = query_data.obs["pert_dose_uM"].to_numpy(dtype=np.float64)
                query_obs_names = query_data.adata.obs_names.astype(str).to_numpy()
                db_obs_names = db_data.adata.obs_names.astype(str).to_numpy()

                cell_truth_queries = 0
                for query_row in range(query_data.adata.n_obs):
                    if not query_cid_valid[query_row]:
                        continue
                    if not (
                        np.isfinite(query_times[query_row]) and np.isfinite(query_doses[query_row])
                    ):
                        continue

                    pubchem_cid = str(query_pubchem_cids[query_row])
                    group = db_data.pubchem_cid_groups.get(pubchem_cid)
                    if group is None:
                        continue

                    gt_db_row = best_row_for_group(
                        group,
                        time_h=float(query_times[query_row]),
                        dose_um=float(query_doses[query_row]),
                    )
                    if gt_db_row < 0 or gt_db_row >= db_data.adata.n_obs:
                        continue

                    truth_rows.append(
                        {
                            "query_dataset": query_dataset,
                            "db_dataset": db_dataset,
                            "query_cell_type": query_cell_type,
                            "query_obs_id": str(query_obs_names[query_row]),
                            "query_row": int(query_row),
                            "pubchem_cid": pubchem_cid,
                            "pert_time_h": float(query_times[query_row]),
                            "pert_dose_uM": float(query_doses[query_row]),
                            "gt_db_cell_type": query_cell_type,
                            "gt_db_obs_id": str(db_obs_names[gt_db_row]),
                            "gt_db_row": int(gt_db_row),
                            "n_genes_gt_pair": float(n_genes),
                        }
                    )
                    cell_truth_queries += 1

                summary_rows.append(
                    {
                        "query_dataset": query_dataset,
                        "db_dataset": db_dataset,
                        "query_cell_type": query_cell_type,
                        "n_query_obs": int(query_data.adata.n_obs),
                        "n_shared_pubchem_cids": int(len(shared_pubchem_cids)),
                        "n_truth_queries": int(cell_truth_queries),
                        "n_genes_gt_pair": float(n_genes),
                    }
                )

                if cell_truth_queries > 0:
                    pair_truth_cell_types += 1
                    if split_by_cell_type and cell_representations:
                        cell_type_truth_queries[query_cell_type] = int(cell_truth_queries)
                        cell_type_representations[query_cell_type] = set(cell_representations)
                pair_truth_queries += cell_truth_queries

                _log(
                    verbose,
                    f"[precompute] pair {query_dataset}->{db_dataset} "
                    f"{idx}/{len(eligible_query_cell_types)} cell_type={query_cell_type}: "
                    f"truth_queries={cell_truth_queries}",
                )

            sorted_representations = sorted(pair_representations)

            if pair_truth_queries <= 0:
                _log(
                    verbose,
                    f"[precompute] pair {query_dataset}->{db_dataset}: "
                    "no truth queries; no tasks scheduled",
                )
                sorted_representations = []

            if split_by_cell_type:
                for query_cell_type in sorted(cell_type_representations):
                    cell_truth_queries = int(cell_type_truth_queries.get(query_cell_type, 0))
                    if cell_truth_queries <= 0:
                        continue

                    for representation in sorted(cell_type_representations[query_cell_type]):
                        representation_slug = slugify_representation(representation)
                        task_rows.append(
                            {
                                "task_id": int(next_task_id),
                                "query_dataset": query_dataset,
                                "db_dataset": db_dataset,
                                "query_cell_type": query_cell_type,
                                "representation": representation,
                                "representation_slug": representation_slug,
                                "n_truth_queries": cell_truth_queries,
                                "n_truth_cell_types": 1,
                                "n_eligible_query_cell_types": int(len(eligible_query_cell_types)),
                                "task_detail_file": task_detail_filename(
                                    output_prefix=output_prefix,
                                    task_id=next_task_id,
                                    query_dataset=query_dataset,
                                    db_dataset=db_dataset,
                                    representation_slug=representation_slug,
                                    query_cell_type=query_cell_type,
                                ),
                            }
                        )
                        next_task_id += 1
            else:
                for representation in sorted_representations:
                    representation_slug = slugify_representation(representation)
                    task_rows.append(
                        {
                            "task_id": int(next_task_id),
                            "query_dataset": query_dataset,
                            "db_dataset": db_dataset,
                            "query_cell_type": "",
                            "representation": representation,
                            "representation_slug": representation_slug,
                            "n_truth_queries": int(pair_truth_queries),
                            "n_truth_cell_types": int(pair_truth_cell_types),
                            "n_eligible_query_cell_types": int(len(eligible_query_cell_types)),
                            "task_detail_file": task_detail_filename(
                                output_prefix=output_prefix,
                                task_id=next_task_id,
                                query_dataset=query_dataset,
                                db_dataset=db_dataset,
                                representation_slug=representation_slug,
                            ),
                        }
                    )
                    next_task_id += 1

            _log(
                verbose,
                f"[precompute] pair {query_dataset}->{db_dataset}: "
                f"truth_queries={pair_truth_queries} "
                f"truth_cell_types={pair_truth_cell_types} "
                f"representations={len(sorted_representations)} "
                f"elapsed_s={time.perf_counter() - pair_t0:.1f}",
            )

    truth_df = pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)
    if not truth_df.empty:
        truth_df = truth_df.sort_values(
            ["query_dataset", "db_dataset", "query_cell_type", "query_obs_id"]
        ).reset_index(drop=True)

    truth_summary_df = pd.DataFrame(summary_rows, columns=TRUTH_SUMMARY_COLUMNS)
    if not truth_summary_df.empty:
        truth_summary_df = truth_summary_df.sort_values(
            ["query_dataset", "db_dataset", "query_cell_type"]
        ).reset_index(drop=True)

    tasks_df = pd.DataFrame(task_rows, columns=TASK_COLUMNS)
    if not tasks_df.empty:
        tasks_df = tasks_df.sort_values(["task_id"]).reset_index(drop=True)

    if settings.include_representations:
        requested_z = settings.include_representations & {
            DERIVED_POPULATION_ZSCORE_REP,
            DERIVED_POPULATION_ZSCORE_DATASET_REP,
        }
        scheduled = set(tasks_df["representation"]) if not tasks_df.empty else set()
        missing_z = requested_z - scheduled
        if missing_z:
            raise FileNotFoundError(
                "Requested population z-score representation(s) "
                f"{sorted(missing_z)} were not scheduled on any pair. "
                "Either caches are missing or the representation name "
                "was dropped before precompute (parentheses in sbatch --wrap). "
                "Use the aliases per_gene_population_zscore_dataset_cell_type / "
                "per_gene_population_zscore_dataset, and run "
                "scripts/slurm/run_precompute_population_zscore.sbatch first."
            )

    _log(
        verbose,
        f"[precompute] done total_truth_rows={len(truth_df)} "
        f"total_tasks={len(tasks_df)} elapsed_s={time.perf_counter() - run_t0:.1f}",
    )
    return truth_df, truth_summary_df, tasks_df


def write_precompute_outputs(
    truth_df: pd.DataFrame,
    truth_summary_df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    output_dir: Path,
    output_prefix: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_path = output_dir / output_filename(output_prefix, "truth_matches.csv")
    truth_summary_path = output_dir / output_filename(output_prefix, "truth_summary.csv")
    tasks_path = output_dir / output_filename(output_prefix, "tasks.csv")

    truth_df.to_csv(truth_path, index=False)
    truth_summary_df.to_csv(truth_summary_path, index=False)
    tasks_df.to_csv(tasks_path, index=False)
    return truth_path, truth_summary_path, tasks_path


def _append_match_id(
    mapping: dict[tuple[str, str, str], list[str]],
    sample_key: tuple[str, str, str],
    match_id: str,
) -> None:
    if sample_key not in mapping:
        mapping[sample_key] = [match_id]
    else:
        mapping[sample_key].append(match_id)


def write_pair_match_anndatas(
    truth_df: pd.DataFrame,
    dataset_paths: dict[str, str | PathLike[str]],
    output_dir: Path,
    output_prefix: str,
    cache_cell_types: bool = True,
    verbose: bool = False,
) -> list[Path]:
    required_columns = {
        "query_dataset",
        "db_dataset",
        "query_cell_type",
        "query_obs_id",
        "gt_db_obs_id",
    }
    missing_columns = sorted(required_columns - set(truth_df.columns))
    if missing_columns:
        raise KeyError(
            "Truth dataframe is missing required columns for pair-match AnnData export: "
            f"{missing_columns}"
        )
    if truth_df.empty:
        return []

    stores = {
        name: DatasetStore(dataset_name=name, dataset_path=Path(path), cache_enabled=cache_cell_types)
        for name, path in dataset_paths.items()
    }
    cache: dict[tuple[str, str], object] = {}

    pair_dir = output_dir / output_filename(output_prefix, "pair_matches")
    pair_dir.mkdir(parents=True, exist_ok=True)

    local_truth = truth_df[
        ["query_dataset", "db_dataset", "query_cell_type", "query_obs_id", "gt_db_obs_id"]
    ].copy()
    query_ds = local_truth["query_dataset"].astype(str).to_numpy()
    db_ds = local_truth["db_dataset"].astype(str).to_numpy()
    local_truth["_pair_a"] = np.where(query_ds <= db_ds, query_ds, db_ds)
    local_truth["_pair_b"] = np.where(query_ds <= db_ds, db_ds, query_ds)

    pair_keys = (
        local_truth[["_pair_a", "_pair_b"]]
        .drop_duplicates()
        .sort_values(["_pair_a", "_pair_b"])
        .itertuples(index=False, name=None)
    )

    output_paths: list[Path] = []
    for dataset_a, dataset_b in pair_keys:
        pair_truth = local_truth.loc[
            (local_truth["_pair_a"] == dataset_a) & (local_truth["_pair_b"] == dataset_b),
            ["query_dataset", "db_dataset", "query_cell_type", "query_obs_id", "gt_db_obs_id"],
        ].copy()
        if pair_truth.empty:
            continue
        if dataset_a not in stores or dataset_b not in stores:
            raise KeyError(
                f"Cannot build pair-match AnnData for {dataset_a}/{dataset_b}: "
                "dataset path is missing."
            )

        pair_truth = pair_truth.sort_values(
            ["query_dataset", "db_dataset", "query_cell_type", "query_obs_id", "gt_db_obs_id"]
        ).reset_index(drop=True)
        pair_truth["direction_idx"] = (
            pair_truth.groupby(["query_dataset", "db_dataset"]).cumcount() + 1
        )
        pair_truth["match_id"] = (
            pair_truth["query_dataset"].astype(str)
            + "_to_"
            + pair_truth["db_dataset"].astype(str)
            + "_"
            + pair_truth["direction_idx"].astype(str).str.zfill(6)
        )

        forward_col = f"{dataset_a}_{dataset_b}_match_id"
        reverse_col = f"{dataset_b}_{dataset_a}_match_id"
        pair_match_columns = [forward_col]
        if reverse_col != forward_col:
            pair_match_columns.append(reverse_col)

        match_ids_by_col: dict[str, dict[tuple[str, str, str], list[str]]] = {
            column: {} for column in pair_match_columns
        }
        sample_keys: set[tuple[str, str, str]] = set()
        for row in pair_truth.itertuples(index=False):
            direction_col = f"{row.query_dataset}_{row.db_dataset}_match_id"
            direction_map = match_ids_by_col.setdefault(direction_col, {})
            query_key = (str(row.query_dataset), str(row.query_cell_type), str(row.query_obs_id))
            db_key = (str(row.db_dataset), str(row.query_cell_type), str(row.gt_db_obs_id))
            _append_match_id(direction_map, query_key, str(row.match_id))
            _append_match_id(direction_map, db_key, str(row.match_id))
            sample_keys.add(query_key)
            sample_keys.add(db_key)

        pair_shared_genes: Optional[np.ndarray] = None
        for cell_type in sorted(pair_truth["query_cell_type"].astype(str).unique()):
            left_data = load_cell_type_data(stores[dataset_a], cell_type, cache)
            right_data = load_cell_type_data(stores[dataset_b], cell_type, cache)
            if left_data is None or right_data is None:
                continue
            shared_genes = np.intersect1d(
                left_data.adata.var_names.values,
                right_data.adata.var_names.values,
                assume_unique=False,
            )
            if shared_genes.size == 0:
                continue
            if pair_shared_genes is None:
                pair_shared_genes = shared_genes
            else:
                pair_shared_genes = np.intersect1d(
                    pair_shared_genes,
                    shared_genes,
                    assume_unique=False,
                )
            if pair_shared_genes.size == 0:
                break

        if pair_shared_genes is None or pair_shared_genes.size == 0:
            _log(
                verbose,
                f"[precompute] pair-match {dataset_a}<->{dataset_b}: "
                "no shared genes after intersection; skipping",
            )
            continue

        grouped_keys: dict[tuple[str, str], list[str]] = {}
        for dataset_name, cell_type, obs_id in sorted(sample_keys):
            grouped_keys.setdefault((dataset_name, cell_type), []).append(obs_id)

        blocks: list[ad.AnnData] = []
        common_layers: Optional[set[str]] = None
        for (dataset_name, cell_type), obs_ids in grouped_keys.items():
            if dataset_name not in stores:
                continue
            cell_data = load_cell_type_data(stores[dataset_name], cell_type, cache)
            if cell_data is None or cell_data.adata.n_obs == 0:
                continue

            var_idx = cell_data.adata.var_names.get_indexer(pair_shared_genes)
            if np.any(var_idx < 0):
                continue

            # Some group_rep h5ads have duplicate obs_names; keep every matching row.
            obs_names_str = np.asarray(cell_data.adata.obs_names.astype(str))
            wanted_names = {str(obs_id) for obs_id in obs_ids}
            selected_rows = np.flatnonzero(np.isin(obs_names_str, list(wanted_names)))
            if selected_rows.size == 0:
                continue

            selected_obs_ids = obs_names_str[selected_rows].tolist()
            block = cell_data.adata[selected_rows, var_idx].copy()

            block.obs = block.obs.copy()
            block.obs["dataset_name"] = dataset_name
            block.obs["source_cell_type"] = cell_type
            block.obs["original_obs_id"] = selected_obs_ids

            sample_keys_for_rows = [
                (dataset_name, cell_type, obs_id) for obs_id in selected_obs_ids
            ]
            for match_column in pair_match_columns:
                col_map = match_ids_by_col.get(match_column, {})
                block.obs[match_column] = [
                    ",".join(col_map.get(sample_key, [])) for sample_key in sample_keys_for_rows
                ]

            # Disambiguate duplicate obs_names with the source row index.
            block.obs_names = pd.Index(
                [
                    f"{dataset_name}::{cell_type}::{obs_id}::r{int(row)}"
                    for obs_id, row in zip(selected_obs_ids, selected_rows)
                ],
                dtype=object,
            )
            block.raw = None
            for key in list(block.uns.keys()):
                del block.uns[key]
            for key in list(block.obsm.keys()):
                del block.obsm[key]
            for key in list(block.varm.keys()):
                del block.varm[key]
            for key in list(block.obsp.keys()):
                del block.obsp[key]
            for key in list(block.varp.keys()):
                del block.varp[key]

            layer_keys = set(block.layers.keys())
            if common_layers is None:
                common_layers = layer_keys
            else:
                common_layers &= layer_keys
            blocks.append(block)

        if not blocks:
            _log(
                verbose,
                f"[precompute] pair-match {dataset_a}<->{dataset_b}: "
                "no sample blocks after filtering; skipping",
            )
            continue

        if common_layers is not None:
            for block in blocks:
                for layer_name in list(block.layers.keys()):
                    if layer_name not in common_layers:
                        del block.layers[layer_name]

        pair_adata = ad.concat(
            blocks,
            axis=0,
            join="inner",
            merge="first",
            uns_merge="first",
        )
        pair_adata.uns["datasets"] = [dataset_a, dataset_b]
        pair_adata.uns["match_id_columns"] = pair_match_columns
        pair_adata.uns["n_truth_matches"] = int(len(pair_truth))
        pair_adata.uns["n_unique_samples"] = int(pair_adata.n_obs)

        pair_path = pair_dir / f"{dataset_a}__{dataset_b}_matches.h5ad"
        pair_adata.write_h5ad(pair_path)
        output_paths.append(pair_path)
        _log(
            verbose,
            f"[precompute] pair-match {dataset_a}<->{dataset_b}: "
            f"samples={pair_adata.n_obs} genes={pair_adata.n_vars} -> {pair_path}",
        )

    return output_paths


def load_task_row(task_file: Path, task_id: int) -> pd.Series:
    tasks_df = pd.read_csv(task_file)
    if tasks_df.empty:
        raise ValueError(f"Task file has no rows: {task_file}")
    selected = tasks_df.loc[tasks_df["task_id"] == int(task_id)]
    if selected.empty:
        raise KeyError(f"Task id {task_id} not found in {task_file}")
    if len(selected) > 1:
        raise ValueError(f"Task id {task_id} appears more than once in {task_file}")
    return selected.iloc[0]


def run_single_task(
    dataset_paths: dict[str, str | PathLike[str]],
    task_file: Path,
    task_id: int,
    include_metrics: frozenset[str],
    task_output_dir: Path,
    output_prefix: str = "cross_dataset_retrieval_parallel",
    cell_type_filter: Optional[set[str]] = None,
    cache_cell_types: bool = True,
    verbose: bool = False,
) -> Path:
    task = load_task_row(task_file, task_id)
    query_dataset = str(task["query_dataset"])
    db_dataset = str(task["db_dataset"])
    representation = str(task["representation"])
    query_cell_type = _task_query_cell_type(task)

    effective_cell_type_filter = cell_type_filter
    if query_cell_type:
        if effective_cell_type_filter is None:
            effective_cell_type_filter = {query_cell_type}
        elif query_cell_type in effective_cell_type_filter:
            effective_cell_type_filter = {query_cell_type}
        else:
            effective_cell_type_filter = set()

    detail_file = ""
    if "task_detail_file" in task.index and not pd.isna(task["task_detail_file"]):
        detail_file = str(task["task_detail_file"]).strip()
    if not detail_file:
        representation_slug = str(task.get("representation_slug", slugify_representation(representation)))
        detail_file = task_detail_filename(
            output_prefix=output_prefix,
            task_id=task_id,
            query_dataset=query_dataset,
            db_dataset=db_dataset,
            representation_slug=representation_slug,
            query_cell_type=query_cell_type,
        )

    _log(
        verbose,
        f"[task {task_id}] start pair={query_dataset}->{db_dataset} rep={representation}",
    )
    t0 = time.perf_counter()
    settings = RetrievalSettings(
        include_metrics=include_metrics,
        include_representations=frozenset({representation}),
    )
    detail_df, _, _ = run_cross_dataset_retrieval(
        dataset_paths=dataset_paths,
        query_datasets=[query_dataset],
        db_datasets=[db_dataset],
        settings=settings,
        include_self_dataset=True,
        cell_type_filter=effective_cell_type_filter,
        cache_cell_types=cache_cell_types,
        verbose=verbose,
    )
    if detail_df.empty:
        detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)

    task_output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = task_output_dir / detail_file
    detail_df.to_csv(detail_path, index=False)

    _log(
        verbose,
        f"[task {task_id}] done rows={len(detail_df)} output={detail_path} "
        f"elapsed_s={time.perf_counter() - t0:.1f}",
    )
    return detail_path


def merge_task_outputs(
    task_file: Path,
    task_output_dir: Path,
    output_dir: Path,
    output_prefix: str = "cross_dataset_retrieval_parallel",
    strict_missing: bool = False,
    verbose: bool = False,
) -> tuple[Path, Path, Path, list[str]]:
    tasks_df = pd.read_csv(task_file)
    detail_frames: list[pd.DataFrame] = []
    missing_files: list[str] = []

    for _, task in tasks_df.iterrows():
        detail_file = ""
        if "task_detail_file" in task.index and not pd.isna(task["task_detail_file"]):
            detail_file = str(task["task_detail_file"]).strip()
        query_cell_type = _task_query_cell_type(task)
        if not detail_file:
            detail_file = task_detail_filename(
                output_prefix=output_prefix,
                task_id=int(task["task_id"]),
                query_dataset=str(task["query_dataset"]),
                db_dataset=str(task["db_dataset"]),
                representation_slug=str(
                    task.get(
                        "representation_slug",
                        slugify_representation(str(task["representation"])),
                    )
                ),
                query_cell_type=query_cell_type,
            )
        detail_path = task_output_dir / detail_file
        if not detail_path.exists():
            missing_files.append(str(detail_path))
            continue
        frame = pd.read_csv(detail_path)
        if frame.empty:
            continue
        detail_frames.append(frame)

    if missing_files:
        _log(verbose, f"[merge] missing task outputs={len(missing_files)}")
        for missing_path in missing_files[:10]:
            _log(verbose, f"[merge] missing: {missing_path}")
        if strict_missing:
            raise FileNotFoundError(
                "Missing task outputs and --strict-missing was set. "
                f"First missing file: {missing_files[0]}"
            )

    if detail_frames:
        detail_df = pd.concat(detail_frames, ignore_index=True)
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
    else:
        detail_df = pd.DataFrame(columns=DETAIL_COLUMNS)

    if detail_df.empty:
        summary_by_cell_df = pd.DataFrame(columns=SUMMARY_BY_CELL_COLUMNS)
        summary_overall_df = pd.DataFrame(columns=SUMMARY_OVERALL_COLUMNS)
    else:
        summary_by_cell_df, summary_overall_df = summarize_retrieval(detail_df)
        if summary_by_cell_df.empty:
            summary_by_cell_df = pd.DataFrame(columns=SUMMARY_BY_CELL_COLUMNS)
        if summary_overall_df.empty:
            summary_overall_df = pd.DataFrame(columns=SUMMARY_OVERALL_COLUMNS)

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / output_filename(output_prefix, "detail.csv")
    summary_cell_path = output_dir / output_filename(output_prefix, "summary_by_cell_type.csv")
    summary_overall_path = output_dir / output_filename(output_prefix, "summary_overall.csv")

    detail_df.to_csv(detail_path, index=False)
    summary_by_cell_df.to_csv(summary_cell_path, index=False)
    summary_overall_df.to_csv(summary_overall_path, index=False)

    _log(
        verbose,
        f"[merge] done detail_rows={len(detail_df)} "
        f"summary_by_cell_rows={len(summary_by_cell_df)} "
        f"summary_overall_rows={len(summary_overall_df)}",
    )
    return detail_path, summary_cell_path, summary_overall_path, missing_files
