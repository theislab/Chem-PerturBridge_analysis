#!/usr/bin/env python3
"""Precompute adaptive-shrinkage logFC layers into sidecar files.

Reads the read-only differential expression h5ads, recovers the moderated
standard error as ``|logFC / t|``, and writes shrunken logFC under two scopes:
one prior per cell line, and one prior per processing batch within a cell line.
Nothing is written back to the source h5ads; the layers land in a separate
sidecar tree that the retrieval loader attaches in memory.

The shrinkage itself runs in R, because ``ashr`` has no Python port. One
subprocess is spawned per prior group.

Run one cell line at a time, or shard the file list across a Slurm array with
``--shard`` and ``--n-shards``.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from chem_perturbridge_analysis.retrieval.ashr import (  # noqa: E402
    BATCH_LAYER,
    CELL_LINE_LAYER,
    sidecar_path,
    standard_errors_from_moderated_t,
    write_sidecar,
)

DEFAULT_RSCRIPT = Path("/home/icb/olga.novitskaia/deg_venv/lib/R/bin/Rscript")
DEFAULT_SHRINK_SCRIPT = REPO_ROOT / "scripts" / "ashr_shrink.R"
DATA_ROOT = Path("/lustre/groups/ml01/workspace/olga.novitskaia/data_updated")

DEFAULT_DATASETS: dict[str, str] = {
    "cigs_mce": "cigs_mce/deg_data/group_rep/full/qc_false/filter_min_cells_0/results",
    "cigs_tcm": "cigs_tcm/deg_data/group_rep/full/qc_false/filter_min_cells_0/results",
    "l1000_phase1": "l1000_phase1/deg_data/group_rep/full/qc_false/filter_min_cells_0/results",
    "l1000_phase2": "l1000_phase2/deg_data/group_rep/full/qc_false/filter_min_cells_0/results",
    "sciplex": "sciplex/deg_data/group_rep/full/qc_false/filter_min_cells_10/results",
    "tahoe": "tahoe/deg_data/group_rep/full/qc_false/filter_min_cells_50/results",
}

LOGFC_LAYER = "logFC"
T_LAYER = "t"
BATCH_OBS_COLUMN = "processing_batch"


def read_string_column(handle: h5py.File, group: str, key: str) -> np.ndarray:
    node = handle[group][key]
    if isinstance(node, h5py.Group):
        categories = node["categories"][:]
        codes = node["codes"][:]
        decoded = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in categories])
        return decoded[codes]
    raw = node[:]
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in raw])


def read_index(handle: h5py.File, group: str) -> np.ndarray:
    index_key = handle[group].attrs.get("_index", "_index")
    if isinstance(index_key, bytes):
        index_key = index_key.decode()
    return read_string_column(handle, group, str(index_key))


def run_ashr(
    betahat: np.ndarray,
    se: np.ndarray,
    rscript: Path,
    shrink_script: Path,
    subsample: int,
    seed: int,
    chunk: int,
    fit_pool: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> np.ndarray:
    """Shrink one group of estimates through a single fitted prior."""
    n = int(betahat.size)
    with tempfile.TemporaryDirectory(prefix="ashr_") as tmpdir:
        tmp = Path(tmpdir)
        in_path, out_path = tmp / "in.bin", tmp / "out.bin"
        with open(in_path, "wb") as handle:
            handle.write(np.ascontiguousarray(betahat, dtype="<f8").tobytes())
            handle.write(np.ascontiguousarray(se, dtype="<f8").tobytes())

        command = [
            str(rscript), str(shrink_script),
            "--input", str(in_path),
            "--output", str(out_path),
            "--n", str(n),
            "--subsample", str(subsample),
            "--seed", str(seed),
            "--chunk", str(chunk),
        ]
        if fit_pool is not None:
            fit_beta, fit_se = fit_pool
            fit_path = tmp / "fit.bin"
            with open(fit_path, "wb") as handle:
                handle.write(np.ascontiguousarray(fit_beta, dtype="<f8").tobytes())
                handle.write(np.ascontiguousarray(fit_se, dtype="<f8").tobytes())
            command += ["--fit-input", str(fit_path), "--fit-n", str(int(fit_beta.size))]

        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ashr failed (exit {result.returncode}):\n{result.stderr}")
        posterior = np.fromfile(out_path, dtype="<f8")

    if posterior.size != n:
        raise RuntimeError(f"ashr returned {posterior.size} values, expected {n}")
    return posterior


def batch_intersection_genes(finite: np.ndarray, batches: np.ndarray) -> np.ndarray:
    """Genes with at least one finite estimate in every batch."""
    keep = np.ones(finite.shape[1], dtype=bool)
    for batch in np.unique(batches):
        keep &= finite[batches == batch].any(axis=0)
    return keep


def shrink_cell_line(
    logfc: np.ndarray,
    se: np.ndarray,
    finite: np.ndarray,
    batches: np.ndarray,
    prior_genes: str,
    **ashr_kwargs,
) -> np.ndarray:
    out = np.full(logfc.shape, np.nan, dtype=np.float32)
    if not finite.any():
        return out

    fit_pool = None
    if prior_genes == "intersection":
        gene_mask = batch_intersection_genes(finite, batches)
        pool_mask = finite & gene_mask[np.newaxis, :]
        if pool_mask.any():
            fit_pool = (logfc[pool_mask], se[pool_mask])

    out[finite] = run_ashr(logfc[finite], se[finite], fit_pool=fit_pool, **ashr_kwargs)
    return out


def shrink_per_batch(
    logfc: np.ndarray,
    se: np.ndarray,
    finite: np.ndarray,
    batches: np.ndarray,
    **ashr_kwargs,
) -> np.ndarray:
    out = np.full(logfc.shape, np.nan, dtype=np.float32)
    for batch in np.unique(batches):
        rows = batches == batch
        mask = finite & rows[:, np.newaxis]
        if not mask.any():
            continue
        out[mask] = run_ashr(logfc[mask], se[mask], **ashr_kwargs)
    return out


def process_cell_line(
    de_path: Path,
    dataset_name: str,
    cell_type: str,
    output_root: Path,
    prior_genes: str,
    verbose: bool,
    **ashr_kwargs,
) -> None:
    started = time.time()
    with h5py.File(de_path, "r") as handle:
        logfc = np.asarray(handle["layers"][LOGFC_LAYER][:], dtype=np.float64)
        t_stat = np.asarray(handle["layers"][T_LAYER][:], dtype=np.float64)
        obs_names = read_index(handle, "obs")
        var_names = read_index(handle, "var")
        batches = read_string_column(handle, "obs", BATCH_OBS_COLUMN)

    se = standard_errors_from_moderated_t(logfc, t_stat)
    finite = np.isfinite(logfc) & np.isfinite(se)
    n_batches = int(np.unique(batches).size)
    if verbose:
        print(
            f"  {dataset_name}:{cell_type} rows={logfc.shape[0]} genes={logfc.shape[1]} "
            f"batches={n_batches} finite={int(finite.sum())}",
            flush=True,
        )

    cell_line = shrink_cell_line(
        logfc, se, finite, batches, prior_genes=prior_genes, **ashr_kwargs
    )
    # A single-batch cell line makes the two scopes the same computation, so
    # record an alias instead of shrinking and storing the same values twice.
    if n_batches == 1:
        layers = {CELL_LINE_LAYER: cell_line}
        aliases = {BATCH_LAYER: CELL_LINE_LAYER}
    else:
        layers = {
            CELL_LINE_LAYER: cell_line,
            BATCH_LAYER: shrink_per_batch(logfc, se, finite, batches, **ashr_kwargs),
        }
        aliases = None

    write_sidecar(
        sidecar_path(output_root, dataset_name, cell_type),
        layers=layers,
        aliases=aliases,
        obs_names=obs_names,
        var_names=var_names,
        provenance={
            "source_file": str(de_path),
            "dataset": dataset_name,
            "cell_type": cell_type,
            "n_batches": n_batches,
            "se_definition": "abs(logFC / t)  [moderated]",
            "prior_genes": prior_genes,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    if verbose:
        print(f"    done in {time.time() - started:.1f}s", flush=True)


def collect_cell_lines(datasets: dict[str, Path]) -> list[tuple[str, str, Path]]:
    found = []
    for name, root in sorted(datasets.items()):
        for path in sorted(root.glob("*_de.h5ad")):
            found.append((name, path.name[: -len("_de.h5ad")], path))
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override or add a dataset directory. Defaults to the six retrieval datasets.",
    )
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names, or 'all'.")
    parser.add_argument("--cell-types", default=None, help="Optional comma-separated filter.")
    parser.add_argument("--prior-genes", choices=("all", "intersection"), default="all",
                        help="Cell-line prior fitted on all finite estimates, or only on genes "
                             "tested in every batch. No effect on the per-batch scope.")
    parser.add_argument("--rscript", type=Path, default=DEFAULT_RSCRIPT)
    parser.add_argument("--shrink-script", type=Path, default=DEFAULT_SHRINK_SCRIPT)
    parser.add_argument("--subsample", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=2_000_000)
    parser.add_argument("--shard", type=int, default=0, help="0-based shard index for array jobs.")
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets = {name: DATA_ROOT / sub for name, sub in DEFAULT_DATASETS.items()}
    for override in args.dataset:
        name, _, path = override.partition("=")
        datasets[name] = Path(path)
    if args.datasets.strip().lower() != "all":
        wanted = {n.strip() for n in args.datasets.split(",") if n.strip()}
        datasets = {n: p for n, p in datasets.items() if n in wanted}

    missing = sorted(name for name, path in datasets.items() if not path.is_dir())
    if missing:
        raise FileNotFoundError(f"Dataset directories not found: {missing}")

    targets = collect_cell_lines(datasets)
    if args.cell_types:
        wanted = {c.strip() for c in args.cell_types.split(",") if c.strip()}
        targets = [t for t in targets if t[1] in wanted]
    targets = targets[args.shard :: args.n_shards]

    if not args.overwrite:
        targets = [
            t for t in targets if not sidecar_path(args.output_root, t[0], t[1]).exists()
        ]

    print(f"ashr precompute: {len(targets)} cell lines -> {args.output_root}", flush=True)
    ashr_kwargs = dict(
        rscript=args.rscript,
        shrink_script=args.shrink_script,
        subsample=args.subsample,
        seed=args.seed,
        chunk=args.chunk,
    )
    for dataset_name, cell_type, de_path in targets:
        process_cell_line(
            de_path=de_path,
            dataset_name=dataset_name,
            cell_type=cell_type,
            output_root=args.output_root,
            prior_genes=args.prior_genes,
            verbose=args.verbose,
            **ashr_kwargs,
        )
    print("done", flush=True)


if __name__ == "__main__":
    main()
