#!/usr/bin/env python3
"""Precompute reusable per-gene population-standardization statistics.

This command writes compact ``.npz`` sufficient-statistic caches only.  It never
materializes standardized ``.h5ad`` copies.

Examples
--------
Explicit line files::

    uv run python scripts/precompute_population_zscore.py \
      --source cigs_mce=CVCL_0023=/data/cigs_mce/CVCL_0023_de.h5ad \
      --source cigs_mce=CVCL_1055=/data/cigs_mce/CVCL_1055_de.h5ad

Discover all direct ``.h5ad`` children of dataset directories::

    uv run python scripts/precompute_population_zscore.py \
      --dataset-dir cigs_mce=/data/cigs_mce/group_rep/results \
      --dataset-dir cigs_tcm=/data/cigs_tcm/group_rep/results \
      --row-chunk-size 1024 \
      --workers 2

``--workers`` parallelizes independent line files only. Dataset-wide statistics are
still pooled in canonical cell-type order, so serial and parallel runs are numerically
identical.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.population_zscore import (  # noqa: E402
    DEFAULT_ROW_CHUNK_SIZE,
    discover_dataset_population_sources,
    infer_line_cell_type,
    load_or_fit_dataset_population_stats_from_source_stats,
    load_or_fit_population_stats,
    stats_qc_record,
)


SCOPE_BOTH = "both"
SCOPE_DATASET_CELL_TYPE = "dataset-cell-type"
SCOPE_DATASET = "dataset"


def _fit_line_task(
    task: tuple[str, str, Path, Path, str, int, bool],
):
    (
        dataset_name,
        cell_type,
        source_path,
        cache_root,
        layer_name,
        row_chunk_size,
        force,
    ) = task
    return load_or_fit_population_stats(
        source_path=source_path,
        dataset_name=dataset_name,
        cell_type=cell_type,
        cache_root=cache_root,
        layer_name=layer_name,
        row_chunk_size=row_chunk_size,
        force=force,
        verbose=True,
    )


def configured_dataset_dirs(repo_root: Path) -> list[tuple[str, Path]]:
    """Return the line-level source directories used by the three notebooks."""
    data_root = Path(repo_root) / "data" / "theislab_temp"
    special = {
        "cigs_mce": (
            data_root
            / "cigs_mce"
            / "group_rep_extracted"
            / "deg_data"
            / "group_rep"
            / "full"
            / "qc_false"
            / "filter_min_cells_0"
            / "results"
        ),
        "cigs_tcm": (
            data_root
            / "cigs_tcm"
            / "group_rep_extracted"
            / "deg_data"
            / "group_rep"
            / "full"
            / "qc_false"
            / "filter_min_cells_0"
            / "results"
        ),
    }
    dataset_names = (
        "l1000_phase1",
        "l1000_phase2",
        "tahoe",
        "cigs_mce",
        "novartis_batch_2500",
        "vcpi_0001",
        "cigs_tcm",
        "vcpi_0002",
        "gdpx2",
        "sciplex",
        "dilimap_train_val",
        "op3",
    )
    return [
        (
            dataset_name,
            special.get(dataset_name, data_root / dataset_name / "group_rep"),
        )
        for dataset_name in dataset_names
        if special.get(
            dataset_name,
            data_root / dataset_name / "group_rep",
        ).is_dir()
    ]


def _split_spec(value: str, *, expected_parts: int, label: str) -> list[str]:
    parts = value.split("=", maxsplit=expected_parts - 1)
    if len(parts) != expected_parts or any(not part.strip() for part in parts):
        raise argparse.ArgumentTypeError(
            f"{label} must contain {expected_parts} non-empty '='-separated fields"
        )
    return [part.strip() for part in parts]


def parse_source_spec(value: str) -> tuple[str, str, Path]:
    dataset_name, cell_type, path = _split_spec(
        value,
        expected_parts=3,
        label="--source DATASET=CELL_TYPE=PATH",
    )
    return dataset_name, cell_type, Path(path)


def parse_dataset_dir_spec(value: str) -> tuple[str, Path]:
    dataset_name, path = _split_spec(
        value,
        expected_parts=2,
        label="--dataset-dir DATASET=PATH",
    )
    return dataset_name, Path(path)


def infer_cell_type(path: Path) -> str:
    """Backward-compatible wrapper for existing callers and tests."""
    return infer_line_cell_type(path)


def collect_sources(
    *,
    explicit_sources: Sequence[tuple[str, str, Path]],
    dataset_dirs: Sequence[tuple[str, Path]],
    recursive: bool,
) -> dict[str, dict[str, Path]]:
    sources: dict[str, dict[str, Path]] = {}

    def add(dataset_name: str, cell_type: str, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Population source does not exist: {path}")
        existing = sources.setdefault(dataset_name, {}).get(cell_type)
        if existing is not None and existing.resolve() != path.resolve():
            raise ValueError(
                f"Conflicting sources for {dataset_name}/{cell_type}: "
                f"{existing} and {path}"
            )
        sources[dataset_name][cell_type] = path

    for dataset_name, cell_type, path in explicit_sources:
        add(dataset_name, cell_type, path)

    for dataset_name, directory in dataset_dirs:
        for cell_type, path in discover_dataset_population_sources(
            directory,
            recursive=recursive,
        ).items():
            add(dataset_name, cell_type, path)
    if not sources:
        raise ValueError("Provide at least one --source or --dataset-dir")
    return {
        dataset_name: dict(sorted(line_sources.items()))
        for dataset_name, line_sources in sorted(sources.items())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute dataset×cell-type and dataset-wide per-gene population "
            "z-score statistics without writing standardized H5AD files."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        type=parse_source_spec,
        metavar="DATASET=CELL_TYPE=PATH",
        help="Add one explicit line-level H5AD source (repeatable).",
    )
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        type=parse_dataset_dir_spec,
        metavar="DATASET=PATH",
        help="Discover line-level H5AD files in a dataset directory (repeatable).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover H5AD files below each --dataset-dir.",
    )
    parser.add_argument(
        "--all-configured",
        action="store_true",
        help=(
            "Discover every currently available dataset source directory used by "
            "the three cross-source notebooks."
        ),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPO_ROOT / "results" / "population_zscore_stats",
    )
    parser.add_argument(
        "--scope",
        choices=(SCOPE_BOTH, SCOPE_DATASET_CELL_TYPE, SCOPE_DATASET),
        default=SCOPE_BOTH,
    )
    parser.add_argument("--layer-name", default="logFC")
    parser.add_argument(
        "--row-chunk-size",
        type=int,
        default=DEFAULT_ROW_CHUNK_SIZE,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Independent line files to process concurrently. The default of 1 "
            "preserves serial behavior; try 2 or 4 on high-memory instances."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute the requested caches even when fingerprints match.",
    )
    parser.add_argument(
        "--qc-output",
        type=Path,
        help="Optionally save one TSV row per generated or reloaded cache.",
    )
    return parser


def run(args: argparse.Namespace) -> pd.DataFrame:
    if args.row_chunk_size < 1:
        raise ValueError("--row-chunk-size must be positive")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    dataset_dirs = list(args.dataset_dir)
    if args.all_configured:
        dataset_dirs.extend(configured_dataset_dirs(REPO_ROOT))
    sources = collect_sources(
        explicit_sources=args.source,
        dataset_dirs=dataset_dirs,
        recursive=args.recursive,
    )
    n_lines = sum(len(line_sources) for line_sources in sources.values())
    print(
        f"[z_score_precompute] {len(sources):,} datasets, {n_lines:,} line files; "
        f"scope={args.scope}; workers={args.workers}; "
        f"cache_root={args.cache_root}",
        flush=True,
    )

    qc_records: list[dict[str, object]] = []
    for dataset_index, (dataset_name, line_sources) in enumerate(
        sources.items(),
        start=1,
    ):
        print(
            f"[z_score_precompute] dataset {dataset_index}/{len(sources)}: "
            f"{dataset_name} ({len(line_sources)} lines)",
            flush=True,
        )
        line_tasks = [
            (
                dataset_name,
                cell_type,
                source_path,
                args.cache_root,
                args.layer_name,
                args.row_chunk_size,
                args.force,
            )
            for cell_type, source_path in line_sources.items()
        ]
        line_stats_by_cell = {}
        line_started_at = time.monotonic()
        if args.workers == 1 or len(line_tasks) == 1:
            for line_index, task in enumerate(line_tasks, start=1):
                cell_type = task[1]
                print(
                    f"[z_score_precompute]   line {line_index}/{len(line_sources)}: "
                    f"{cell_type}",
                    flush=True,
                )
                stats = _fit_line_task(task)
                line_stats_by_cell[cell_type] = stats
        else:
            worker_count = min(args.workers, len(line_tasks))
            print(
                f"[z_score_precompute]   processing {len(line_tasks)} lines with "
                f"{worker_count} workers",
                flush=True,
            )
            context = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=context,
            ) as executor:
                future_to_cell = {
                    executor.submit(_fit_line_task, task): task[1]
                    for task in line_tasks
                }
                for completed_count, future in enumerate(
                    as_completed(future_to_cell),
                    start=1,
                ):
                    cell_type = future_to_cell[future]
                    stats = future.result()
                    line_stats_by_cell[cell_type] = stats
                    elapsed = time.monotonic() - line_started_at
                    print(
                        f"[z_score_precompute]   completed "
                        f"{completed_count}/{len(line_tasks)}: {cell_type} "
                        f"(elapsed {elapsed / 60.0:.1f}m)",
                        flush=True,
                    )

        ordered_line_stats = [
            line_stats_by_cell[cell_type]
            for cell_type in sorted(line_stats_by_cell)
        ]
        if args.scope in {SCOPE_BOTH, SCOPE_DATASET_CELL_TYPE}:
            qc_records.extend(
                stats_qc_record(stats)
                for stats in ordered_line_stats
            )

        if args.scope in {SCOPE_BOTH, SCOPE_DATASET}:
            dataset_stats = load_or_fit_dataset_population_stats_from_source_stats(
                source_stats=ordered_line_stats,
                dataset_name=dataset_name,
                cache_root=args.cache_root,
                force=args.force,
                verbose=True,
            )
            qc_records.append(stats_qc_record(dataset_stats))

        print(
            f"[z_score_precompute] dataset {dataset_name} complete in "
            f"{(time.monotonic() - line_started_at) / 60.0:.1f}m",
            flush=True,
        )

    qc = pd.DataFrame(qc_records)
    if args.qc_output is not None:
        args.qc_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.qc_output.with_name(
            f".{args.qc_output.name}.tmp-{os.getpid()}"
        )
        try:
            qc.to_csv(temporary, sep="\t", index=False)
            os.replace(temporary, args.qc_output)
        finally:
            if temporary.exists():
                temporary.unlink()
        print(f"[z_score_precompute] saved QC to {args.qc_output}", flush=True)
    print("[z_score_precompute] complete", flush=True)
    return qc


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
