#!/usr/bin/env python3
"""Consolidate the three saved retrieval runs into the representation-sweep inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "representation_sweep"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "representation_sweep"
PRIMARY_RUN = "cross_dataset_retrieval_parallel_cigs"
SUPPLEMENTAL_RUNS = (
    (
        "cross_dataset_retrieval_parallel_ashr",
        ("ashr_posterior_mean(logFC)_cell_line",),
    ),
    (
        "cross_dataset_retrieval_parallel_cigs_zscore",
        ("per_gene_population_zscore(logFC)", "per_gene_population_zscore_dataset"),
    ),
)
CONTROL_KEYS = ["query_dataset", "db_dataset", "metric"]
SUMMARY_KEYS = ["query_dataset", "db_dataset", "representation", "metric"]
SCORE_COLUMN = "retrieval_score_mean"


def read_summary(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = set([*SUMMARY_KEYS, SCORE_COLUMN]) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if frame.empty or frame[SUMMARY_KEYS].isna().any().any():
        raise ValueError(f"{path} has empty or missing summary keys")
    if frame.duplicated(SUMMARY_KEYS).any():
        raise ValueError(f"{path} is not unique per (pair, representation, metric)")
    return frame


def representation_scores(frame: pd.DataFrame, representation: str) -> pd.Series:
    return (
        frame.loc[frame.representation.eq(representation)]
        .set_index(CONTROL_KEYS)[SCORE_COLUMN]
        .sort_index()
    )


def consolidate_runs(input_dir: Path) -> pd.DataFrame:
    """Keep primary metrics, add ASHR cell-line means and replace both z-score scopes.

    Raw-logFC scores must match exactly across runs. Each selected representation
    must cover the primary pair/metric keys. Input row and column order is retained
    so rebuilding the saved inputs also preserves their CSV serialization.
    """
    input_dir = Path(input_dir)
    primary = read_summary(input_dir / PRIMARY_RUN / f"{PRIMARY_RUN}_summary_overall.csv")
    reference = representation_scores(primary, "logFC")
    if reference.empty:
        raise ValueError("The primary run must contain logFC controls")
    for representation in primary.representation.unique():
        if not representation_scores(primary, representation).index.equals(reference.index):
            raise ValueError(f"Primary {representation} does not cover the logFC pair/metric keys")

    summary = primary
    metrics = set(primary.metric)
    for run, representations in SUPPLEMENTAL_RUNS:
        extra = read_summary(input_dir / run / f"{run}_summary_overall.csv")
        if set(extra.columns) != set(primary.columns):
            raise ValueError(f"{run} has different summary columns from the primary run")
        extra = extra.loc[extra.metric.isin(metrics), primary.columns]
        control = representation_scores(extra, "logFC")
        try:
            pd.testing.assert_series_equal(reference, control, check_exact=True)
        except AssertionError as error:
            raise ValueError(f"{run} logFC controls differ from the primary run: {error}") from error
        for representation in representations:
            selected = representation_scores(extra, representation)
            if not selected.index.equals(reference.index):
                raise ValueError(
                    f"{run}: {representation} does not cover the primary pair/metric keys"
                )
        summary = pd.concat(
            [
                summary.loc[~summary.representation.isin(representations)],
                extra.loc[extra.representation.isin(representations)],
            ],
            ignore_index=True,
        )
    return summary


def build_inputs(input_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Validate all inputs before writing the consolidated summary and truth counts."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    summary = consolidate_runs(input_dir)
    source_truth = input_dir / PRIMARY_RUN / f"{PRIMARY_RUN}_truth_summary.csv"
    truth_bytes = source_truth.read_bytes()
    truth = pd.read_csv(source_truth)
    missing = {"query_dataset", "db_dataset", "n_shared_pubchem_cids"} - set(truth.columns)
    if missing:
        raise ValueError(f"{source_truth} is missing columns: {sorted(missing)}")
    summary_bytes = summary.to_csv(index=False).encode("utf-8")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary_overall.csv"
    truth_path = output_dir / "truth_summary.csv"
    summary_path.write_bytes(summary_bytes)
    truth_path.write_bytes(truth_bytes)
    print(
        f"Wrote {summary_path} ({len(summary)} rows, "
        f"{summary.representation.nunique()} representations)"
    )
    print(f"Wrote {truth_path}")
    return summary_path, truth_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Folder containing the three cross_dataset_retrieval_parallel_* run folders.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Destination for summary_overall.csv and truth_summary.csv (default: retained table data).",
    )
    args = parser.parse_args(argv)
    build_inputs(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
