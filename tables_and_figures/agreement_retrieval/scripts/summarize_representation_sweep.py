#!/usr/bin/env python3
"""Rank one representation sweep and write its manuscript tables.

Reads summary_overall.csv and truth_summary.csv from a single run directory.
Use --build-inputs to rebuild those files from the three saved retrieval runs first.
Writes representation_ranking.tsv there and three LaTeX tables separately.
Rankings retain the original best-metric selection, unweighted directed-pair
means, and core / minimum-support / all-pairs scopes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "representation_sweep"

PAIR_KEYS = ["query_dataset", "db_dataset"]


SCORE_COLUMN = "retrieval_score_mean"


REFERENCE_REPRESENTATION = "logFC"


CORE_DATASETS = ("l1000_phase1", "l1000_phase2", "sciplex", "tahoe")


MIN_CONTEXT_SHARED_DRUGS = 10


REPRESENTATION_DISPLAY = {
    "t": r"moderated $t$",
    "B": r"$B$ statistic",
    "P.Value": r"$p$ value",
    "logFC": r"$\mathrm{logFC}$",
    "sign(logFC)": r"$\mathrm{sign}(\mathrm{logFC})$",
    "signed_-log10(p)*sign(logFC)": (
        r"$-\log_{10}(p)\cdot\mathrm{sign}(\mathrm{logFC})$"
    ),
    "signed_-log10(adj.p)*sign(logFC)": (
        r"$-\log_{10}(p_{\mathrm{adj}})\cdot\mathrm{sign}(\mathrm{logFC})$"
    ),
    "z=sign(logFC)*Phi^-1(1-p/2)": (
        r"$z=\mathrm{sign}(\mathrm{logFC})\cdot\Phi^{-1}(1-p/2)$"
    ),
    "per_gene_population_zscore(logFC)": (
        r"per-gene population $z$-score$(\mathrm{logFC})$, per cell line"
    ),
    "per_gene_population_zscore_dataset": (
        r"per-gene population $z$-score$(\mathrm{logFC})$, per dataset"
    ),
    "ashr_posterior_mean(logFC)_cell_line": (
        r"ashr posterior mean$(\mathrm{logFC})$, per cell line"
    ),
    "ashr_posterior_mean(logFC)_batch": (
        r"ashr posterior mean$(\mathrm{logFC})$, per batch"
    ),
}


METRIC_DISPLAY = {"cosine": "cosine", "pearson": "Pearson", "mrrmse": "MRRMSE"}


def check_grain(summary: pd.DataFrame, source: str) -> None:
    grain = summary.groupby(PAIR_KEYS + ["representation", "metric"]).size()
    if (grain != 1).any():
        raise ValueError(
            f"{source} is not unique per (pair, representation, metric); "
            f"found up to {int(grain.max())} rows per combination"
        )


def load_summary(results_dir: Path) -> pd.DataFrame:
    """Load the per-pair retrieval summary and check its grain."""
    path = results_dir / "summary_overall.csv"
    summary = pd.read_csv(path)
    check_grain(summary, str(path))
    if summary.empty or REFERENCE_REPRESENTATION not in set(summary.representation):
        raise ValueError(f"{path} must contain {REFERENCE_REPRESENTATION} reference rows")
    return summary


def supported_pairs(results_dir: Path) -> set[tuple[str, str]]:
    """Directed pairs whose dataset combination clears the shared-drug floor.

    The floor is evaluated per unordered dataset combination on its best cell
    type, matching how the overlap_group_rep notebooks build contexts.
    """
    truth = pd.read_csv(results_dir / "truth_summary.csv")
    truth = truth.assign(
        combo=[tuple(sorted(pair)) for pair in zip(truth.query_dataset, truth.db_dataset)]
    )
    best = truth.groupby("combo")["n_shared_pubchem_cids"].max()
    kept = set(best[best >= MIN_CONTEXT_SHARED_DRUGS].index)
    return {(a, b) for combo in kept for a, b in (combo, combo[::-1])}


def rank_representations(summary: pd.DataFrame, scope: str) -> pd.DataFrame:
    """One row per representation, at its best metric, ranked by mean score.

    ``mean_retrieval_score`` is an unweighted mean over directed pairs, so a
    pair with two queries counts as much as one with tens of thousands. The
    metric is chosen on the same data the score is reported from, which makes
    every score here mildly optimistic.
    """
    n_pairs = summary.groupby(PAIR_KEYS).ngroups
    per_config = summary.groupby(["representation", "metric"], as_index=False)[SCORE_COLUMN].mean()
    best = (
        per_config.sort_values(SCORE_COLUMN, ascending=False)
        .groupby("representation", as_index=False)
        .first()
        .sort_values(SCORE_COLUMN, ascending=False)
    )

    reference_metric = best.loc[best.representation == REFERENCE_REPRESENTATION, "metric"].iloc[0]
    reference = pair_scores(summary, REFERENCE_REPRESENTATION, reference_metric)

    rows = []
    for rank, row in enumerate(best.itertuples(), start=1):
        scores = pair_scores(summary, row.representation, row.metric)
        shared = scores.index.intersection(reference.index)
        is_reference = row.representation == REFERENCE_REPRESENTATION
        rows.append(
            {
                "scope": scope,
                "n_pairs": n_pairs,
                "rank": rank,
                "representation": row.representation,
                "best_metric": row.metric,
                # Not rounded here: round() returns a float, so a value like 0.6670
                # loses its trailing zero and writes as "0.667". Formatting happens on
                # write instead, which keeps every score the same width.
                "mean_retrieval_score": getattr(row, SCORE_COLUMN),
                "beats_logFC": pd.NA if is_reference else int((scores[shared] > reference[shared]).sum()),
                "n_pairs_compared": pd.NA if is_reference else len(shared),
                "logFC_best_metric": reference_metric,
            }
        )
    return pd.DataFrame(rows)


def pair_scores(summary: pd.DataFrame, representation: str, metric: str) -> pd.Series:
    """Per-pair retrieval score for one representation/metric combination."""
    selected = summary[(summary.representation == representation) & (summary.metric == metric)]
    return selected.set_index(PAIR_KEYS)[SCORE_COLUMN]


def build_scopes(summary: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    core = summary[
        summary.query_dataset.isin(CORE_DATASETS) & summary.db_dataset.isin(CORE_DATASETS)
    ]
    keep = supported_pairs(results_dir)
    min_support = summary[summary.set_index(PAIR_KEYS).index.isin(keep)]
    return pd.concat(
        [
            rank_representations(core, "core"),
            rank_representations(min_support, "min_support"),
            rank_representations(summary, "all"),
        ],
        ignore_index=True,
    )


def render_representation(name: str) -> str:
    try:
        return REPRESENTATION_DISPLAY[name]
    except KeyError:
        raise SystemExit(
            f"No LaTeX rendering for representation {name!r}. Add it to "
            "REPRESENTATION_DISPLAY -- emitting it raw would break the build."
        ) from None


def build_rows(block: pd.DataFrame) -> list[str]:
    rows = []
    for row in block.itertuples():
        beats = row.beats_logFC
        if pd.isna(beats):
            # logFC is the reference, so it has no comparison against itself.
            # The column is right-aligned so the counts line up on their "/ n";
            # multicolumn overrides that for this one cell to centre it.
            beats_cell = r"\multicolumn{1}{c}{n/a}"
        else:
            beats_cell = f"{int(beats)} / {int(row.n_pairs_compared)}"
        rows.append(
            f"{int(row.rank)} & {render_representation(row.representation)} & "
            f"{METRIC_DISPLAY[row.best_metric]} & "
            f"{float(row.mean_retrieval_score):.4f} & {beats_cell} \\\\"
        )
    return rows


def build_table(block: pd.DataFrame, scope: str) -> str:
    lines = [
        r"\begin{table*}[t]",
        r"\refstepcounter{table}",
        f"\\label{{tab:cross_dataset_retrieval_representation_ranking_{scope}}}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{rllrr}",
        r"\toprule",
        r"Rank & Representation & Best metric & Mean retrieval score & Beats logFC \\",
        r"\midrule",
        *build_rows(block),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--build-inputs", type=Path, nargs="?", const=REPO_ROOT / "results" / "representation_sweep",
        metavar="RUNS_DIR",
        help="Consolidate three saved runs before ranking (default source: results/representation_sweep).",
    )
    parser.add_argument(
        "--latex-dir", type=Path, default=REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "results" / "default" / "representation_sweep",
        help="Directory for the three manuscript tables (separate from results).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.build_inputs is not None:
        from build_representation_sweep_inputs import build_inputs

        build_inputs(args.build_inputs, args.results_dir)
    summary = load_summary(args.results_dir)
    ranking = build_scopes(summary, args.results_dir)
    output = args.results_dir / "representation_ranking.tsv"
    ranking.to_csv(output, sep="\t", index=False, float_format="%.4f")
    print(f"Wrote {output}")

    # Render the rounded TSV just as the original two-script pipeline did.
    ranking = pd.read_csv(output, sep="\t")
    args.latex_dir.mkdir(parents=True, exist_ok=True)
    for scope, block in ranking.groupby("scope", sort=False):
        table = build_table(block.sort_values("rank"), scope)
        path = args.latex_dir / f"representation_ranking_{scope}.tex"
        path.write_text(table)
        print(f"Wrote {path} ({len(block)} representations, "
              f"{int(block.n_pairs.iloc[0])} directed pairs)")


if __name__ == "__main__":
    main()
