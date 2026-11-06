#!/usr/bin/env python3
"""Render Table 8 from the saved raw/z-score and moderated-t panels.

Inputs are under tables_and_figures/agreement_retrieval/data/table8. The observed, centroid and
peer compound counts are checked before displaying a single count column.

  python tables_and_figures/agreement_retrieval/scripts/build_table8_combined_latex.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from build_table6_combined_latex import (  # noqa: E402
    CMIDRULE_TRIM_PT,
    COUNT_GUTTER_PT,
    pair_label,
    stacked_cell,
)
from table_common import PAIR_ORDER  # noqa: E402

DEFAULT_RAW_ZSCORE = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table8" / "table_8_raw_zscore.tsv"
DEFAULT_T = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table8" / "table_8_moderated_t.tsv"
DEFAULT_OUT = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "results" / "default" / "table8" / "table_8_allgene_spearman_combined.tex"

# Display label, then the raw/z-score Normalization value (None = the moderated-t file).
REPRESENTATIONS = [
    ("raw logFC", "raw", "raw_zscore"),
    (r"dataset $z$-score", "dataset z-score", "raw_zscore"),
    (r"dataset $\times$ cell-line $z$-score", "dataset x cell line z-score", "raw_zscore"),
    (r"moderated $t$", None, "t"),
]

# LaTeX header -> (mean, lo, hi), source before target throughout.
ESTIMATES = [
    (r"Observed", "observed"),
    (r"Centroid", "source_centroid"),
    (r"Source peer", "source_peer"),
    (r"Target peer", "target_peer"),
    (r"$\Delta$ centroid", "delta_vs_centroid"),
    (r"$\Delta$ source", "delta_vs_source_peer"),
    (r"$\Delta$ target", "delta_vs_target_peer"),
    (r"Pct.\ source", "pct_source"),
    (r"Pct.\ target", "pct_target"),
]

COUNT_COLUMNS = [
    "n_compounds_observed",
    "n_compounds_centroid",
    "n_compounds_source_peer",
    "n_compounds_target_peer",
]

# One count column, but headed the way Table 7 heads its shared one: the
# spanning $n_{\mathrm{cmpd}}$ over a sub-label naming the panels it covers,
# so a reader is told the four coincide rather than left to assume it.
N_SUB_HEADERS = ["obs / cent / peers"]
N_GROUP_HEADER = (
    rf"\multicolumn{{{len(N_SUB_HEADERS)}}}{{c@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}}}"
    r"{$n_{\mathrm{cmpd}}$}"
)
# The gutter before the estimates lives in the template of the last count
# column, so a rule spanning that group covers it too and reads as shifted
# right; adding it to the right trim pulls the rule back over the columns.
N_GROUP_RULE = (
    rf"\cmidrule(l{{{CMIDRULE_TRIM_PT}pt}}"
    rf"r{{{COUNT_GUTTER_PT + CMIDRULE_TRIM_PT}pt}}){{2-{1 + len(N_SUB_HEADERS)}}}"
)


def load_raw_zscore(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {"dataset_a", "dataset_b", "Normalization", *COUNT_COLUMNS}
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} lacks {sorted(missing)}")
    return frame


def load_t(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "dataset_a",
        "dataset_b",
        *COUNT_COLUMNS,
        "delta_vs_source_centroid_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} lacks {sorted(missing)}")
    # Align the raw/z-score name for the source-centroid delta.
    return frame.rename(
        columns={
            "delta_vs_source_centroid_mean": "delta_vs_centroid_mean",
            "delta_vs_source_centroid_ci_low": "delta_vs_centroid_ci_low",
            "delta_vs_source_centroid_ci_high": "delta_vs_centroid_ci_high",
        }
    )


def compound_count(row: pd.Series, where: str) -> int:
    """The one count, once the four panels are confirmed to agree."""
    counts = {int(row[column]) for column in COUNT_COLUMNS}
    if len(counts) != 1:
        detail = ", ".join(f"{c}={int(row[c])}" for c in COUNT_COLUMNS)
        raise SystemExit(
            f"{where}: the four panels no longer share a compound count ({detail}). "
            "Table 8 prints a single n_cmpd column on the assumption that they do; "
            "split it the way build_table6_combined_latex.py does."
        )
    return counts.pop()


def select_row(
    raw_zscore: pd.DataFrame, t: pd.DataFrame, source: str, normalization, a: str, b: str
) -> pd.Series:
    if source == "raw_zscore":
        match = raw_zscore[
            (raw_zscore["dataset_a"] == a)
            & (raw_zscore["dataset_b"] == b)
            & (raw_zscore["Normalization"] == normalization)
        ]
    else:
        match = t[(t["dataset_a"] == a) & (t["dataset_b"] == b)]
    if len(match) != 1:
        raise SystemExit(
            f"expected one row for {a}-{b} / {normalization or 'moderated t'}, "
            f"got {len(match)}"
        )
    return match.iloc[0]


def estimate_cells(row: pd.Series) -> list[str]:
    cells = []
    for _header, stem in ESTIMATES:
        cells.append(
            stacked_cell(
                float(row[f"{stem}_mean"]),
                float(row[f"{stem}_ci_low"]),
                float(row[f"{stem}_ci_high"]),
                False,
            )
        )
    return cells


def header_rows() -> list[str]:
    """Names over the count group, then the panel label and the CI line."""
    top = (
        "Dataset pair & "
        + N_GROUP_HEADER
        + " & "
        + " & ".join(name for name, _ in ESTIMATES)
        + r" \\"
    )
    bottom = (
        " & "
        + " & ".join(N_SUB_HEADERS)
        + " & "
        + " & ".join([r"\textcolor{gray}{95\% CI}"] * len(ESTIMATES))
        + r" \\"
    )
    return [top, N_GROUP_RULE, bottom]


def build_tabular(raw_zscore: pd.DataFrame, t: pd.DataFrame) -> str:
    n_cols = 1 + len(N_SUB_HEADERS) + len(ESTIMATES)
    # Every value is below 1 in magnitude, so the widest an estimate column ever
    # holds is a 16-character ``[-0.674, -0.260]``, 44.4pt at \tiny; 1.62cm
    # leaves a small buffer. Because the table is scaled to \textwidth, slack
    # here costs rendered font size, so the fit is deliberately tight. The count
    # column stays natural-width and centered; estimate columns are left-aligned
    # so each point lines up with its interval.
    colspec = (
        r"@{}>{\raggedright\arraybackslash}p{2.75cm}"
        + r"@{\hspace{1.5pt}}c" * len(N_SUB_HEADERS)
        + rf"@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}"
        + rf"*{{{len(ESTIMATES)}}}{{>{{\raggedright\arraybackslash}}p{{1.62cm}}}}@{{}}"
    )
    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        *header_rows(),
        r"\midrule",
    ]

    for i, (label, normalization, source) in enumerate(REPRESENTATIONS):
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{label}}}}} \\")
        lines.append(r"\addlinespace[2pt]")
        for a, b in PAIR_ORDER:
            row = select_row(raw_zscore, t, source, normalization, a, b)
            where = f"{a}-{b} / {normalization or 'moderated t'}"
            cells = [
                pair_label(a, b),
                str(compound_count(row, where)),
                *estimate_cells(row),
            ]
            lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\addlinespace[1pt]")
        if lines[-1] == r"\addlinespace[1pt]":
            lines.pop()

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def wrap_table(tabular: str) -> str:
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\refstepcounter{table}",
            r"\label{tab:cross_dataset_allgene_spearman_bootstrap_ci}",
            r"\centering",
            r"\tiny",
            r"\setlength{\tabcolsep}{2pt}",
            r"\renewcommand{\arraystretch}{0.95}",
            r"% Requires booktabs, graphicx, xcolor, array.",
            r"\resizebox{\textwidth}{!}{%",
            tabular,
            r"}",
            r"\end{table*}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--raw-zscore", type=Path, default=DEFAULT_RAW_ZSCORE)
    parser.add_argument("--moderated-t", type=Path, default=DEFAULT_T)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.raw_zscore, args.moderated_t):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    raw_zscore = load_raw_zscore(args.raw_zscore)
    t = load_t(args.moderated_t)
    tex = wrap_table(build_tabular(raw_zscore, t))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")
    print(f"[write] {args.out}")
    print("[check] all four panels share one compound count in every row")
    print(
        f"[info] {len(REPRESENTATIONS)} representations x {len(PAIR_ORDER)} pairs "
        f"= {len(REPRESENTATIONS) * len(PAIR_ORDER)} rows"
    )


if __name__ == "__main__":
    main()
