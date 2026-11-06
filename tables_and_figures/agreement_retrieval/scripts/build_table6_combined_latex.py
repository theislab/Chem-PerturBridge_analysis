#!/usr/bin/env python3
"""Render Table 6 from the saved raw/z-score and moderated-t panels.

Inputs are under tables_and_figures/agreement_retrieval/data/table6. Means, CI bounds, counts,
normalization labels and row order are preserved from the original panels.

  python tables_and_figures/agreement_retrieval/scripts/build_table6_combined_latex.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(SCRIPT_DIR))

from table_common import PAIR_ORDER  # noqa: E402

DEFAULT_RAW_ZSCORE = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table6" / "table_6_raw_zscore.tsv"
DEFAULT_T = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table6" / "table_6_moderated_t.tsv"
DEFAULT_OUT = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "results" / "default" / "table6" / "table_6_deg_spearman_combined.tex"

# Display label, then the raw/z-score Normalization value (None = the moderated-t file).
REPRESENTATIONS = [
    ("raw logFC", "raw", "raw_zscore"),
    (r"dataset $z$-score", "dataset z-score", "raw_zscore"),
    (r"dataset $\times$ cell-line $z$-score", "dataset x cell line z-score", "raw_zscore"),
    (r"moderated $t$", None, "t"),
]

# LaTeX header -> (mean, lo, hi, signed)
ESTIMATES = [
    (r"Observed", "observed_mean", "observed_ci_low", "observed_ci_high", False),
    (
        r"Centroid",
        "source_centroid_mean",
        "source_centroid_ci_low",
        "source_centroid_ci_high",
        False,
    ),
    (
        r"Source peer",
        "source_peer_mean",
        "source_peer_ci_low",
        "source_peer_ci_high",
        False,
    ),
    (
        r"Target peer",
        "target_peer_mean",
        "target_peer_ci_low",
        "target_peer_ci_high",
        False,
    ),
    (
        r"$\Delta$ centroid",
        "delta_vs_centroid_mean",
        "delta_vs_centroid_ci_low",
        "delta_vs_centroid_ci_high",
        True,
    ),
    (
        r"$\Delta$ source",
        "delta_vs_source_peer_mean",
        "delta_vs_source_peer_ci_low",
        "delta_vs_source_peer_ci_high",
        True,
    ),
    (
        r"$\Delta$ target",
        "delta_vs_target_peer_mean",
        "delta_vs_target_peer_ci_low",
        "delta_vs_target_peer_ci_high",
        True,
    ),
    (
        r"Pct.\ source",
        "pct_source_mean",
        "pct_source_ci_low",
        "pct_source_ci_high",
        False,
    ),
    (
        r"Pct.\ target",
        "pct_target_mean",
        "pct_target_ci_low",
        "pct_target_ci_high",
        False,
    ),
]


# Two-tier header for the count block. A per-column $n^{\mathrm{cent}}$ is
# 17.3pt at \tiny against 6.8pt for a two-digit count, so the superscripts,
# not the intercolumn gap, set the width. Hoisting the n into a spanning
# header lets each column shrink to its digits.
#
# The gutter before the estimates lives in the template of the last count
# column, so anything spanning columns 2-5 -- the \multicolumn and the
# \cmidrule alike -- covers it too and drifts right by up to its full width.
# Repeating it in the \multicolumn spec and adding it to the rule's right trim
# pulls both back over the four columns.
COUNT_GUTTER_PT = 14
CMIDRULE_TRIM_PT = 2
N_GROUP_HEADER = (
    rf"\multicolumn{{4}}{{c@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}}}"
    r"{$n_{\mathrm{cmpd}}$}"
)
N_GROUP_RULE = (
    rf"\cmidrule(l{{{CMIDRULE_TRIM_PT}pt}}"
    rf"r{{{COUNT_GUTTER_PT + CMIDRULE_TRIM_PT}pt}}){{2-5}}"
)
N_SUB_HEADERS = ["obs", "cent", "src", "tgt"]


def pair_label(dataset_a: str, dataset_b: str) -> str:
    """En-dash join, matching the template's CIGS-MCE--CIGS-TCM."""
    from table_common import DATASET_DISPLAY

    return f"{DATASET_DISPLAY[dataset_a]}--{DATASET_DISPLAY[dataset_b]}"


def fmt_point(mean: float) -> str:
    return f"{mean:.3f}"


def fmt_interval(lo: float, hi: float) -> str:
    return f"[{lo:.3f}, {hi:.3f}]"


def stacked_cell(mean: float, lo: float, hi: float, signed: bool) -> str:
    """OP3-style cell: point on the first line, gray interval under ``\\newline``.

    Works inside ``tabularx`` ``X``/``p`` columns, which is why the OP3 table
    can use ``\\newline`` rather than a nested tabular or makecell.
    """
    del signed
    point = fmt_point(mean)
    interval = fmt_interval(lo, hi)
    # Left-aligned nested tabular so the point and the interval share an edge.
    # A centered inner column makes the shorter point sit in the middle of
    # the longer ``[lo, hi]`` string.
    return (
        r"\begin{tabular}[t]{@{}l@{}}"
        f"{point}"
        r"\\"
        rf"\textcolor{{gray}}{{{interval}}}"
        r"\end{tabular}"
    )


def load_raw_zscore(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "dataset_a",
        "dataset_b",
        "Normalization",
        "n_compounds_observed",
        "n_compounds_centroid",
        "n_compounds_source_peer",
        "n_compounds_target_peer",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} lacks {sorted(missing)}")
    return frame


def load_t(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    required = {
        "dataset_a",
        "dataset_b",
        "n_compounds_observed",
        "n_compounds_centroid",
        "n_compounds_source_peer",
        "n_compounds_target_peer",
        "delta_vs_source_centroid_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} lacks {sorted(missing)}")
    # Align the raw/z-score name for the source-centroid delta.
    frame = frame.rename(
        columns={
            "delta_vs_source_centroid_mean": "delta_vs_centroid_mean",
            "delta_vs_source_centroid_ci_low": "delta_vs_centroid_ci_low",
            "delta_vs_source_centroid_ci_high": "delta_vs_centroid_ci_high",
        }
    )
    return frame


def select_row(raw_zscore: pd.DataFrame, t: pd.DataFrame, source: str, normalization, a: str, b: str) -> pd.Series:
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
            f"expected one row for {a}-{b} / {normalization or 'moderated t'}, got {len(match)}"
        )
    return match.iloc[0]


def estimate_cells(row: pd.Series) -> list[str]:
    cells = []
    for _header, mean_c, lo_c, hi_c, signed in ESTIMATES:
        cells.append(
            stacked_cell(float(row[mean_c]), float(row[lo_c]), float(row[hi_c]), signed)
        )
    return cells


def header_rows() -> list[str]:
    """Two header rows: names over the count group, units over the CI line."""
    top = (
        "Dataset pair & "
        + N_GROUP_HEADER
        + " & "
        + " & ".join(name for name, *_ in ESTIMATES)
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
    n_cols = 1 + 4 + len(ESTIMATES)
    # Every value is below 1 in magnitude, so the widest an estimate column
    # ever holds is a 16-character ``[-0.674, -0.260]``, 44.4pt at \tiny;
    # 1.62cm leaves a small buffer. Because the table is scaled to \textwidth,
    # slack here costs rendered font size, so the fit is deliberately tight.
    # Count columns stay natural-width and centered; estimate columns are
    # left-aligned so each point lines up with its interval.
    colspec = (
        r"@{}>{\raggedright\arraybackslash}p{2.75cm}"
        r"@{\hspace{1.5pt}}c@{\hspace{1.5pt}}c@{\hspace{1.5pt}}c@{\hspace{1.5pt}}c"
        rf"@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}"
        r"*{9}{>{\raggedright\arraybackslash}p{1.62cm}}@{}"
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
            cells = [
                pair_label(a, b),
                str(int(row["n_compounds_observed"])),
                str(int(row["n_compounds_centroid"])),
                str(int(row["n_compounds_source_peer"])),
                str(int(row["n_compounds_target_peer"])),
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
            r"\label{tab:cross_dataset_deg_lfc_spearman_bootstrap_ci}",
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
    print(
        f"[info] {len(REPRESENTATIONS)} representations x {len(PAIR_ORDER)} pairs "
        f"= {len(REPRESENTATIONS) * len(PAIR_ORDER)} rows"
    )


if __name__ == "__main__":
    main()
