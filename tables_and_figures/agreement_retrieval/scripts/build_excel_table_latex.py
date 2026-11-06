#!/usr/bin/env python3
"""Base utilities for building updated replicate LaTeX tables from the reviewer
Excel file (reviewer_replicate_update.xlsx).

All three table scripts (Tables 10, 11, 14) share the same sheet layout:

  col 0  Representation
  col 1  Dataset
  col 2  Status
  col 3  n conditions
  col 4  n compounds
  col 5  Observed [95% CI]
  col 6  Centroid [95% CI]
  col 7  Observed minus centroid [95% CI]
  col 8  Individual-peer mean [95% CI]
  col 9  Observed minus peer [95% CI]
  col 10 Corrected percentile [95% CI]

Values are already formatted as "mean [lo, hi]" strings in the Excel file.

Requires: pandas, openpyxl
  pip install openpyxl          # or: uv pip install openpyxl
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

EXCEL_PATH = (
    REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "replicates" / "reviewer_replicate_update.xlsx"
)

# ── Column headers ────────────────────────────────────────────────────────────

COUNT_HEADERS = [
    r"$n_{\mathrm{cmpd}}$",
    r"$n_{\mathrm{cond}}$",
]

N_VALUE_COLS = 6
N_COLS = 1 + len(COUNT_HEADERS) + N_VALUE_COLS  # 9 total


def value_headers(metric: str) -> list[str]:
    """Value-column headers, named the way the paper replicate tables name them.

    ``metric`` is the noun those tables append to the score columns
    ("Spearman" in Tables 10 and 14, "direction" in Table 11). The two $\\Delta$
    columns follow the cross-dataset tables and name their baseline instead.
    """
    return [
        f"Observed {metric}",
        f"Centroid baseline {metric}",
        f"Peer mean {metric}",
        r"$\Delta$ centroid",
        r"$\Delta$ peer",
        "Percentile",
    ]


# ── Data loading ──────────────────────────────────────────────────────────────


def load_sheet(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one sheet and return only data rows (drop header/notes).

    Rows come back ordered by dataset name, case-insensitively, so "sci-Plex"
    sorts between "OP3" and "Tahoe-100M" rather than after the VCPI datasets.
    The sort is stable and keyed on the dataset alone, so filtering by
    representation afterwards still yields one contiguous, sorted block.
    """
    try:
        import openpyxl  # noqa: F401 – ensure dependency present
    except ImportError:
        raise SystemExit(
            "openpyxl is required to read the Excel file.\n"
            "Install it with:  pip install openpyxl\n"
            "or (uv):          uv pip install openpyxl"
        )

    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)

    # Keep only rows that have both a Representation and a Dataset value.
    # This drops the blank separator row, the "Notes" row, and any long-text
    # footnote rows that appear below the data.
    repr_col = df.columns[0]
    dataset_col = df.columns[1]
    df = df.dropna(subset=[repr_col, dataset_col]).copy()

    df.columns = [
        "repr",
        "dataset",
        "status",
        "n_cond",
        "n_cmpd",
        "observed",
        "centroid",
        "obs_centroid",
        "peer",
        "obs_peer",
        "percentile",
    ]
    order = df["dataset"].str.casefold().argsort(kind="stable")
    return df.iloc[order].reset_index(drop=True)


# ── LaTeX helpers ─────────────────────────────────────────────────────────────


def format_count(n: int | float) -> str:
    r"""Thousands separator as ``{,}`` so LaTeX does not add math-comma spacing."""
    return f"{int(n):,}".replace(",", "{,}")


_LEADING_PLUS = re.compile(r"\+(?=[.\d])")


def strip_plus(cell: str) -> str:
    """Drop explicit ``+`` signs; the cross-dataset tables sign only negatives."""
    return _LEADING_PLUS.sub("", cell)


def split_estimate(cell: str) -> tuple[str, str]:
    """Split ``mean [lo, hi]`` into the point and the bracketed interval."""
    point, bracket, rest = cell.partition("[")
    if not bracket:
        return cell.strip(), ""
    return point.strip(), "[" + rest.strip()


def format_estimate(cell: str) -> str:
    r"""Stack the point over its interval, the way the cross-dataset tables do.

    Point on the first line, 95\% interval in gray underneath.
    """
    point, interval = split_estimate(strip_plus(cell))
    if not interval:
        return point
    return (
        r"\begin{tabular}[t]{@{}l@{}}"
        + point
        + r"\\\textcolor{gray}{"
        + interval
        + r"}\end{tabular}"
    )


def header_rows(metric: str) -> list[str]:
    r"""Two-row header: labels, then a gray ``95\% CI`` marker per value column.

    The counts carry their own headers rather than sharing a spanning $n$, so
    $n_{\mathrm{cmpd}}$ and $n_{\mathrm{cond}}$ are each named in full. Nothing
    sits beneath them in the second row, so no \cmidrule is needed.
    """
    top = ["Dataset", *COUNT_HEADERS, *value_headers(metric)]
    sub = [
        "",
        *([""] * len(COUNT_HEADERS)),
        *([r"\textcolor{gray}{95\% CI}"] * N_VALUE_COLS),
    ]
    return [
        " & ".join(top) + r" \\",
        " & ".join(sub) + r" \\",
    ]


# Excel representation label -> block label as typeset in the paper tables.
# The replicate tables write plain "logFC"; the z-score wording comes from the
# cross-dataset tables (tables_and_figures/agreement_retrieval/results/default/table6 and table8).
DISPLAY_LABELS = {
    "raw logFC": "logFC",
    "raw logFC sign": "logFC sign",
    "z-score (dataset)": r"dataset $z$-score",
    "z-score (dataset x cell type)": r"dataset $\times$ cell-line $z$-score",
    "moderated t": r"moderated $t$",
}


def split_metric(label: str) -> tuple[str, str]:
    """Split an Excel label into its representation stem and metric suffix.

    Table 14 labels carry the metric ("raw logFC, Spearman"); Tables 10 and 11
    do not, and return an empty suffix.
    """
    stem, _, metric = label.partition(", ")
    return stem, metric


def _tex_repr(label: str, drop_metric: bool) -> str:
    """Typeset a representation label the way the paper tables do."""
    stem, metric = split_metric(label)
    display = DISPLAY_LABELS.get(stem, stem.replace("_", r"\_"))
    if metric and not drop_metric:
        display += f", {metric}"
    return display


def build_tabular(df: pd.DataFrame, representations: list[str], metric: str) -> str:
    # Column spec follows tables_and_figures/agreement_retrieval/results/default/table6 and table8: a wide ragged
    # dataset column, tight centred count columns, then equal-width ragged
    # value columns holding the stacked point/interval cells.
    col_spec = (
        r"@{}>{\raggedright\arraybackslash}p{2.75cm}"
        r"@{\hspace{1.5pt}}c@{\hspace{1.5pt}}c@{\hspace{14pt}}"
        rf"*{{{N_VALUE_COLS}}}{{>{{\raggedright\arraybackslash}}p{{2.1cm}}}}@{{}}"
    )
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        *header_rows(metric),
        r"\midrule",
    ]

    # A single block needs no heading, and when every block shares one metric
    # the header already names it, so the suffix is dropped from the labels.
    label_blocks = len(representations) > 1
    metrics = {split_metric(label)[1] for label in representations}
    drop_metric = len(metrics) == 1

    for pos, repr_label in enumerate(representations):
        if pos:
            lines.append(r"\midrule")
        if label_blocks:
            lines.append(
                rf"\multicolumn{{{N_COLS}}}{{@{{}}l@{{}}}}"
                rf"{{\textit{{{_tex_repr(repr_label, drop_metric)}}}}} \\"
            )
        lines.append(r"\addlinespace[2pt]")

        block = df[df["repr"] == repr_label]
        if block.empty:
            raise SystemExit(
                f"No rows found for representation {repr_label!r} "
                f"in the loaded data. "
                f"Available: {df['repr'].unique().tolist()}"
            )

        for position, (_, row) in enumerate(block.iterrows()):
            if position:
                lines.append(r"\addlinespace[1pt]")
            cells = [
                str(row["dataset"]),
                format_count(row["n_cmpd"]),
                format_count(row["n_cond"]),
                *(
                    format_estimate(str(row[key]))
                    # Peer mean sits before the two delta columns.
                    for key in (
                        "observed",
                        "centroid",
                        "peer",
                        "obs_centroid",
                        "obs_peer",
                        "percentile",
                    )
                ),
            ]
            lines.append(" & ".join(cells) + r" \\")

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def wrap_table(tabular: str, label: str) -> str:
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\refstepcounter{table}",
            rf"\label{{{label}}}",
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


# ── Argument parsing & render entry point ────────────────────────────────────


def parse_args(
    doc: str,
    default_out: Path,
    default_sheet: str,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=EXCEL_PATH,
        help=f"Path to the reviewer Excel file (default: {EXCEL_PATH})",
    )
    parser.add_argument(
        "--sheet",
        type=str,
        default=default_sheet,
        help="Sheet name to read from the Excel file",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output .tex file path",
    )
    return parser.parse_args()


def render(
    args: argparse.Namespace,
    representations: list[str],
    label: str,
    metric: str,
) -> None:
    """Build and write one combined LaTeX table from one Excel sheet."""
    if not args.excel.exists():
        raise SystemExit(f"Excel file not found: {args.excel}")

    df = load_sheet(args.excel, args.sheet)
    tabular = build_tabular(df, representations, metric)
    tex = wrap_table(tabular, label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(tex, encoding="utf-8")

    n_datasets = len(df[df["repr"] == representations[0]])
    try:
        display_out = args.out.resolve().relative_to(REPO_ROOT)
    except ValueError:
        display_out = args.out
    print(f"[write] {display_out}")
    print(
        f"[info]  {len(representations)} representation(s) × {n_datasets} datasets "
        f"= {len(representations) * n_datasets} rows"
    )
