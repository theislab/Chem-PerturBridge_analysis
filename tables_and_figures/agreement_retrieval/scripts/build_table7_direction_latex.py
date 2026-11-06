#!/usr/bin/env python3
"""Render Table 7 using its two saved input sources.

Observed, centroid and centroid-delta values come from the saved notebook CI
TSV. Individual-peer values come from the reviewer workbook, historical sheet "Table 5 Direction". These sources have distinct historical bootstrap intervals and
are intentionally kept separate. Inputs are under tables_and_figures/agreement_retrieval/data/table7.
Reading the workbook requires openpyxl; --tolerance checks observed means.

  python tables_and_figures/agreement_retrieval/scripts/build_table7_direction_latex.py
"""

from __future__ import annotations

import argparse
import re
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
from table_common import DATASET_DISPLAY, PAIR_ORDER  # noqa: E402

DEFAULT_CI = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table7" / "direction_cluster_bca_ci.tsv"
DEFAULT_WORKBOOK = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table7" / "reviewer_manuscript_tables.xlsx"
DEFAULT_OUT = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "results" / "default" / "table7" / "table_7_direction_combined.tex"

# Workbook sheet names retain the original numbering.
SHEET = "Table 5 Direction"
NORMALIZATION = "raw"
SUMMARY_LEVEL = "dataset_pair"
# The workbook names pairs by display label. Those labels contain hyphens, so
# the en dash is what makes the split unambiguous.
PAIR_SEPARATOR = "\u2013"

CI_METRICS = {
    "observed": "observed_direction_agreement_p05",
    "centroid": "baseline_pair_direction_agreement_p05",
    "delta_vs_centroid": "delta_vs_baseline_pair_direction_agreement_p05",
}
WORKBOOK_COLUMNS = {
    "peer": "Individual-peer mean [95% CI]",
    "delta_vs_peer": "Observed minus peer [95% CI]",
    "pct": "Corrected percentile [95% CI]",
}

# LaTeX header -> key in the per-pair estimate dict.
ESTIMATES = [
    ("Observed", "observed"),
    ("Centroid", "centroid"),
    ("Source peer", "source_peer"),
    ("Target peer", "target_peer"),
    (r"$\Delta$ centroid", "delta_vs_centroid"),
    (r"$\Delta$ source", "delta_vs_source_peer"),
    (r"$\Delta$ target", "delta_vs_target_peer"),
    (r"Pct.\ source", "pct_source"),
    (r"Pct.\ target", "pct_target"),
]

# A peer score is discarded wherever the observed score is missing, and direction
# agreement gates both peer sides on the same observed value, so the two peer
# panels are the observed panel. One column, labelled for all three.
N_SUB_HEADERS = ["obs / peers", "cent"]
# The two counts mean different things -- one panel set against another -- so
# they get more air between them than Table 6 gives its four related counts.
COUNT_INNER_GAP_PT = 10
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


CI_PATTERN = re.compile(
    r"^\s*([+-]?\d*\.?\d+)\s*\[\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\]\s*$"
)


def parse_ci(text: object, where: str) -> tuple[float, float, float]:
    """Recover (mean, lo, hi) from a rendered ``0.923 [0.892, 0.949]`` cell."""
    match = CI_PATTERN.match(str(text))
    if not match:
        raise SystemExit(f"cannot parse {text!r} from {where}")
    return tuple(float(group) for group in match.groups())  # type: ignore[return-value]


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_ci(path: Path) -> dict[tuple[str, str], dict]:
    """Observed, centroid and delta-vs-centroid, exactly as published."""
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["summary_level"] == SUMMARY_LEVEL]
    print(f"[read] {len(frame)} {SUMMARY_LEVEL} rows <- {relative(path)}")

    out: dict[tuple[str, str], dict] = {}
    for a, b in PAIR_ORDER:
        pair = frame[(frame["dataset_a"] == a) & (frame["dataset_b"] == b)]
        record: dict = {}
        for key, metric in CI_METRICS.items():
            match = pair[pair["metric"] == metric]
            if len(match) != 1:
                raise SystemExit(
                    f"expected one {metric} row for {a}-{b}, got {len(match)}"
                )
            row = match.iloc[0]
            record[key] = (float(row["mean"]), float(row["ci_low"]), float(row["ci_high"]))
            record[f"n_{key}"] = int(row["n_compounds"])
        out[(a, b)] = record
    return out


def load_peers(path: Path) -> dict[tuple[str, str], dict]:
    """Peer means, deltas and percentiles from the workbook's direction sheet."""
    try:
        sheet = pd.read_excel(path, sheet_name=SHEET)
    except ImportError as error:
        raise SystemExit(
            f"reading {SHEET} needs openpyxl ({error}). Run this with an interpreter "
            "that has it, e.g. /home/icb/olga.novitskaia/deg_venv/bin/python."
        )
    sheet = sheet[sheet["Normalization"] == NORMALIZATION]
    if sheet.empty:
        raise SystemExit(f"no {NORMALIZATION!r} rows in {SHEET}")
    print(f"[read] {len(sheet)} rows <- {relative(path)} [{SHEET}]")

    reverse = {label: name for name, label in DATASET_DISPLAY.items()}
    out: dict[tuple[str, str], dict] = {}
    for label, group in sheet.groupby("Dataset pair"):
        parts = str(label).split(PAIR_SEPARATOR)
        unknown = [part for part in parts if part not in reverse]
        if len(parts) != 2 or unknown:
            raise SystemExit(f"cannot map {label!r} in {SHEET} (unknown: {unknown})")
        key = (reverse[parts[0]], reverse[parts[1]])

        record: dict = {}
        counts = set()
        for side in ("source", "target"):
            rows = group[group["Peer side"] == side]
            if len(rows) != 1:
                raise SystemExit(f"expected one {side} row for {label}, got {len(rows)}")
            row = rows.iloc[0]
            where = f"{SHEET} / {label} / {side}"
            record[f"{side}_peer"] = parse_ci(row[WORKBOOK_COLUMNS["peer"]], where)
            record[f"delta_vs_{side}_peer"] = parse_ci(
                row[WORKBOOK_COLUMNS["delta_vs_peer"]], where
            )
            record[f"pct_{side}"] = parse_ci(row[WORKBOOK_COLUMNS["pct"]], where)
            record["observed_workbook"] = parse_ci(row["Observed [95% CI]"], where)
            counts.add(int(row["n compounds"]))
        if len(counts) != 1:
            raise SystemExit(
                f"{label}: source and target peer panels cover different cluster "
                f"counts {sorted(counts)}; the table's single count column assumes "
                "they agree."
            )
        record["n_peer"] = counts.pop()
        out[key] = record
    return out


def merge(ci: dict, peers: dict, tolerance: float) -> dict[tuple[str, str], dict]:
    """Join the two sources, checking they describe the same observed panel."""
    worst = 0.0
    offender = None
    for key in PAIR_ORDER:
        if key not in peers:
            raise SystemExit(f"{SHEET} has no rows for {key[0]}-{key[1]}")
        record = ci[key]
        peer = peers[key]

        gap = abs(record["observed"][0] - peer["observed_workbook"][0])
        if gap > worst:
            worst, offender = gap, key
        if peer["n_peer"] != record["n_observed"]:
            raise SystemExit(
                f"{key[0]}-{key[1]}: workbook peer panel covers {peer['n_peer']} "
                f"clusters against {record['n_observed']} observed; "
                "the shared count column requires equal panel sizes."
            )
        record.update({k: v for k, v in peer.items() if k != "observed_workbook"})

    if worst > tolerance:
        raise SystemExit(
            f"workbook and notebook disagree on observed direction agreement by "
            f"{worst:.3e} at {offender[0]}-{offender[1]} (tolerance {tolerance:.0e})."
        )
    print(f"[check] workbook vs notebook observed agrees, max |diff| = {worst:.2e}")
    print("[check] peer panels cover the same clusters as observed in all pairs")
    return ci


def build_tabular(data: dict) -> str:
    # Direction agreement is bounded by 0 and 1, so an estimate cell never holds
    # more than a 16-character ``[-0.204, -0.005]``, 44.4pt at \tiny. Matching
    # Table 6's widths keeps the two tables the same size on the page.
    colspec = (
        r"@{}>{\raggedright\arraybackslash}p{2.75cm}"
        + r"@{\hspace{1.5pt}}c"
        + rf"@{{\hspace{{{COUNT_INNER_GAP_PT}pt}}}}c" * (len(N_SUB_HEADERS) - 1)
        + rf"@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}"
        + rf"*{{{len(ESTIMATES)}}}{{>{{\raggedright\arraybackslash}}p{{1.62cm}}}}@{{}}"
    )

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

    lines = [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        top,
        N_GROUP_RULE,
        bottom,
        r"\midrule",
    ]

    for a, b in PAIR_ORDER:
        record = data[(a, b)]
        cells = [
            pair_label(a, b),
            str(record["n_observed"]),
            str(record["n_centroid"]),
            *(stacked_cell(*record[key], False) for _, key in ESTIMATES),
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
            r"\label{tab:cross_dataset_direction_agreement_bootstrap_ci}",
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
    parser.add_argument("--ci", type=Path, default=DEFAULT_CI)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="allowed gap between workbook and notebook observed means; the "
        "workbook rounds to three decimals, so it cannot be tighter than 5e-4",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.ci, args.workbook):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    data = merge(load_ci(args.ci), load_peers(args.workbook), args.tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(wrap_table(build_tabular(data)), encoding="utf-8")
    print(f"[write] {relative(args.out)}")
    print(f"[info] {len(PAIR_ORDER)} pairs x {len(ESTIMATES)} estimates")


if __name__ == "__main__":
    main()
