#!/usr/bin/env python3
"""Render Tables 12 and 13 from saved strict-retrieval and individual-peer summaries.

Inputs are under tables_and_figures/agreement_retrieval/data/table12_13. Local strict panels provide
centroids and centroid deltas. Saved reseeded peer tables provide raw, sign
and moderated-t observed/peer values. The reviewer combined summary supplies
z-score observed/peer values. Keep this source split to preserve the tables.

Use --retrieval-ci PATH (repeatable) for current unified runner CI files.

  python tables_and_figures/agreement_retrieval/scripts/build_table12_13_retrieval_latex.py
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
from build_table7_direction_latex import parse_ci, relative  # noqa: E402
from table_common import PAIR_ORDER  # noqa: E402

RETRIEVAL_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data" / "table12_13"
DEFAULT_COMBINED = RETRIEVAL_DIR / "retrieval_combined_cluster_bca_ci.tsv"
DEFAULT_LOGFC_PEERS = RETRIEVAL_DIR / "table12_13_peer_baselines_sensitivity.tsv"
DEFAULT_T_PEERS = RETRIEVAL_DIR / "table12_13_peer_baselines_moderated_t_sensitivity.tsv"
DEFAULT_OUT_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "results" / "default" / "table12_13"
TABLE_NUMBERS = {"cosine": 12, "negative_l2": 13}

# Similarity key -> (display name, panel filename suffix, peer-file label).
SIMILARITIES = {
    "cosine": ("cosine", "cosine", "Cosine"),
    "negative_l2": ("negative $L_2$", "negative_l2", "L2"),
    "spearman": ("Spearman", "spearman", "Spearman"),
}

# Block label, then where its numbers come from. ``peers`` names a local peer
# summary and the Scale within it; ``scale`` selects rows in the reviewer
# summary instead. ``panel`` supplies the centroid, with all saved inputs in RETRIEVAL_DIR.
BLOCKS = [
    {
        "label": "raw logFC",
        "computation": "raw",
        "peers": ("logfc", "raw"),
        "panel": "retrieval_logfc_{}",
    },
    {
        "label": r"dataset $z$-score",
        "computation": "z-score-dataset",
        "scale": "per_gene_population_zscore_dataset",
        "panel": "retrieval_zscore_dataset_{}",
    },
    {
        "label": r"dataset $\times$ cell-line $z$-score",
        "computation": "z-score-dataset-cell-type",
        "scale": "per_gene_population_zscore_dataset_cell_type",
        "panel": "retrieval_zscore_dataset_cell_type_{}",
    },
    {
        "label": r"moderated $t$",
        "computation": "moderated-t",
        "peers": ("moderated_t", "moderated t"),
        "panel": "retrieval_moderated_t_{}",
    },
    {
        "label": r"sign(logFC)",
        "computation": "direction",
        "peers": ("logfc", "sign(logFC)"),
        "panel": "retrieval_logfc_direction_{}",
    },
]

# LaTeX header -> estimate key.
ESTIMATES = [
    (r"Observed", "observed"),
    (r"Centroid", "centroid"),
    (r"Source peer", "source_peer"),
    (r"Target peer", "target_peer"),
    (r"$\Delta$ centroid", "delta_vs_centroid"),
    (r"$\Delta$ source", "delta_vs_source_peer"),
    (r"$\Delta$ target", "delta_vs_target_peer"),
    (r"Pct.\ source", "pct_source"),
    (r"Pct.\ target", "pct_target"),
    # Last rather than beside Observed: it is observed-side only, since both peer summaries
    # carry a single "AUROC [95% CI]" column and the reviewer summary a single
    # observed_auroc, so it has no centroid or peer counterpart to line up with.
    (r"AUROC", "auroc"),
]

# Estimate key -> metric in the reviewer summary. No centroid here on purpose;
# see the module docstring.
COMBINED_METRICS = {
    "observed": "observed_normalized_best_positive_rank",
    "auroc": "observed_auroc",
    "source_peer": "source_peer_normalized_rank",
    "target_peer": "target_peer_normalized_rank",
    "delta_vs_source_peer": "delta_vs_source_peer_normalized_rank",
    "delta_vs_target_peer": "delta_vs_target_peer_normalized_rank",
    "pct_source": "source_individual_corrected_percentile",
    "pct_target": "target_individual_corrected_percentile",
}

NOTEBOOK_PEER_METRICS = {
    **COMBINED_METRICS,
    "source_peer": "source_individual_normalized_best_positive_rank",
    "target_peer": "target_individual_normalized_best_positive_rank",
    "delta_vs_source_peer": "delta_observed_vs_source_individual_rank",
    "delta_vs_target_peer": "delta_observed_vs_target_individual_rank",
}
STRICT_CI_METRICS = {
    "centroid": "baseline_normalized_best_positive_rank",
    "delta_vs_centroid": "delta_vs_baseline_normalized_best_positive_rank",
}

# Estimate key -> column in the local peer summaries.
PEER_COLUMNS = {
    "observed": "Observed rank [95% CI]",
    "auroc": "AUROC [95% CI]",
    "source_peer": "Source baseline rank [95% CI]",
    "target_peer": "Target baseline rank [95% CI]",
    "delta_vs_source_peer": "Delta source [95% CI]",
    "delta_vs_target_peer": "Delta target [95% CI]",
    "pct_source": "Source corrected percentile [95% CI]",
    "pct_target": "Target corrected percentile [95% CI]",
}

MISSING = "--"

N_SUB_HEADERS = [r"$n_{\mathrm{cmpd}}$", r"$n_{\mathrm{rows}}$"]
# The compound count is one number for the observed, centroid and peer panels
# alike, so the group header says so once rather than repeating it per column.
N_GROUP_HEADER = (
    rf"\multicolumn{{{len(N_SUB_HEADERS)}}}{{c@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}}}"
    r"{obs / cent / peers}"
)
# Two counts of different kinds, so they get more air than Table 6 gives its
# four related ones.
COUNT_INNER_GAP_PT = 10
N_GROUP_RULE = (
    rf"\cmidrule(l{{{CMIDRULE_TRIM_PT}pt}}"
    rf"r{{{COUNT_GUTTER_PT + CMIDRULE_TRIM_PT}pt}}){{2-{1 + len(N_SUB_HEADERS)}}}"
)


def load_combined(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame["metric"].isin(set(COMBINED_METRICS.values()))]
    print(f"[read] {len(frame)} metric rows <- {relative(path)}")
    return frame


def load_retrieval_ci(paths: list[Path]) -> pd.DataFrame:
    """Read one combined CI file or several per-computation runner outputs."""
    keys = ["computation", "similarity_metric", "dataset_a", "dataset_b", "family", "metric"]
    required = {*keys, "mean", "ci_low", "ci_high", "n_compounds", "n_rows"}
    frames = []
    for path in paths:
        frame = pd.read_csv(path, sep="\t")
        missing = sorted(required - set(frame.columns))
        if missing:
            raise SystemExit(f"{path}: missing unified retrieval CI columns {missing}")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated(keys).any():
        raise SystemExit("unified retrieval CI contains duplicate computation/pair/family/metric keys")
    return combined


def retrieval_ci_record(
    frame: pd.DataFrame, computation: str, similarity: str, a: str, b: str,
) -> dict:
    rows = frame.loc[
        frame["computation"].eq(computation) & frame["similarity_metric"].eq(similarity)
        & frame["dataset_a"].eq(a) & frame["dataset_b"].eq(b)
    ]
    where = f"{a}-{b} / {computation} / {similarity}"
    metrics = COMBINED_METRICS if computation.startswith("z-score") else NOTEBOOK_PEER_METRICS
    record = {}
    for family, mapping in (("peer", metrics), ("strict", STRICT_CI_METRICS)):
        for key, metric in mapping.items():
            matches = rows.loc[rows["family"].eq(family) & rows["metric"].eq(metric)]
            if len(matches) != 1:
                raise SystemExit(f"expected one {family}/{metric} row for {where}, got {len(matches)}")
            row = matches.iloc[0]
            record[key] = tuple(float(row[column]) for column in ("mean", "ci_low", "ci_high"))
            if key == "observed":
                record["n_cmpd"] = int(row["n_compounds"])
                record["n_rows"] = int(row["n_rows"])
    return record


def load_panel(similarity: str, stem: str, directory: Path) -> pd.DataFrame:
    suffix = SIMILARITIES[similarity][1]
    prefix = f"table_{TABLE_NUMBERS[similarity]}_" if similarity in TABLE_NUMBERS else ""
    path = directory / f"{prefix}{stem.format(suffix)}.tsv"
    if not path.exists():
        raise SystemExit(f"missing panel {path}")
    return pd.read_csv(path, sep="\t")


def load_peers(path: Path, scale: str, similarity: str) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t")
    label = SIMILARITIES[similarity][2]
    frame = frame[(frame["Scale"] == scale) & (frame["Similarity"] == label)]
    if len(frame) != len(PAIR_ORDER):
        raise SystemExit(
            f"{relative(path)}: expected {len(PAIR_ORDER)} rows for "
            f"{scale}/{label}, got {len(frame)}"
        )
    return frame


def as_int(value: object) -> int:
    """Counts arrive both as 11042 and as the rendered ``11,042``."""
    return int(str(value).replace(",", ""))


def fmt_count(value: int) -> str:
    """LaTeX thousands separator, so 21857 sets as 21,857 without a space."""
    return f"{value:,}".replace(",", "{,}")


def single_count(counts: set[int], where: str) -> int:
    if len(counts) != 1:
        raise SystemExit(
            f"{where}: panels cover different cluster counts {sorted(counts)}; "
            "the table's single count column assumes they agree."
        )
    return counts.pop()


def combined_record(
    combined: pd.DataFrame, scale: str, similarity: str, a: str, b: str
) -> dict:
    rows = combined[
        (combined["dataset_a"] == a)
        & (combined["dataset_b"] == b)
        & (combined["scale_variant"] == scale)
        & (combined["similarity_metric"] == similarity)
    ]
    where = f"{a}-{b} / {scale} / {similarity}"
    if rows.empty:
        raise SystemExit(f"reviewer summary has no rows for {where}")

    record: dict = {}
    counts = set()
    row_counts = set()
    for key, metric in COMBINED_METRICS.items():
        match = rows[rows["metric"] == metric]
        if len(match) != 1:
            raise SystemExit(f"expected one {metric} row for {where}, got {len(match)}")
        row = match.iloc[0]
        record[key] = (float(row["mean"]), float(row["ci_low"]), float(row["ci_high"]))
        counts.add(int(row["n_compounds"]))
        row_counts.add(int(row["n_rows"]))
    record["n_cmpd"] = single_count(counts, where)
    record["n_rows"] = single_count(row_counts, f"{where} (n_rows)")
    return record


def peer_record(peers: pd.DataFrame, a: str, b: str) -> dict:
    rows = peers[(peers["dataset_a"] == a) & (peers["dataset_b"] == b)]
    if len(rows) != 1:
        raise SystemExit(f"expected one peer row for {a}-{b}, got {len(rows)}")
    row = rows.iloc[0]
    where = f"{a}-{b}"
    record = {key: parse_ci(row[column], where) for key, column in PEER_COLUMNS.items()}
    record["n_cmpd"] = as_int(row["n_cmpd"])
    record["n_rows"] = as_int(row["n_rows"])
    return record


def add_centroid(record: dict, panel: pd.DataFrame, a: str, b: str) -> None:
    """Fold in the centroid and its delta from the local panel."""
    rows = panel[(panel["dataset_a"] == a) & (panel["dataset_b"] == b)]
    if len(rows) != 1:
        raise SystemExit(f"expected one panel row for {a}-{b}, got {len(rows)}")
    row = rows.iloc[0]

    record["centroid"] = (
        float(row["baseline_mean"]),
        float(row["baseline_ci_low"]),
        float(row["baseline_ci_high"]),
    )
    record["delta_vs_centroid"] = (
        float(row["delta_mean"]),
        float(row["delta_ci_low"]),
        float(row["delta_ci_high"]),
    )


def block_records(block: dict, similarity: str, sources: dict) -> dict:
    if "retrieval_ci" in sources:
        return {
            (a, b): retrieval_ci_record(sources["retrieval_ci"], block["computation"], similarity, a, b)
            for a, b in PAIR_ORDER
        }
    panel = (
        load_panel(similarity, block["panel"], RETRIEVAL_DIR)
        if block["panel"]
        else None
    )
    peers = None
    if "peers" in block:
        name, scale = block["peers"]
        peers = load_peers(sources[name], scale, similarity)

    out: dict = {}
    for a, b in PAIR_ORDER:
        if peers is None:
            record = combined_record(
                sources["combined"], block["scale"], similarity, a, b
            )
        else:
            record = peer_record(peers, a, b)
        if panel is not None:
            add_centroid(record, panel, a, b)
        out[(a, b)] = record

    origin = "reviewer summary" if peers is None else "saved peer table"
    centroid = "panel centroid" if panel is not None else f"centroid {MISSING}"
    print(f"[block] {block['label']:38s} {origin:17s} {centroid}")
    return out


def cell(record: dict, key: str) -> str:
    value = record.get(key)
    return MISSING if value is None else stacked_cell(*value, False)


def build_tabular(similarity: str, sources: dict) -> str:
    n_cols = 1 + len(N_SUB_HEADERS) + len(ESTIMATES)
    colspec = (
        r"@{}>{\raggedright\arraybackslash}p{2.75cm}"
        + r"@{\hspace{1.5pt}}c"
        + rf"@{{\hspace{{{COUNT_INNER_GAP_PT}pt}}}}c" * (len(N_SUB_HEADERS) - 1)
        + rf"@{{\hspace{{{COUNT_GUTTER_PT}pt}}}}"
        + rf"*{{{len(ESTIMATES)}}}{{>{{\raggedright\arraybackslash}}p{{1.62cm}}}}@{{}}"
    )
    top = (
        "Dataset pair & "
        + (N_GROUP_HEADER.replace("obs / cent / peers", "observed") if "retrieval_ci" in sources else N_GROUP_HEADER)
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

    for i, block in enumerate(BLOCKS):
        records = block_records(block, similarity, sources)
        if i:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{n_cols}}}{{l}}{{\textit{{{block['label']}}}}} \\")
        lines.append(r"\addlinespace[2pt]")
        for a, b in PAIR_ORDER:
            record = records[(a, b)]
            cells = [
                pair_label(a, b),
                fmt_count(record["n_cmpd"]),
                fmt_count(record["n_rows"]),
                *[cell(record, key) for _header, key in ESTIMATES],
            ]
            lines.append(" & ".join(cells) + r" \\")
            lines.append(r"\addlinespace[1pt]")
        if lines[-1] == r"\addlinespace[1pt]":
            lines.pop()

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def wrap_table(tabular: str, similarity: str) -> str:
    return "\n".join(
        [
            r"\begin{table*}[t]",
            r"\refstepcounter{table}",
            rf"\label{{tab:cross_dataset_retrieval_{similarity}_bootstrap_ci}}",
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
    parser.add_argument("--combined", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--logfc-peers", type=Path, default=DEFAULT_LOGFC_PEERS)
    parser.add_argument("--moderated-t-peers", type=Path, default=DEFAULT_T_PEERS)
    parser.add_argument(
        "--retrieval-ci", type=Path, action="append",
        help="Unified runner CI file; repeat for separate computations. Replaces saved panel inputs.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--similarity",
        action="append",
        choices=sorted(SIMILARITIES),
        help="repeatable; defaults to cosine and negative_l2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.retrieval_ci:
        sources = {"retrieval_ci": load_retrieval_ci(args.retrieval_ci)}
    else:
        for path in (args.combined, args.logfc_peers, args.moderated_t_peers):
            if not path.exists():
                raise SystemExit(f"missing {path}")
        sources = {
            "combined": load_combined(args.combined),
            "logfc": args.logfc_peers,
            "moderated_t": args.moderated_t_peers,
        }
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for similarity in args.similarity or ["cosine", "negative_l2"]:
        print(f"\n=== {similarity}")
        tex = wrap_table(
            build_tabular(similarity, sources), similarity,
        )
        prefix = f"table_{TABLE_NUMBERS[similarity]}_" if similarity in TABLE_NUMBERS else ""
        out = args.out_dir / f"{prefix}retrieval_{similarity}_combined.tex"
        out.write_text(tex, encoding="utf-8")
        print(
            f"[write] {relative(out)} "
            f"({len(BLOCKS)} blocks x {len(PAIR_ORDER)} pairs "
            f"= {len(BLOCKS) * len(PAIR_ORDER)} rows)"
        )


if __name__ == "__main__":
    main()
