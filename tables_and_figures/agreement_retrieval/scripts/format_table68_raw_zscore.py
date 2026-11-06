#!/usr/bin/env python3
"""Format the raw and z-score Tables 6/8 bootstrap intervals into paper table layout.

Reads results/overlap_group_rep_agreement_metrics/capfix_table68_cluster_bca_ci.tsv (written by
scripts/helper_group_rep_deg_signature_ci.py) and emits one table per analysis:

  deg        -> tables_and_figures/agreement_retrieval/data/table6/table_6_raw_zscore.{tsv,md}
  signature  -> tables_and_figures/agreement_retrieval/data/table8/table_8_raw_zscore.{tsv,md}

Each table carries all three normalizations for every dataset pair: display columns hold
"mean [low, high]" strings, followed by the raw numeric columns behind them.

The raw layer reads summary_level="dataset_pair", the single level helper_group_rep_deg_signature_ci
now summarizes it at. Within that layer the observed metric and the source centroid
family carry the notebook references' own labels rather than the pb_ ones, since
_stable_seed hashes the label alongside the level; "raw_reference" per analysis
below is what redirects those lookups, and it has to stay in step with
RAW_REFERENCE_NAMES in helper_group_rep_deg_signature_ci.py or the metric goes missing and the
build fails loudly.

n compounds is the observed panel's count. The baselines sit on a wider panel --
701 against 230 compounds for CIGS MCE/TCM in Table 6 -- so their counts are kept
in their own columns rather than folded into one, which is the step that makes
`delta != observed - baseline` look like an error in the reference workbook.

The two peer sides get a column each. They agree in 51 of the 54 pair/metric/
normalization combinations, but L1000 Phase I x Phase II in Table 6 has 254
source peers against 255 target peers in all three normalizations, so a single
peer column would have to pick a side and silently misreport the other. The
centroid stays source-only because the displayed centroid column is itself the
source centroid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
# Match the summary helper's output directory; historical CI identifiers stay unchanged.
DEFAULT_INPUT = REPO_ROOT / "results" / "overlap_group_rep_agreement_metrics" / "capfix_table68_cluster_bca_ci.tsv"
# Accept saved summaries from before the manuscript renumbering.
LEGACY_INPUT = DEFAULT_INPUT.with_name("capfix_table46_cluster_bca_ci.tsv")
DEFAULT_OUT_DIR = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data"
SCORES_ROOT = REPO_ROOT / "results" / "parallel_cross_source"

DATASET_DISPLAY = {
    "cigs_mce": "CIGS-MCE",
    "cigs_tcm": "CIGS-TCM",
    "l1000_phase1": "L1000 Phase I",
    "l1000_phase2": "L1000 Phase II",
    "sciplex": "sci-Plex",
    "tahoe": "Tahoe-100M",
}

PAIR_ORDER = [
    ("cigs_mce", "cigs_tcm"),
    ("l1000_phase1", "cigs_mce"),
    ("l1000_phase1", "l1000_phase2"),
    ("l1000_phase1", "sciplex"),
    ("l1000_phase1", "tahoe"),
    ("l1000_phase2", "cigs_mce"),
    ("l1000_phase2", "sciplex"),
    ("l1000_phase2", "tahoe"),
    ("tahoe", "sciplex"),
]

# Saved summary levels retain their historical numbers: they also determine bootstrap seeds.
ANALYSES = {
    "deg": {
        "table": "Table 6",
        "out_subdir": "table6",
        "stem": "table_6_raw_zscore",
        # metric stems: "pair" for the baseline families, "delta" for the contrasts
        "observed": "observed_deg_lfc_spearman_sym_p05",
        "pair": "deg_lfc_spearman_pair_p05",
        "delta": "deg_lfc_spearman_p05",
        "levels": {
            # Merged with the legacy trio onto the notebook's level so the peer family
            # shares the observed metric's bootstrap seed; see helper_group_rep_deg_signature_ci.py.
            "raw": "dataset_pair",
            "per_gene_population_zscore_dataset": "w4_deg_dataset_pair",
            "per_gene_population_zscore_dataset_cell_type": "table_4_dataset_cell_type_dataset_pair",
        },
        # Raw-layer metrics helper_group_rep_deg_signature_ci publishes under the notebook's own
        # labels rather than the pb_ ones, so that they draw the same bootstrap
        # samples; see RAW_REFERENCE_NAMES there.
        "raw_reference": {
            "source_centroid": "baseline_pair_deg_lfc_spearman_p05",
            "delta_vs_centroid": "delta_vs_baseline_pair_deg_lfc_spearman_p05",
        },
        # The deg reference bootstraps its whole trio in one call, so nothing here.
        "raw_reference_levels": {},
        "caption": (
            "Table 6: Cross-dataset DEG-restricted Spearman agreement against centroid "
            "and individual-peer baselines, for each normalization. Means are "
            "condition-weighted; intervals are 95% BCa bootstrap confidence intervals "
            "clustered by PubChem CID. The DEG set is defined by adjusted $p < 0.05$."
        ),
    },
    "signature": {
        "table": "Table 8",
        "out_subdir": "table8",
        "stem": "table_8_raw_zscore",
        "observed": "observed_spearman_logfc",
        "pair": "spearman_logfc_pair",
        "delta": "spearman_logfc",
        "levels": {
            # See the deg arm: merged onto the notebook's level for a shared seed.
            "raw": "dataset_pair",
            "per_gene_population_zscore_dataset": "w4_signature_dataset_pair",
            "per_gene_population_zscore_dataset_cell_type": "table_6_dataset_cell_type_dataset_pair",
        },
        "raw_reference": {
            "source_centroid": "baseline_pair_spearman_logfc",
            "delta_vs_centroid": "delta_pair_mean_spearman_logfc",
        },
        # The signature reference bootstraps its delta off a separate baseline-adjusted
        # frame, so helper_group_rep_deg_signature_ci emits it at that level; same panel, different
        # draws. Keyed by display name, applied on top of "levels" for the raw layer.
        "raw_reference_levels": {
            "delta_vs_centroid": "dataset_pair_baseline_adjusted",
        },
        "caption": (
            "Table 8: Cross-dataset all-gene Spearman agreement against centroid and "
            "individual-peer baselines, for each normalization. Means are "
            "condition-weighted; intervals are 95% BCa bootstrap confidence intervals "
            "clustered by PubChem CID."
        ),
    },
}

# scale_variant -> (display label, metric prefix). The raw layer says "pb_" where the
# z layers say "w4_", and its observed metric carries no prefix at all.
NORMALIZATIONS = [
    ("raw", "raw", "pb_"),
    ("dataset z-score", "per_gene_population_zscore_dataset", "w4_"),
    ("dataset x cell line z-score", "per_gene_population_zscore_dataset_cell_type", "w4_"),
]

DISPLAY_COLUMNS = [
    "Dataset pair",
    "Normalization",
    "n compounds",
    "n matched rows",
    "Observed [95% CI]",
    "Source peer baseline [95% CI]",
    "Centroid (source) baseline [95% CI]",
    "Target peer baseline [95% CI]",
    "Observed - centroid [95% CI]",
    "Observed - target peer [95% CI]",
    "Observed - source peer [95% CI]",
    "Corrected percentile (source) [95% CI]",
    "Corrected percentile (target) [95% CI]",
    "Peer cap",
]


def format_ci(row: pd.Series, digits: int = 3) -> str:
    return f"{row['mean']:.{digits}f} [{row['ci_low']:.{digits}f}, {row['ci_high']:.{digits}f}]"


def dataset_pair_label(dataset_a: str, dataset_b: str) -> str:
    return f"{DATASET_DISPLAY[dataset_a]}-{DATASET_DISPLAY[dataset_b]}"


def peer_cap(analysis: str, run_tag: str, cache: dict[tuple[str, str], str]) -> str:
    """Report the cap the layer was scored under, read from the run itself.

    Taking it from planned_run.json rather than parsing the run tag means a
    mislabelled tag shows up as a wrong cap in the table instead of being echoed
    back as though it were verified.
    """
    key = (analysis, run_tag)
    if key not in cache:
        planned = SCORES_ROOT / analysis / run_tag / "planned_run.json"
        cap = json.loads(planned.read_text())["settings"]["max_baseline_peers"]
        cache[key] = "uncapped" if int(cap) == 0 else str(int(cap))
    return cache[key]


def metric_names(cfg: dict, prefix: str) -> dict[str, str]:
    pair, delta = cfg["pair"], cfg["delta"]
    names = {
        "observed": cfg["observed"] if prefix == "pb_" else f"w4_{cfg['observed']}",
        "source_peer": f"{prefix}source_peer_{pair}",
        "source_centroid": f"{prefix}source_centroid_{pair}",
        "target_peer": f"{prefix}target_peer_{pair}",
        "delta_vs_centroid": f"{prefix}delta_vs_source_centroid_{delta}",
        "delta_vs_target_peer": f"{prefix}delta_vs_target_peer_{delta}",
        "delta_vs_source_peer": f"{prefix}delta_vs_source_peer_{delta}",
        "pct_source": f"{prefix}source_peer_{pair}_corrected_percentile",
        "pct_target": f"{prefix}target_peer_{pair}_corrected_percentile",
    }
    if prefix == "pb_":
        names.update(cfg["raw_reference"])
    return names


def build_table(ci: pd.DataFrame, analysis: str) -> pd.DataFrame:
    cfg = ANALYSES[analysis]
    ci = ci.loc[ci["analysis"] == analysis]
    cap_cache: dict[tuple[str, str], str] = {}
    rows = []

    for label, scale_variant, prefix in NORMALIZATIONS:
        level = cfg["levels"][scale_variant]
        # Metric -> the level it was bootstrapped at, which is the layer's own except
        # for the reference metrics listed in "raw_reference_levels".
        metric_level = {}
        if prefix == "pb_":
            metric_level = {
                cfg["raw_reference"][key]: value
                for key, value in cfg["raw_reference_levels"].items()
            }
        variant = ci.loc[ci["scale_variant"] == scale_variant]
        layer = variant.loc[variant["summary_level"] == level]
        if layer.empty:
            print(f"[skip] {cfg['table']} / {label}: no rows at summary_level={level}")
            continue
        names = metric_names(cfg, prefix)

        for a, b in PAIR_ORDER:
            pair = variant.loc[(variant["dataset_a"] == a) & (variant["dataset_b"] == b)]
            sub = pair.loc[pair["summary_level"] == level]
            if sub.empty:
                print(f"[skip] {cfg['table']} / {label}: no rows for {a}-{b}")
                continue
            by_metric = pair.set_index(["summary_level", "metric"])
            wanted = {name: (metric_level.get(name, level), name) for name in names.values()}
            missing = [k for k in wanted.values() if k not in by_metric.index]
            if missing:
                raise KeyError(
                    f"{cfg['table']} / {label} / {a}-{b} is missing {missing}. "
                    "Rebuild the CIs with python scripts/helper_group_rep_deg_signature_ci.py --compute raw,z-score-dataset,z-score-dataset-cell-type."
                )
            m = {key: by_metric.loc[wanted[name]] for key, name in names.items()}
            obs = m["observed"]

            row = {
                "Dataset pair": dataset_pair_label(a, b),
                "Normalization": label,
                "n compounds": int(obs["n_compounds"]),
                "n matched rows": f"{int(obs['n_rows']):,}",
                "Observed [95% CI]": format_ci(obs),
                "Source peer baseline [95% CI]": format_ci(m["source_peer"]),
                "Centroid (source) baseline [95% CI]": format_ci(m["source_centroid"]),
                "Target peer baseline [95% CI]": format_ci(m["target_peer"]),
                "Observed - centroid [95% CI]": format_ci(m["delta_vs_centroid"]),
                "Observed - target peer [95% CI]": format_ci(m["delta_vs_target_peer"]),
                "Observed - source peer [95% CI]": format_ci(m["delta_vs_source_peer"]),
                "Corrected percentile (source) [95% CI]": format_ci(m["pct_source"]),
                "Corrected percentile (target) [95% CI]": format_ci(m["pct_target"]),
                "Peer cap": peer_cap(analysis, str(obs["run_tag"]), cap_cache),
            }
            for key in (
                "observed",
                "source_peer",
                "source_centroid",
                "target_peer",
                "delta_vs_centroid",
                "delta_vs_target_peer",
                "delta_vs_source_peer",
                "pct_source",
                "pct_target",
            ):
                row[f"{key}_mean"] = float(m[key]["mean"])
                row[f"{key}_ci_low"] = float(m[key]["ci_low"])
                row[f"{key}_ci_high"] = float(m[key]["ci_high"])
            row.update(
                {
                    "n_compounds_observed": int(obs["n_compounds"]),
                    "n_compounds_centroid": int(m["source_centroid"]["n_compounds"]),
                    "n_compounds_source_peer": int(m["source_peer"]["n_compounds"]),
                    "n_compounds_target_peer": int(m["target_peer"]["n_compounds"]),
                    "n_rows": int(obs["n_rows"]),
                    "scale_variant": scale_variant,
                    "run_tag": str(obs["run_tag"]),
                    "dataset_a": a,
                    "dataset_b": b,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, caption: str) -> str:
    lines = [
        caption,
        "",
        "| " + " | ".join(DISPLAY_COLUMNS) + " |",
        "| " + " | ".join("---" for _ in DISPLAY_COLUMNS) + " |",
    ]
    for _, r in table.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in DISPLAY_COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    default_input = DEFAULT_INPUT
    if not default_input.exists() and LEGACY_INPUT.exists():
        default_input = LEGACY_INPUT
    p.add_argument("--input", type=Path, default=default_input)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--analysis",
        choices=[*sorted(ANALYSES), "both"],
        default="both",
        help="Which table to format (default: both).",
    )
    p.add_argument("--no-write", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"Missing input: {args.input}\n"
            "Generate it with python scripts/helper_group_rep_deg_signature_ci.py --compute raw,z-score-dataset,z-score-dataset-cell-type."
        )

    ci = pd.read_csv(args.input, sep="\t")
    analyses = sorted(ANALYSES) if args.analysis == "both" else [args.analysis]

    for analysis in analyses:
        cfg = ANALYSES[analysis]
        table = build_table(ci, analysis)
        if table.empty:
            print(f"[skip] {cfg['table']}: nothing to format")
            continue

        if not args.no_write:
            out_dir = args.out_dir / cfg["out_subdir"]
            out_dir.mkdir(parents=True, exist_ok=True)
            extra = [c for c in table.columns if c not in DISPLAY_COLUMNS]
            tsv_path = out_dir / f"{cfg['stem']}.tsv"
            md_path = out_dir / f"{cfg['stem']}.md"
            table[DISPLAY_COLUMNS + extra].to_csv(tsv_path, sep="\t", index=False)
            md_path.write_text(to_markdown(table, cfg["caption"]), encoding="utf-8")
            print(f"Wrote {tsv_path}")
            print(f"Wrote {md_path}")

        print(f"\n{cfg['table']} ({len(table)} rows)")
        print(
            table[
                [
                    "Dataset pair",
                    "Normalization",
                    "n compounds",
                    "Observed [95% CI]",
                    "Source peer baseline [95% CI]",
                    "Observed - source peer [95% CI]",
                    "Peer cap",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
