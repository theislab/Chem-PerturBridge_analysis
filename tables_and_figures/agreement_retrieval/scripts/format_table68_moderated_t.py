#!/usr/bin/env python3
"""Format the moderated-t Tables 6/8 bootstrap intervals into paper table layout.

Reads results/overlap_group_rep_agreement_metrics/moderated_t_table68_cluster_bca_ci.tsv (written by
scripts/helper_group_rep_deg_signature_ci.py --compute moderated-t) and emits one table per analysis:

  deg        -> tables_and_figures/agreement_retrieval/data/table6/table_6_moderated_t.{tsv,md}
  signature  -> tables_and_figures/agreement_retrieval/data/table8/table_8_moderated_t.{tsv,md}

These extend the published moderated-t panels (table_6_moderated_t_reference.tsv and
table_8_moderated_t_reference.tsv) rather than replacing them: the observed / centroid /
delta trio is reproduced bit for bit from the same summariser, and the peer baselines,
target centroid and corrected percentiles are added alongside. The default check asserts that
reproduction against the published files, since silently drifting one of those three
columns is the failure this layout is meant to make impossible.

There is only one normalization here, so a pair gets one row.

A row is assembled from up to three summary levels, which is unavoidable rather than
untidy. The reference published the trio under its own names on "dataset_pair" (and, for
the signature delta, on "dataset_pair_baseline_adjusted"), and summary_level feeds the
bootstrap seed, so moving them onto the peer level to tidy this up would silently change
every published bound. Note also that "baseline_pair" is the reference's name for what
our scorer calls the source centroid.

n compounds is the observed panel's count. The baselines sit on a wider panel --
701 against 230 compounds for CIGS MCE/TCM in Table 6 -- so their counts stay in
their own columns instead of being folded into one, which is the step that makes
`delta != observed - baseline` look like an error.

The two peer sides get a column each. They agree in eight of the nine Table 6
pairs, but L1000 Phase I x Phase II has 254 source peers against 255 target
peers, so a single peer column would have to pick a side and silently misreport
the other. Table 8 is unaffected: source and target agree in every pair. The
centroid stays source-only because the displayed centroid column used for that
count is itself the source centroid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

# Shared with the logFC tables so pretty names and row order cannot drift apart.
from format_table68_raw_zscore import (
    PAIR_ORDER,
    dataset_pair_label,
    format_ci,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
# Match the summary helper's output directory; historical CI identifiers stay unchanged.
DEFAULT_INPUT = (
    REPO_ROOT / "results" / "overlap_group_rep_agreement_metrics" / "moderated_t_table68_cluster_bca_ci.tsv"
)
# Accept saved summaries from before the manuscript renumbering.
LEGACY_INPUT = DEFAULT_INPUT.with_name("moderated_t_table46_cluster_bca_ci.tsv")
DEFAULT_OUT_ROOT = REPO_ROOT / "tables_and_figures" / "agreement_retrieval" / "data"
SCORES_ROOT = REPO_ROOT / "results" / "parallel_cross_source"

# Saved summary levels retain their historical numbers: they also determine bootstrap seeds.
ANALYSES = {
    "deg": {
        "table": "Table 6",
        "out_subdir": "table6",
        "stem": "table_6_moderated_t",
        "published": "table6/table_6_moderated_t_reference.tsv",
        # metric stems: "pair" for the baseline families, "delta" for the contrasts
        "pair": "deg_t_spearman_pair_p05",
        "delta": "deg_t_spearman_p05",
        "peer_level": "table_4_moderated_t_peer_dataset_pair",
        "legacy": {
            "observed": ("observed_deg_t_spearman_sym_p05", "dataset_pair"),
            "source_centroid": ("baseline_pair_deg_t_spearman_p05", "dataset_pair"),
            "delta_vs_source_centroid": (
                "delta_vs_baseline_pair_deg_t_spearman_p05",
                "dataset_pair",
            ),
        },
        "caption": (
            "Table 6 (moderated t): Cross-dataset DEG-restricted Spearman agreement of "
            "moderated t statistics against centroid and individual-peer baselines. Means "
            "are condition-weighted; intervals are 95% BCa bootstrap confidence intervals "
            "clustered by PubChem CID. The DEG set is defined by adjusted $p < 0.05$."
        ),
    },
    "signature": {
        "table": "Table 8",
        "out_subdir": "table8",
        "stem": "table_8_moderated_t",
        "published": "table8/table_8_moderated_t_reference.tsv",
        "pair": "spearman_t_pair",
        "delta": "spearman_t",
        "peer_level": "table_6_moderated_t_peer_dataset_pair",
        "legacy": {
            "observed": ("observed_spearman_t", "dataset_pair"),
            "source_centroid": ("baseline_pair_spearman_t", "dataset_pair"),
            "delta_vs_source_centroid": (
                "delta_pair_mean_spearman_t",
                "dataset_pair_baseline_adjusted",
            ),
        },
        "caption": (
            "Table 8 (moderated t): Cross-dataset all-gene Spearman agreement of moderated "
            "t statistics against centroid and individual-peer baselines. Means are "
            "condition-weighted; intervals are 95% BCa bootstrap confidence intervals "
            "clustered by PubChem CID."
        ),
    },
}

# Published trio -> its column prefix in the reference panel, for --check.
PUBLISHED_PREFIXES = {
    "observed": "observed",
    "source_centroid": "baseline",
    "delta_vs_source_centroid": "delta",
}

DISPLAY_COLUMNS = [
    "Dataset pair",
    "n compounds",
    "n matched rows",
    "Observed [95% CI]",
    "Centroid (source) baseline [95% CI]",
    "Centroid (target) baseline [95% CI]",
    "Source peer baseline [95% CI]",
    "Target peer baseline [95% CI]",
    "Observed - centroid (source) [95% CI]",
    "Observed - centroid (target) [95% CI]",
    "Observed - source peer [95% CI]",
    "Observed - target peer [95% CI]",
    "Corrected percentile (source) [95% CI]",
    "Corrected percentile (target) [95% CI]",
    "Peer cap",
]

METRIC_ORDER = [
    "observed",
    "source_centroid",
    "target_centroid",
    "source_peer",
    "target_peer",
    "delta_vs_source_centroid",
    "delta_vs_target_centroid",
    "delta_vs_source_peer",
    "delta_vs_target_peer",
    "pct_source",
    "pct_target",
]


def peer_cap(analysis: str, run_tag: str, cache: dict[tuple[str, str], str]) -> str:
    """Report the cap the run was scored under, read from the run itself.

    Taking it from planned_run.json rather than parsing the run tag means a mislabelled
    tag shows up as a wrong cap in the table instead of being echoed back as verified.
    """
    key = (analysis, run_tag)
    if key not in cache:
        planned = SCORES_ROOT / analysis / run_tag / "planned_run.json"
        cap = json.loads(planned.read_text())["settings"]["max_baseline_peers"]
        cache[key] = "uncapped" if int(cap) == 0 else str(int(cap))
    return cache[key]


def metric_sources(cfg: dict) -> dict[str, tuple[str, str]]:
    """Map each output quantity to the (metric, summary_level) that carries it."""
    pair, delta, level = cfg["pair"], cfg["delta"], cfg["peer_level"]
    sources = dict(cfg["legacy"])
    sources.update(
        {
            "target_centroid": (f"pbt_target_centroid_{pair}", level),
            "source_peer": (f"pbt_source_peer_{pair}", level),
            "target_peer": (f"pbt_target_peer_{pair}", level),
            "delta_vs_target_centroid": (f"pbt_delta_vs_target_centroid_{delta}", level),
            "delta_vs_source_peer": (f"pbt_delta_vs_source_peer_{delta}", level),
            "delta_vs_target_peer": (f"pbt_delta_vs_target_peer_{delta}", level),
            "pct_source": (f"pbt_source_peer_{pair}_corrected_percentile", level),
            "pct_target": (f"pbt_target_peer_{pair}_corrected_percentile", level),
        }
    )
    return sources


def build_table(ci: pd.DataFrame, analysis: str) -> pd.DataFrame:
    cfg = ANALYSES[analysis]
    ci = ci.loc[ci["analysis"] == analysis]
    sources = metric_sources(cfg)
    cap_cache: dict[tuple[str, str], str] = {}
    rows = []

    for a, b in PAIR_ORDER:
        pair_rows = ci.loc[(ci["dataset_a"] == a) & (ci["dataset_b"] == b)]
        if pair_rows.empty:
            print(f"[skip] {cfg['table']}: no rows for {a}-{b}")
            continue

        by_key = pair_rows.set_index(["metric", "summary_level"])
        missing = [f"{m} @ {lvl}" for m, lvl in sources.values() if (m, lvl) not in by_key.index]
        if missing:
            raise KeyError(
                f"{cfg['table']} / {a}-{b} is missing {missing}. "
                "Rebuild with scripts/helper_group_rep_deg_signature_ci.py --compute moderated-t --verify."
            )
        m = {key: by_key.loc[source] for key, source in sources.items()}
        obs = m["observed"]

        row = {
            "Dataset pair": dataset_pair_label(a, b),
            "n compounds": int(obs["n_compounds"]),
            "n matched rows": f"{int(obs['n_rows']):,}",
            "Observed [95% CI]": format_ci(obs),
            "Centroid (source) baseline [95% CI]": format_ci(m["source_centroid"]),
            "Centroid (target) baseline [95% CI]": format_ci(m["target_centroid"]),
            "Source peer baseline [95% CI]": format_ci(m["source_peer"]),
            "Target peer baseline [95% CI]": format_ci(m["target_peer"]),
            "Observed - centroid (source) [95% CI]": format_ci(m["delta_vs_source_centroid"]),
            "Observed - centroid (target) [95% CI]": format_ci(m["delta_vs_target_centroid"]),
            "Observed - source peer [95% CI]": format_ci(m["delta_vs_source_peer"]),
            "Observed - target peer [95% CI]": format_ci(m["delta_vs_target_peer"]),
            "Corrected percentile (source) [95% CI]": format_ci(m["pct_source"]),
            "Corrected percentile (target) [95% CI]": format_ci(m["pct_target"]),
            "Peer cap": peer_cap(analysis, str(obs["run_tag"]), cap_cache),
        }
        for key in METRIC_ORDER:
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
                "run_tag": str(obs["run_tag"]),
                "dataset_a": a,
                "dataset_b": b,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def check_against_published(table: pd.DataFrame, analysis: str, tolerance: float) -> None:
    """Fail if the trio has drifted from the published panel it is meant to extend."""
    cfg = ANALYSES[analysis]
    published_path = DEFAULT_OUT_ROOT / cfg["published"]
    if not published_path.exists():
        print(f"[check] {cfg['table']}: no published panel at {published_path}, skipped")
        return

    published = pd.read_csv(published_path, sep="\t")
    merged = table.merge(published, on=["dataset_a", "dataset_b"], suffixes=("", "_pub"))
    if len(merged) != len(table):
        raise SystemExit(f"[check] {cfg['table']}: {len(table)} rows matched {len(merged)}")

    worst, worst_at = 0.0, ""
    for key, prefix in PUBLISHED_PREFIXES.items():
        for stat in ("mean", "ci_low", "ci_high"):
            reference_column = f"{prefix}_{stat}"
            # Overlapping reference columns receive the suffix during the merge.
            if reference_column in table.columns:
                reference_column += "_pub"
            diff = (merged[f"{key}_{stat}"] - merged[reference_column]).abs()
            if diff.max() > worst:
                worst, worst_at = diff.max(), f"{key}_{stat}"
    if worst > tolerance:
        raise SystemExit(
            f"[check] {cfg['table']} FAILED: trio differs from {published_path.name} by "
            f"{worst:.2e} at {worst_at} (tolerance {tolerance:.0e}). The published columns "
            "must not move; rerun scripts/helper_group_rep_deg_signature_ci.py --compute moderated-t --verify."
        )
    print(
        f"[check] {cfg['table']}: trio reproduces {published_path.name} "
        f"in {len(merged) * 9} values, max |difference| {worst:.2e}"
    )


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
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument(
        "--analysis",
        choices=[*sorted(ANALYSES), "both"],
        default="both",
        help="Which table to format (default: both).",
    )
    p.add_argument("--no-write", action="store_true")
    p.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the comparison of the trio against the published panels.",
    )
    p.add_argument("--tolerance", type=float, default=1e-12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(
            f"Missing input: {args.input}\n"
            "Generate it with scripts/helper_group_rep_deg_signature_ci.py --compute moderated-t --verify."
        )

    ci = pd.read_csv(args.input, sep="\t")
    analyses = sorted(ANALYSES) if args.analysis == "both" else [args.analysis]

    for analysis in analyses:
        cfg = ANALYSES[analysis]
        table = build_table(ci, analysis)
        if table.empty:
            print(f"[skip] {cfg['table']}: nothing to format")
            continue

        if not args.no_check:
            check_against_published(table, analysis, args.tolerance)

        if not args.no_write:
            out_dir = args.out_root / cfg["out_subdir"]
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
                    "n compounds",
                    "Observed [95% CI]",
                    "Centroid (source) baseline [95% CI]",
                    "Source peer baseline [95% CI]",
                    "Observed - source peer [95% CI]",
                    "Peer cap",
                ]
            ].to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()
