"""Summarize Tables 6/8 scores for raw, population z-scores, and moderated-t.

All modes share condition aggregation and compound-cluster BCa intervals. Historical
bootstrap inputs are preserved: base seed, summary levels, ordered group values,
metric labels, value-column names, and the sorted compound axis. Display/CLI names
must never replace those internal identifiers.

By default, rebuild all four computations into the two existing summary files.
Use --compute to select components and --verify to check moderated-t legacy rows.
"""

from __future__ import annotations

import argparse
import difflib
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

if __package__:
    from .cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table
else:
    from cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table

REPO = Path(__file__).resolve().parents[1]
# Filenames follow the current tables; historical summary-level keys preserve bootstrap seeds.
OUT_PATH = REPO / "results/overlap_group_rep_agreement_metrics/capfix_table68_cluster_bca_ci.tsv"

DRUG_LINE_TIME_COLUMNS = ["dataset_a", "dataset_b", "cell_type", "time_key", "pubchem_cid"]
IDENTITY = [*DRUG_LINE_TIME_COLUMNS, "matched_condition_key", "left_obs_id", "right_obs_id"]

RAW_VARIANT = "raw"
DATASET_VARIANT = "per_gene_population_zscore_dataset"
CELL_TYPE_VARIANT = "per_gene_population_zscore_dataset_cell_type"

Z_SCORE_DEG_METRICS = [
    "w4_observed_deg_lfc_spearman_sym_p05",
    "w4_source_centroid_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_source_centroid_deg_lfc_spearman_p05",
    "w4_target_centroid_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_target_centroid_deg_lfc_spearman_p05",
    "w4_source_peer_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_source_peer_deg_lfc_spearman_p05",
    "w4_source_peer_deg_lfc_spearman_pair_p05_sd_score",
    "w4_source_peer_deg_lfc_spearman_pair_p05_fraction_below_observed",
    "w4_source_peer_deg_lfc_spearman_pair_p05_corrected_percentile",
    "w4_target_peer_deg_lfc_spearman_pair_p05",
    "w4_delta_vs_target_peer_deg_lfc_spearman_p05",
    "w4_target_peer_deg_lfc_spearman_pair_p05_sd_score",
    "w4_target_peer_deg_lfc_spearman_pair_p05_fraction_below_observed",
    "w4_target_peer_deg_lfc_spearman_pair_p05_corrected_percentile",
]
Z_SCORE_SIGNATURE_METRICS = [
    "w4_observed_spearman_logfc",
    "w4_source_centroid_spearman_logfc_pair",
    "w4_delta_vs_source_centroid_spearman_logfc",
    "w4_target_centroid_spearman_logfc_pair",
    "w4_delta_vs_target_centroid_spearman_logfc",
    "w4_source_peer_spearman_logfc_pair",
    "w4_delta_vs_source_peer_spearman_logfc",
    "w4_source_peer_spearman_logfc_pair_sd_score",
    "w4_source_peer_spearman_logfc_pair_fraction_below_observed",
    "w4_source_peer_spearman_logfc_pair_corrected_percentile",
    "w4_target_peer_spearman_logfc_pair",
    "w4_delta_vs_target_peer_spearman_logfc",
    "w4_target_peer_spearman_logfc_pair_sd_score",
    "w4_target_peer_spearman_logfc_pair_fraction_below_observed",
    "w4_target_peer_spearman_logfc_pair_corrected_percentile",
]

# The observed metric the notebook's own overlap_group_rep_deg_cluster_bca_ci.tsv
# publishes at summary_level="dataset_pair". raw_peer_name maps the w4 observed metric
# onto exactly these names, so the merged raw call reproduces them rather than adding
# a second copy; asserted in summarize_analysis so a rename cannot silently break it.
RAW_ANCHOR_METRICS = {
    "deg": "observed_deg_lfc_spearman_sym_p05",
    "signature": "observed_spearman_logfc",
}

# Raw-layer columns the notebook references publish under a different label.
# summarize_layer renames the source column to the metric name and bootstraps
# mean_<name>, and _stable_seed hashes both, so emitting these under the reference's
# own labels is what puts them on its draws rather than merely on its level and key.
# Each pb_ column is bit-identical to the legacy one it is renamed onto, over every
# scored row and with the same NaN pattern, so only the label moves.
RAW_REFERENCE_NAMES = {
    "deg": {
        "pb_source_centroid_deg_lfc_spearman_pair_p05": "baseline_pair_deg_lfc_spearman_p05",
        "pb_delta_vs_source_centroid_deg_lfc_spearman_p05": (
            "delta_vs_baseline_pair_deg_lfc_spearman_p05"
        ),
    },
    "signature": {
        "pb_source_centroid_spearman_logfc_pair": "baseline_pair_spearman_logfc",
        "pb_delta_vs_source_centroid_spearman_logfc": "delta_pair_mean_spearman_logfc",
    },
}

# Reference metrics whose bootstrap ran at a level other than the layer's own. The
# signature notebook takes its delta off a separate baseline-adjusted frame, so it
# lands on "dataset_pair_baseline_adjusted" while everything else sits on
# "dataset_pair". That frame is the same panel -- identical n_rows, n_finite_rows,
# n_compounds and mean to 1.1e-16 -- so the split is bookkeeping in the reference
# rather than a different aggregation, and reproducing it costs only an extra call.
#
# This is not the seed split that prompted the merge. That one emitted the observed
# metric from two calls, leaving one quantity with two intervals; here the delta is
# emitted once and simply drawn on its own stream, which every metric already is,
# since _stable_seed hashes metric_label and value_col per metric anyway.
RAW_REFERENCE_LEVELS = {
    "deg": {},
    "signature": {"delta_pair_mean_spearman_logfc": "dataset_pair_baseline_adjusted"},
}


def raw_peer_name(z_score_metric: str) -> str:
    """Translate a z-layer metric name to its raw-layer counterpart.

    The raw run drops the "{variant}__" prefix and says "pb_" where the z layers say
    "w4_", except the observed metric which carries no prefix at all. Anything that
    fails to translate would be silently skipped as a missing column, so callers
    check presence explicitly rather than trusting this mapping.
    """
    stem = z_score_metric.removeprefix("w4_")
    return stem if stem.startswith("observed_") else f"pb_{stem}"


# summary_level feeds the per-group bootstrap seed, so it must match the reference
# run for the draws to be identical. Every raw metric shares the reference level, which
# is what puts the peer family on the same stream as the observed metric.
SUMMARY_LEVELS = {
    ("deg", DATASET_VARIANT): "w4_deg_dataset_pair",
    ("deg", CELL_TYPE_VARIANT): "table_4_dataset_cell_type_dataset_pair",
    ("signature", DATASET_VARIANT): "w4_signature_dataset_pair",
    ("signature", CELL_TYPE_VARIANT): "table_6_dataset_cell_type_dataset_pair",
    ("deg", RAW_VARIANT): "dataset_pair",
    ("signature", RAW_VARIANT): "dataset_pair",
}

# analysis -> scale_variant -> capfix run directory holding that layer's scores.
CAPFIX_RUNS = {
    "deg": {
        RAW_VARIANT: "capfix_raw_256",
        DATASET_VARIANT: "capfix_z_dataset_256",
        CELL_TYPE_VARIANT: "capfix_z_cell_type_256",
    },
    "signature": {
        RAW_VARIANT: "capfix_raw_256",
        DATASET_VARIANT: "capfix_z_256",
        CELL_TYPE_VARIANT: "capfix_z_256",
    },
}
SCORED_STEMS = {"deg": "deg_scored_metrics.tsv", "signature": "signature_scored_metrics.tsv"}


# The reference lineage omits scale_variant, so grouping matches it exactly. Raw-only runs
# make the column constant anyway.
MODERATED_T_GROUP_COLS = ["dataset_a", "dataset_b"]
N_BOOT = 2000
SEED = 20260505

# Peer columns, new in these runs. Mirrors Z_SCORE_DEG_METRICS / Z_SCORE_SIGNATURE_METRICS in
# the raw/z-score families so moderated-t rows can be joined to their logFC partners metric by
# metric. The centroid pair/delta live in the legacy family instead, since the reference
# already publishes them under its own names.
PEER_METRICS = {
    "deg": [
        "pbt_target_centroid_deg_t_spearman_pair_p05",
        "pbt_delta_vs_target_centroid_deg_t_spearman_p05",
        "pbt_source_peer_deg_t_spearman_pair_p05",
        "pbt_delta_vs_source_peer_deg_t_spearman_p05",
        "pbt_source_peer_deg_t_spearman_pair_p05_sd_score",
        "pbt_source_peer_deg_t_spearman_pair_p05_fraction_below_observed",
        "pbt_source_peer_deg_t_spearman_pair_p05_corrected_percentile",
        "pbt_target_peer_deg_t_spearman_pair_p05",
        "pbt_delta_vs_target_peer_deg_t_spearman_p05",
        "pbt_target_peer_deg_t_spearman_pair_p05_sd_score",
        "pbt_target_peer_deg_t_spearman_pair_p05_fraction_below_observed",
        "pbt_target_peer_deg_t_spearman_pair_p05_corrected_percentile",
    ],
    "signature": [
        "pbt_target_centroid_spearman_t_pair",
        "pbt_delta_vs_target_centroid_spearman_t",
        "pbt_source_peer_spearman_t_pair",
        "pbt_delta_vs_source_peer_spearman_t",
        "pbt_source_peer_spearman_t_pair_sd_score",
        "pbt_source_peer_spearman_t_pair_fraction_below_observed",
        "pbt_source_peer_spearman_t_pair_corrected_percentile",
        "pbt_target_peer_spearman_t_pair",
        "pbt_delta_vs_target_peer_spearman_t",
        "pbt_target_peer_spearman_t_pair_sd_score",
        "pbt_target_peer_spearman_t_pair_fraction_below_observed",
        "pbt_target_peer_spearman_t_pair_corrected_percentile",
    ],
}

# published name -> our column. Verified by identical means and compound counts, which is
# also what identifies the reference's "baseline_pair" as the source-centroid baseline.
LEGACY_ALIASES = {
    "deg": {
        "dataset_pair": {
            "observed_deg_t_spearman_sym_p05": "pbt_observed_deg_t_spearman_sym_p05",
            "baseline_pair_deg_t_spearman_p05": (
                "pbt_source_centroid_deg_t_spearman_pair_p05"
            ),
            "delta_vs_baseline_pair_deg_t_spearman_p05": (
                "pbt_delta_vs_source_centroid_deg_t_spearman_p05"
            ),
        },
    },
    "signature": {
        "dataset_pair": {
            "observed_spearman_t": "pbt_observed_spearman_t",
            "baseline_pair_spearman_t": "pbt_source_centroid_spearman_t_pair",
        },
        # The reference puts the all-gene (current Table 8) delta on its own level, so it needs its own call.
        "dataset_pair_baseline_adjusted": {
            "delta_pair_mean_spearman_t": "pbt_delta_vs_source_centroid_spearman_t",
        },
    },
}
PEER_SUMMARY_LEVELS = {
    "deg": "table_4_moderated_t_peer_dataset_pair",
    "signature": "table_6_moderated_t_peer_dataset_pair",
}
# Fallback filenames for saved moderated-t runs from before filename unification.
MODERATED_T_SCORED_STEMS = {
    "deg": "deg_moderated_t_scored_metrics.tsv",
    "signature": "signature_moderated_t_scored_metrics.tsv",
}
TABLES = {"deg": "Table 6", "signature": "Table 8"}
# Published CI tables the legacy family must reproduce, for --verify.
REFERENCE_CI = {
    "deg": REPO
    / "results/overlap_group_rep_deg_metrics_t_stats/overlap_group_rep_deg_t_cluster_bca_ci.tsv",
    "signature": REPO
    / "results/overlap_signature_similarity_group_rep_t_stats/signature_similarity_t_cluster_bca_ci.tsv",
}


MODERATED_T_OUT_PATH = REPO / "results/overlap_group_rep_agreement_metrics/moderated_t_table68_cluster_bca_ci.tsv"
COMPUTATIONS = ("raw", "z-score-dataset", "z-score-dataset-cell-type", "moderated-t")
VARIANT_COMPUTATIONS = {
    RAW_VARIANT: "raw",
    DATASET_VARIANT: "z-score-dataset",
    CELL_TYPE_VARIANT: "z-score-dataset-cell-type",
}


def canonical_cluster_ids(values: pd.Series) -> pd.Series:
    """Return compound ids as integral text, matching the reference summaries.

    cluster_bca_nested_mean_ci_table orders its cluster axis by first appearance
    and we sort before handing the frame over, so the axis inherits this dtype:
    int64 ids sort numerically (2123, 2478, ...) and text ids lexically ('104741',
    '11167602', ...). Those are permutations of each other, which leaves every mean
    bit-identical while moving every bootstrap interval bound. The reference read
    the column as text, so any other dtype here silently rewrites all published
    CIs without touching a single point estimate.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    integral = numeric.notna() & (numeric % 1 == 0)
    canonical = values.astype("string").str.strip()
    canonical.loc[integral] = numeric.loc[integral].astype("int64").astype("string")
    return canonical.fillna("")


def require_metrics(path: Path, analysis: str, metrics: list[str]) -> None:
    """Fail loudly on a renamed or missing metric.

    Absence here always means a rename, never an unscored layer, so it must not be
    downgraded to a warning that leaves a quietly truncated table behind.
    """
    header = set(pd.read_csv(path, sep="\t", nrows=0).columns)
    missing = [metric for metric in metrics if metric not in header]
    if not missing:
        return
    lines = [f"{analysis}: {len(missing)} of {len(metrics)} metrics absent from {path.name}"]
    for metric in missing:
        lines.append(f"  missing {metric}")
        for candidate in difflib.get_close_matches(metric, header, n=2, cutoff=0.6):
            lines.append(f"     did you mean {candidate}?")
    lines.append("Rescore with the current scorer, or update the metric lists if a name changed.")
    raise SystemExit("\n".join(lines))


def summarize_layer(
    path: Path,
    analysis: str,
    variant: str | None,
    columns: dict[str, str],
    summary_level: str,
    group_cols: Sequence[str] = ("dataset_a", "dataset_b", "scale_variant"),
) -> pd.DataFrame | None:
    """Aggregate one (run, scale_variant) layer to drug x line x time, then bootstrap.

    columns maps output metric name -> column name in this run's scored file, which is
    what lets all representations share this code despite their different naming.

    group_cols is the bootstrap's grouping key, and it is load-bearing beyond grouping:
    _stable_seed hashes repr() of the whole key tuple, so carrying scale_variant in it
    reseeds every cell relative to a reference run whose frame had no such column.
    """
    header = pd.read_csv(path, sep="\t", nrows=0).columns.tolist()
    present = {name: column for name, column in columns.items() if column in header}
    if not present:
        print(f"[skip] {analysis}/{variant}: none of {len(columns)} metrics in {path.name}")
        return None
    if len(present) != len(columns):
        missing = sorted(set(columns) - set(present))
        print(f"[warn] {analysis}/{variant}: {len(missing)} metrics absent, e.g. {missing[:3]}")

    usecols = [c for c in IDENTITY if c in header] + list(present.values())
    frame = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    # Both must precede the groupby and sort below, which is what fixes the cluster
    # order. time_key is the same hazard as pubchem_cid: as int64 it sorts 6 before 24,
    # as text "24" before "6", which reorders the time blocks within each cell line and
    # so permutes the compounds' first-appearance order. It bites only where a pair
    # spans more than one time point, which here is L1000 Phase I-Phase II alone.
    frame["pubchem_cid"] = canonical_cluster_ids(frame["pubchem_cid"])
    frame["time_key"] = frame["time_key"].astype(str)
    frame = frame.rename(columns={column: name for name, column in present.items()})
    # The legacy moderated-t frame had no scale_variant during aggregation.
    # Keep that exact grouping/sort order as well as the bootstrap group keys.
    condition_cols = DRUG_LINE_TIME_COLUMNS
    if variant is not None:
        frame["scale_variant"] = variant
        condition_cols = ["scale_variant", *DRUG_LINE_TIME_COLUMNS]

    agg = {
        "n_matched_sample_pairs": ("left_obs_id", "size"),
        "n_matching_conditions": ("matched_condition_key", "nunique"),
    }
    for name in present:
        agg[f"mean_{name}"] = (name, "mean")
    drug = (
        frame.groupby(condition_cols, as_index=False)
        .agg(**agg)
        .sort_values(condition_cols)
        .reset_index(drop=True)
    )
    print(
        f"[{analysis}/{variant}] {path.parent.name}: rows={len(frame):,} "
        f"drug_line_times={len(drug):,} metrics={len(present)} level={summary_level}"
    )
    ci = cluster_bca_nested_mean_ci_table(
        drug,
        group_cols=list(group_cols),
        metric_cols={name: f"mean_{name}" for name in present},
        cluster_col="pubchem_cid",
        n_boot=N_BOOT,
        seed=SEED,
        summary_level=summary_level,
    )
    if variant is None:
        # Preserve the original moderated-t output column order.
        ci["analysis"] = analysis
        ci["scale_variant"] = RAW_VARIANT
    else:
        if "scale_variant" not in ci.columns:
            ci.insert(2, "scale_variant", variant)
        ci["analysis"] = analysis
    ci["run_tag"] = path.parent.name
    return ci


def summarize_analysis(
    analysis: str,
    z_score_metrics: list[str],
    computations: Sequence[str] | None = None,
    *, scored_path: Path | None = None,
):
    """Aggregate every layer of one analysis: raw plus the two z layers."""
    selected = set(COMPUTATIONS if computations is None else computations)
    base = REPO / "results/parallel_cross_source" / analysis
    runs = CAPFIX_RUNS[analysis]
    stem = SCORED_STEMS[analysis]
    out = []

    raw_path = scored_path if scored_path is not None else base / runs[RAW_VARIANT] / stem
    if "raw" in selected and raw_path.exists():
        renames = RAW_REFERENCE_NAMES[analysis]
        raw_metrics = {}
        for z_score_metric in z_score_metrics:
            column = raw_peer_name(z_score_metric)
            raw_metrics[renames.get(column, column)] = column
        anchors = [RAW_ANCHOR_METRICS[analysis], *renames.values()]
        missing = [a for a in anchors if a not in raw_metrics]
        if missing:
            raise SystemExit(
                f"{analysis}: the raw layer would publish without {missing}, so it "
                "would not reproduce the notebook reference. Fix raw_peer_name or "
                "RAW_REFERENCE_NAMES rather than adding the anchors back as a "
                "separate call."
            )
        # One call per reference level, on the notebook's own key. Nearly every raw
        # metric shares the layer's level; RAW_REFERENCE_LEVELS peels off the few the
        # reference bootstrapped elsewhere. Those runs predate scale_variant, so it
        # must stay out of the key to seed alike.
        levels = RAW_REFERENCE_LEVELS[analysis]
        by_level: dict[str, dict[str, str]] = {}
        for name, column in raw_metrics.items():
            level = levels.get(name, SUMMARY_LEVELS[(analysis, RAW_VARIANT)])
            by_level.setdefault(level, {})[name] = column
        for level, metrics in by_level.items():
            out.append(
                summarize_layer(
                    raw_path,
                    analysis,
                    RAW_VARIANT,
                    metrics,
                    level,
                    group_cols=("dataset_a", "dataset_b"),
                )
            )
    elif "raw" in selected:
        print(f"[skip] {analysis}/raw: {raw_path.parent.name} not scored yet")

    for variant in (DATASET_VARIANT, CELL_TYPE_VARIANT):
        if VARIANT_COMPUTATIONS[variant] not in selected:
            continue
        path = scored_path if scored_path is not None else base / runs[variant] / stem
        if not path.exists():
            print(f"[skip] {analysis}/{variant}: {path.parent.name} not scored yet")
            continue
        out.append(
            summarize_layer(
                path,
                analysis,
                variant,
                {m: f"{variant}__{m}" for m in z_score_metrics},
                SUMMARY_LEVELS[(analysis, variant)],
            )
        )

    frames = [f for f in out if f is not None]
    return pd.concat(frames, ignore_index=True) if frames else None


def summarize_family(
    path: Path,
    analysis: str,
    columns: dict[str, str],
    summary_level: str,
) -> pd.DataFrame:
    """Summarize moderated-t without adding scale_variant to the bootstrap key."""
    require_metrics(path, analysis, list(columns.values()))
    return summarize_layer(
        path, analysis, None, columns, summary_level,
        group_cols=MODERATED_T_GROUP_COLS,
    )


def summarize_moderated_t(
    analysis: str, run_tag: str, *, scored_path: Path | None = None,
) -> pd.DataFrame | None:
    if scored_path is not None:
        path = scored_path
    else:
        run_dir = REPO / "results/parallel_cross_source" / analysis / run_tag
        path = run_dir / SCORED_STEMS[analysis]
        if not path.exists():
            # Saved runs predating the unified filenames remain readable.
            path = run_dir / MODERATED_T_SCORED_STEMS[analysis]
    if not path.exists():
        print(f"[skip] {TABLES[analysis]} ({analysis}): {run_tag} not scored yet")
        return None

    families: list[tuple[dict[str, str], str]] = [
        (aliases, level) for level, aliases in LEGACY_ALIASES[analysis].items()
    ]
    peers = PEER_METRICS[analysis]
    families.append(({metric: metric for metric in peers}, PEER_SUMMARY_LEVELS[analysis]))

    expected = [column for columns, _ in families for column in columns.values()]
    require_metrics(path, analysis, expected)
    return pd.concat(
        [summarize_family(path, analysis, columns, level) for columns, level in families],
        ignore_index=True,
    )


def summarize_run(metrics_path: Path, *, analysis: str, computations: Sequence[str]) -> Path:
    """Summarize this run's selected computations beside its scored file."""
    frames = []
    if any(name != "moderated-t" for name in computations):
        metrics = Z_SCORE_DEG_METRICS if analysis == "deg" else Z_SCORE_SIGNATURE_METRICS
        frames.append(summarize_analysis(analysis, metrics, computations, scored_path=metrics_path))
    if "moderated-t" in computations:
        frames.append(summarize_moderated_t(
            analysis, metrics_path.parent.name, scored_path=metrics_path,
        ))
    table = pd.concat([frame for frame in frames if frame is not None], ignore_index=True)
    output_path = metrics_path.with_name(
        metrics_path.stem.removesuffix("_scored_metrics") + "_cluster_bca_ci.tsv"
    )
    _write_summary(table, output_path)
    return output_path


def verify(table: pd.DataFrame) -> int:
    """Check the legacy family against the published CI tables, bound for bound."""
    worst = 0.0
    checked = missing = 0
    for analysis, reference_path in REFERENCE_CI.items():
        if not reference_path.exists():
            print(f"[verify] {analysis}: reference {reference_path.name} absent, skipped")
            continue
        reference = pd.read_csv(reference_path, sep="\t")
        legacy = {
            name for level in LEGACY_ALIASES[analysis] for name in LEGACY_ALIASES[analysis][level]
        }
        ours = table[(table["analysis"] == analysis) & (table["metric"].isin(legacy))]
        for _, row in ours.iterrows():
            match = reference[
                (reference["dataset_a"] == row["dataset_a"])
                & (reference["dataset_b"] == row["dataset_b"])
                & (reference["summary_level"] == row["summary_level"])
                & (reference["metric"] == row["metric"])
            ]
            if match.empty:
                missing += 1
                continue
            reference_row = match.iloc[0]
            for column in ("mean", "ci_low", "ci_high"):
                worst = max(worst, abs(float(row[column]) - float(reference_row[column])))
            checked += 1
    print(
        f"\n[verify] legacy rows checked={checked} unmatched={missing} "
        f"max|ours-published|={worst:.2e}"
    )
    if checked and worst < 1e-12:
        print("[verify] legacy family reproduces the published panels bit for bit")
        return 0
    if not checked:
        print("[verify] nothing could be compared")
        return 1
    print("[verify] FAILED: seed lineage differs (summary_level, names, or group_cols)")
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--compute", action="append", metavar="NAME",
        help="Repeat or comma-separate: " + ", ".join(COMPUTATIONS) + ". Default: all four.",
    )
    parser.add_argument(
        "--run-tag", default="moderated_t_256",
        help="Moderated-t scoring run tag. Raw/z-score runs retain their capfix tags.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_PATH.parent,
        help="Directory for the two existing summary filenames.",
    )
    parser.add_argument(
        "--out", type=Path,
        help="Single output file; select either raw/z-score computations or moderated-t.",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Check moderated-t legacy means and CI bounds against published t_stats panels.",
    )
    return parser


def _write_summary(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)
    print(f"\nwrote {path}  rows={len(table):,}")
    print(
        table.groupby(["analysis", "scale_variant", "summary_level"], sort=False)
        .size().rename("rows").to_string()
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    requested = (
        [name.strip() for value in args.compute for name in value.split(",") if name.strip()]
        if args.compute is not None else list(COMPUTATIONS)
    )
    unknown = sorted(set(requested) - set(COMPUTATIONS))
    if not requested or unknown:
        parser.error("--compute must select from: " + ", ".join(COMPUTATIONS))
    # Fixed processing order keeps output row order independent of CLI ordering.
    selected = [name for name in COMPUTATIONS if name in requested]
    include_moderated_t = "moderated-t" in selected
    include_logfc = any(name != "moderated-t" for name in selected)
    if args.out is not None and include_moderated_t and include_logfc:
        parser.error("--out requires either raw/z-score computations or moderated-t; use --out-dir for both")
    if args.verify and not include_moderated_t:
        parser.error("--verify requires moderated-t in --compute")

    if include_logfc:
        frames = [
            frame for frame in (
                summarize_analysis("deg", Z_SCORE_DEG_METRICS, selected),
                summarize_analysis("signature", Z_SCORE_SIGNATURE_METRICS, selected),
            ) if frame is not None
        ]
        if not frames:
            raise SystemExit("No raw/z-score layers could be summarized; nothing scored yet?")
        _write_summary(
            pd.concat(frames, ignore_index=True),
            args.out if args.out is not None else args.out_dir / OUT_PATH.name,
        )

    if include_moderated_t:
        frames = [
            frame for frame in (
                summarize_moderated_t(analysis, args.run_tag)
                for analysis in ("deg", "signature")
            ) if frame is not None
        ]
        if not frames:
            raise SystemExit(f"Nothing scored under run tag {args.run_tag!r}.")
        table = pd.concat(frames, ignore_index=True)
        _write_summary(
            table, args.out if args.out is not None else args.out_dir / MODERATED_T_OUT_PATH.name,
        )
        return verify(table) if args.verify else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
