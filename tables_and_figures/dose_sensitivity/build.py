#!/usr/bin/env python3
"""Build dose-sensitivity summaries, the agreement figure and both paper tables."""
from __future__ import annotations

import argparse
from decimal import Decimal, ROUND_HALF_UP
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table

MIN_CONTEXT_SHARED_DRUGS = 10

CONTEXT_COLUMNS = ["dataset_a", "dataset_b", "cell_type", "time_key"]

PAIR_COLUMNS = ["dataset_a", "dataset_b"]

DRUG_LINE_TIME_COLUMNS = [*CONTEXT_COLUMNS, "pubchem_cid"]

METRICS = [
    "observed_deg_lfc_spearman_sym_p05",
    "baseline_pair_deg_lfc_spearman_p05",
    "delta_vs_baseline_pair_deg_lfc_spearman_p05",
    "observed_direction_agreement_p05",
    "baseline_pair_direction_agreement_p05",
    "delta_vs_baseline_pair_direction_agreement_p05",
]

SPECS = [
    ("exact", 0, 1.0, 0.0, False),
    ("2x", 1, 2.0, float(np.log10(2.0)), False),
    ("3x", 2, 3.0, float(np.log10(3.0)), False),
    ("10x_reference", 3, 10.0, 1.0, True),
]

DISPLAY_LABELS = {
    "l1000_phase1": "L1000 Phase I",
    "l1000_phase2": "L1000 Phase II",
    "tahoe": "Tahoe-100M",
    "cigs_mce": "CIGS-MCE",
    "cigs_tcm": "CIGS-TCM",
    "sciplex": "sci-Plex",
}

EXACT_MATCH_TOLERANCE = 1e-12

BOOTSTRAP_ITERATIONS = 2000

BOOTSTRAP_RANDOM_SEED = 20260505

JITTER_RANDOM_SEED = 20260505

MISMATCH_BIN_LABELS = ["exact", "(exact, 2x]", "(2x, 3x]", "(3x, 10x]"]

MISMATCH_BIN_UPPER_EDGES = [
    EXACT_MATCH_TOLERANCE,
    float(np.log10(2.0)),
    float(np.log10(3.0)),
    1.0,
]

MATCHED_CONDITION_GROUP_COLUMNS = [
    "dataset_a",
    "dataset_b",
    "cell_type",
    "time_key",
    "pubchem_cid",
    "left_dose_key",
    "right_dose_key",
    "matched_condition_key",
]

PANEL_SPECS = [
    ("mean_observed_spearman_logfc", "observed_spearman_logfc", "logFC"),
    ("mean_observed_spearman_t", "observed_spearman_t", "t"),
]

# String keys retain the original signature grouping and bootstrap cluster order.
KEY_DTYPES = {column: str for column in [
    *MATCHED_CONDITION_GROUP_COLUMNS, "left_obs_id", "right_obs_id",
]}
MATCH_COLUMNS = list(dict.fromkeys([
    *MATCHED_CONDITION_GROUP_COLUMNS, "left_obs_id", "right_obs_id",
    "abs_delta_log10_dose",
]))


def read_metrics(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", usecols=columns, dtype=KEY_DTYPES)
    if frame.empty:
        raise ValueError(f"No rows in {path}")
    doses = pd.to_numeric(frame["abs_delta_log10_dose"], errors="coerce")
    if not np.isfinite(doses).all() or (doses < 0).any() or (doses > 1 + 1e-12).any():
        raise ValueError(f"Expected finite dose mismatches between 1x and 10x in {path}")
    frame["abs_delta_log10_dose"] = doses
    return frame


def read_matched_pairs(run_dir: Path) -> pd.DataFrame:
    """Read all planned pairs, including rows absent from the scored TSV."""
    manifest = pd.read_csv(run_dir / "task_manifest.tsv", sep="\t")
    if manifest.empty or manifest["task_id"].duplicated().any():
        raise ValueError("Expected a nonempty task manifest with unique task IDs")
    frames = []
    for task in manifest.sort_values("task_id").itertuples(index=False):
        # Manifests may contain absolute paths from a different checkout.
        path = run_dir / "task_inputs" / Path(task.input_file).name
        frame = read_metrics(path, MATCH_COLUMNS)
        if len(frame) != int(task.n_input_rows):
            raise ValueError(f"Unexpected matched-pair count in {path}")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)

def eligible_contexts(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame.groupby(CONTEXT_COLUMNS, as_index=False).agg(
        n_context_matching_drugs=("pubchem_cid", "nunique")
    )
    return counts.loc[
        counts["n_context_matching_drugs"] >= MIN_CONTEXT_SHARED_DRUGS, CONTEXT_COLUMNS
    ]

def qualify(frame: pd.DataFrame, max_abs_delta: float, contexts: pd.DataFrame) -> pd.DataFrame:
    candidates = frame.loc[frame["abs_delta_log10_dose"] <= max_abs_delta + 1e-12]
    return candidates.merge(contexts, on=CONTEXT_COLUMNS, how="inner")

def summarize(matched: pd.DataFrame, metrics: pd.DataFrame, spec) -> pd.DataFrame:
    label, order, max_fold, max_abs_delta, is_reference = spec

    contexts = eligible_contexts(matched.loc[matched["abs_delta_log10_dose"] <= max_abs_delta + 1e-12])
    matched_q = qualify(matched, max_abs_delta, contexts)
    metrics_q = qualify(metrics, max_abs_delta, contexts)

    counts = matched_q.groupby(PAIR_COLUMNS, as_index=False).agg(
        n_matched_sample_pairs=("left_obs_id", "size"),
        n_matching_conditions=("matched_condition_key", "nunique"),
        n_matching_drugs=("pubchem_cid", "nunique"),
        n_matching_lines=("cell_type", "nunique"),
    )
    context_counts = (
        contexts.groupby(PAIR_COLUMNS, as_index=False)
        .size()
        .rename(columns={"size": "n_eligible_contexts"})
    )

    drug_line_time = metrics_q.groupby(DRUG_LINE_TIME_COLUMNS, as_index=False)[METRICS].mean()
    scored = metrics_q.groupby(PAIR_COLUMNS, as_index=False).agg(
        n_scored_sample_pairs=("left_obs_id", "size")
    )
    scored_groups = (
        drug_line_time.groupby(PAIR_COLUMNS, as_index=False)
        .size()
        .rename(columns={"size": "n_scored_drug_line_times"})
    )
    means = drug_line_time.groupby(PAIR_COLUMNS, as_index=False)[METRICS].mean()
    means = means.rename(columns={metric: f"mean_{metric}" for metric in METRICS})

    summary = (
        counts.merge(context_counts, on=PAIR_COLUMNS, how="left")
        .merge(scored, on=PAIR_COLUMNS, how="left")
        .merge(scored_groups, on=PAIR_COLUMNS, how="left")
        .merge(means, on=PAIR_COLUMNS, how="left")
    )
    summary.insert(0, "dose_threshold", label)
    summary.insert(1, "threshold_order", order)
    summary.insert(2, "max_fold_difference", max_fold)
    summary.insert(3, "max_abs_delta_log10_dose", max_abs_delta)
    summary.insert(4, "is_reference", is_reference)
    return summary

def pretty_label(dataset_name: str) -> str:
    return DISPLAY_LABELS.get(dataset_name, dataset_name)

def dataset_pair_label(frame: pd.DataFrame) -> pd.Series:
    return frame["dataset_a"].map(pretty_label) + " vs " + frame["dataset_b"].map(pretty_label)

def assign_mismatch_bin(abs_delta_log10_dose: pd.Series) -> pd.Categorical:
    values = pd.to_numeric(abs_delta_log10_dose, errors="coerce").to_numpy(dtype=float)
    assigned = np.full(values.shape, None, dtype=object)
    lower_edge = -np.inf
    for upper_edge, label in zip(MISMATCH_BIN_UPPER_EDGES, MISMATCH_BIN_LABELS):
        mask = (values > lower_edge) & (values <= upper_edge + 1e-12)
        assigned[mask] = label
        # Advance past the tolerance so values admitted to this bin are not
        # re-admitted to the next one and overwritten.
        lower_edge = upper_edge + 1e-12
    return pd.Categorical(assigned, categories=MISMATCH_BIN_LABELS, ordered=True)


def build_deg_summaries(matched: pd.DataFrame, metrics: pd.DataFrame):
    thresholds = pd.concat(
        [summarize(matched, metrics, spec) for spec in SPECS], ignore_index=True,
    ).sort_values(["threshold_order", *PAIR_COLUMNS]).reset_index(drop=True)
    retention = thresholds.pivot(
        index=PAIR_COLUMNS, columns="dose_threshold", values="n_matched_sample_pairs",
    ).reindex(columns=[spec[0] for spec in SPECS]).fillna(0).astype(int)
    retention = retention.rename(columns={"10x_reference": "10x"}).reset_index()
    quantiles = (
        matched.assign(fold=np.power(10.0, matched["abs_delta_log10_dose"]))
        .groupby(PAIR_COLUMNS).agg(
            n=("fold", "size"), min=("fold", "min"), median=("fold", "median"),
            q95=("fold", lambda values: values.quantile(0.95)), max=("fold", "max"),
        ).reset_index()
    )
    mismatch = quantiles.merge(retention, on=PAIR_COLUMNS, how="left")
    mismatch["diff"] = mismatch["10x"] - mismatch["3x"]
    return mismatch.sort_values("10x", ascending=False).reset_index(drop=True), thresholds



def build_signature_summary(metrics: pd.DataFrame):
    conditions = (
        metrics.assign(dose_fold_difference=np.power(10.0, metrics["abs_delta_log10_dose"]))
        .groupby(MATCHED_CONDITION_GROUP_COLUMNS, as_index=False).agg(
            n_matched_sample_pairs=("left_obs_id", "size"),
            abs_delta_log10_dose=("abs_delta_log10_dose", "mean"),
            dose_fold_difference=("dose_fold_difference", "mean"),
            mean_pair_mean_abs_t=("pair_mean_abs_t", "mean"),
            mean_observed_spearman_logfc=("observed_spearman_logfc", "mean"),
            mean_observed_spearman_t=("observed_spearman_t", "mean"),
        ).sort_values(MATCHED_CONDITION_GROUP_COLUMNS).reset_index(drop=True)
    )
    conditions["mismatch_bin"] = assign_mismatch_bin(conditions["abs_delta_log10_dose"])
    conditions["dataset_pair"] = dataset_pair_label(conditions)
    # These labels and value columns are part of the historical bootstrap seed.
    ci = cluster_bca_nested_mean_ci_table(
        conditions.assign(mismatch_bin=lambda frame: frame["mismatch_bin"].astype(str)),
        group_cols=["mismatch_bin"],
        metric_cols={metric: value_col for value_col, metric, _ in PANEL_SPECS},
        cluster_col="pubchem_cid", n_boot=BOOTSTRAP_ITERATIONS,
        seed=BOOTSTRAP_RANDOM_SEED, summary_level="mismatch_bin_pooled",
    )
    return conditions, ci

DOSE_PAIR_ORDER = [
    ("l1000_phase1", "l1000_phase2"),
    ("cigs_mce", "cigs_tcm"),
    ("l1000_phase1", "sciplex"),
    ("l1000_phase2", "sciplex"),
    ("l1000_phase1", "cigs_mce"),
    ("l1000_phase2", "cigs_mce"),
    ("l1000_phase1", "tahoe"),
    ("l1000_phase2", "tahoe"),
    ("tahoe", "sciplex"),
]



def _dose_latex_rows(mismatch):
    order = {pair: index for index, pair in enumerate(DOSE_PAIR_ORDER)}
    return sorted(
        mismatch.to_dict("records"),
        key=lambda row: (
            order.get((row["dataset_a"], row["dataset_b"]), len(order)),
            row["dataset_a"],
            row["dataset_b"],
        ),
    )


def _dose_pair_label(row):
    return "--".join(
        DISPLAY_LABELS.get(str(row[column]), str(row[column]).replace("_", r"\_"))
        for column in ("dataset_a", "dataset_b")
    )


def _dose_number(value, places, *, signed=False):
    if not math.isfinite(float(value)):
        return "n/a"
    number = Decimal(repr(float(value))).quantize(
        Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP
    )
    return f"{number:+}" if signed else str(number)


def _dose_count(value):
    return f"{int(value):,}".replace(",", "{,}")


def render_mismatch_latex(mismatch, min_context_shared_drugs=10):
    """Render fold-difference quantiles and threshold retention counts."""
    lines = [
        r"\begin{table*}[t]",
        r"\refstepcounter{table}",
        r"\label{tab:dose_mismatch}",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Fold difference} &",
        r"\multicolumn{5}{c}{\# Matched pairs by dose threshold} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-10}",
        r"Dataset pair & min & median & q95 & max &",
        r"exact & 2$\times$ & 3$\times$ & 10$\times$ & $\#_{10\times}$--$\#_{3\times}$ \\",
        r"\midrule",
    ]
    for row in _dose_latex_rows(mismatch):
        lines.extend([
            _dose_pair_label(row) + " &",
            " & ".join(_dose_number(row[column], 2) for column in ("min", "median", "q95", "max")) + " &",
            " & ".join(_dose_count(row[column]) for column in ("exact", "2x", "3x", "10x", "diff")) + r" \\",
        ])
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def render_threshold_latex(mismatch, threshold_metrics):
    """Render observed / source-centroid delta scores at selected thresholds."""
    rows = _dose_latex_rows(mismatch)
    scores = {
        (row["dataset_a"], row["dataset_b"], row["dose_threshold"]): row
        for row in threshold_metrics.to_dict("records")
    }
    lines = [
        r"\begin{table*}[t]",
        r"\refstepcounter{table}",
        r"\label{tab:dose_threshold_sensitivity}",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.4cm}@{\hspace{0.8pt}}cccccc}",
        r"\toprule",
        r"& \multicolumn{3}{c}{DEG Spearman} & \multicolumn{3}{c}{Direction} \\",
        r"\cmidrule(lr){2-4} \cmidrule(lr){5-7}",
        r"Dataset pair & exact & 2$\times$ & 10$\times$ & exact & 2$\times$ & 10$\times$ \\",
        r"\midrule",
    ]
    metrics = [
        ("mean_observed_deg_lfc_spearman_sym_p05", "mean_delta_vs_baseline_pair_deg_lfc_spearman_p05"),
        ("mean_observed_direction_agreement_p05", "mean_delta_vs_baseline_pair_direction_agreement_p05"),
    ]
    for row in rows:
        cells = []
        for observed_column, delta_column in metrics:
            for threshold in ("exact", "2x", "10x_reference"):
                score = scores.get((row["dataset_a"], row["dataset_b"], threshold))
                if score is None or not all(math.isfinite(float(score[column])) for column in (observed_column, delta_column)):
                    cells.append("n/a")
                else:
                    cells.append(
                        _dose_number(score[observed_column], 3)
                        + " / "
                        + _dose_number(score[delta_column], 3, signed=True)
                    )
        lines.extend([
            _dose_pair_label(row),
            "  & " + " & ".join(cells[:3]),
            "  & " + " & ".join(cells[3:]) + r" \\",
        ])
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines) + "\n"


def plot_agreement(matched_condition_dose_metrics: pd.DataFrame, pooled_ci: pd.DataFrame, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_theme(style="whitegrid")
    pair_order = list(dict.fromkeys(matched_condition_dose_metrics["dataset_pair"]))
    palette = sns.color_palette(n_colors=len(pair_order))
    pair_colors = dict(zip(pair_order, palette))
    bin_positions = {label: position for position, label in enumerate(MISMATCH_BIN_LABELS)}

    bin_counts = (
        matched_condition_dose_metrics["mismatch_bin"].value_counts().reindex(MISMATCH_BIN_LABELS).fillna(0).astype(int)
    )

    np.random.seed(JITTER_RANDOM_SEED)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharex=True, constrained_layout=True)

    for axis, (value_col, metric_name, short_title) in zip(axes, PANEL_SPECS):
        axis.axhline(0.0, color="0.45", linewidth=0.8, alpha=0.7)
        sns.violinplot(
            data=matched_condition_dose_metrics,
            x="mismatch_bin",
            y=value_col,
            hue="dataset_pair",
            hue_order=pair_order,
            order=MISMATCH_BIN_LABELS,
            palette=pair_colors,
            dodge=True,
            density_norm="width",
            inner=None,
            cut=0,
            linewidth=0.3,
            saturation=1.0,
            alpha=0.30,
            legend=False,
            ax=axis,
        )
        sns.stripplot(
            data=matched_condition_dose_metrics,
            x="mismatch_bin",
            y=value_col,
            hue="dataset_pair",
            hue_order=pair_order,
            order=MISMATCH_BIN_LABELS,
            palette=pair_colors,
            dodge=True,
            size=3.6,
            alpha=0.45,
            linewidth=0.25,
            edgecolor="w",
            jitter=0.30,
            legend=False,
            ax=axis,
        )

        metric_ci = pooled_ci.loc[pooled_ci["metric"] == metric_name].copy()
        metric_ci["bin_position"] = metric_ci["mismatch_bin"].map(bin_positions)
        metric_ci = metric_ci.dropna(subset=["bin_position"]).sort_values("bin_position")
        mean_values = metric_ci["mean"].to_numpy(dtype=float)
        lower_errors = np.clip(mean_values - metric_ci["ci_low"].to_numpy(dtype=float), 0.0, None)
        upper_errors = np.clip(metric_ci["ci_high"].to_numpy(dtype=float) - mean_values, 0.0, None)
        axis.errorbar(
            metric_ci["bin_position"].to_numpy(dtype=float),
            mean_values,
            yerr=np.vstack([lower_errors, upper_errors]),
            marker="o",
            markersize=6,
            markerfacecolor="white",
            markeredgewidth=1.4,
            linewidth=2.0,
            linestyle="--",
            capsize=4,
            color="black",
            zorder=5,
        )

        axis.set_title(
            f"Dose mismatch vs observed {short_title} correlation "
            f"(n = {len(matched_condition_dose_metrics):,})"
        )
        axis.set_xlabel("Dose mismatch between matched conditions")
        axis.set_ylabel(f"Observed cross-dataset Spearman({short_title})")
        axis.set_ylim(-0.8, 1.0)
        axis.set_xlim(-0.5, len(MISMATCH_BIN_LABELS) - 0.5)
        axis.grid(False)
        for boundary in np.arange(0.5, len(MISMATCH_BIN_LABELS) - 1 + 0.5, 1.0):
            axis.axvline(boundary, color="0.75", linewidth=0.8, zorder=0)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
        for position, label in enumerate(MISMATCH_BIN_LABELS):
            axis.annotate(
                f"n={bin_counts[label]:,}",
                (position, -0.76),
                ha="center",
                fontsize=8.0,
                color="0.35",
            )

    pair_handles = [
        Line2D([], [], marker="o", linestyle="", markersize=5, color=pair_colors[pair], label=pair)
        for pair in pair_order
    ]
    summary_handles = [
        Line2D(
            [],
            [],
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
            color="black",
            linewidth=2.0,
            linestyle="--",
            markersize=6,
            label="bin mean, 95% BCa CI clustered by compound",
        )
    ]
    pair_legend = fig.legend(
        handles=pair_handles,
        title="Dataset pair",
        loc="upper left",
        ncol=1,
        bbox_to_anchor=(1.0, 0.98),
    )
    fig.add_artist(pair_legend)
    fig.legend(
        handles=summary_handles,
        loc="upper left",
        ncol=1,
        fontsize=9,
        bbox_to_anchor=(1.0, 0.30),
    )
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deg-run", type=Path,
        default=REPO_ROOT / "results/parallel_cross_source/deg/raw_256",
        help="Raw DEG run containing scores, task_manifest.tsv and task_inputs/.",
    )
    parser.add_argument(
        "--signature-run", type=Path,
        default=REPO_ROOT / "results/parallel_cross_source/signature/raw_moderated_t_256",
        help="Signature run containing signature_scored_metrics.tsv with raw and "
             "moderated-t scores (--compute raw,moderated-t).",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/default")
    args = parser.parse_args(argv)
    required = [args.deg_run / "task_manifest.tsv", args.deg_run / "deg_scored_metrics.tsv",
                args.signature_run / "signature_scored_metrics.tsv"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("Missing current-run inputs: " + ", ".join(missing))
    matched = read_matched_pairs(args.deg_run)
    deg = read_metrics(args.deg_run / "deg_scored_metrics.tsv", [*MATCH_COLUMNS, *METRICS])
    # The figure needs both logFC and t: use --compute raw,moderated-t.
    signature = read_metrics(args.signature_run / "signature_scored_metrics.tsv", [
        *MATCH_COLUMNS, "pair_mean_abs_t", "observed_spearman_logfc", "observed_spearman_t",
    ])
    print(f"Read {len(matched):,} planned pairs, {len(deg):,} DEG scores and "
          f"{len(signature):,} signature scores.", flush=True)
    mismatch, thresholds = build_deg_summaries(matched, deg)
    conditions, ci = build_signature_summary(signature)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("dose_mismatch.tsv", mismatch),
                        ("dose_threshold_metrics.tsv", thresholds),
                        ("dose_agreement_ci.tsv", ci)]:
        frame.to_csv(args.out_dir / name, sep="\t", index=False)
    (args.out_dir / "dose_mismatch.tex").write_text(render_mismatch_latex(mismatch))
    (args.out_dir / "dose_threshold_sensitivity.tex").write_text(
        render_threshold_latex(mismatch, thresholds)
    )
    plot_agreement(conditions, ci, args.out_dir / "dose_agreement.pdf")
    print(f"Wrote three TSVs, two LaTeX tables and the figure to {args.out_dir}")


if __name__ == "__main__":
    main()
