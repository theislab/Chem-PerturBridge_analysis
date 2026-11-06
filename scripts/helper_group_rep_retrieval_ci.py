"""Tables 12/13 CIs, retaining the strict, reseeded-peer and reviewer hash lineages.

Historical metric names, group order and summary levels below are bootstrap inputs.
Computation/family labels are attached only after bootstrapping. Input row order is
also significant: compounds receive bootstrap positions in first-appearance order.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

if __package__:
    from .cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table
else:
    from cluster_bootstrap_ci import cluster_bca_nested_mean_ci_table


BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_RANDOM_SEED = 20260505
DATASET_SCALE = "per_gene_population_zscore_dataset"
CELL_TYPE_SCALE = "per_gene_population_zscore_dataset_cell_type"
COMPUTATIONS = {
    "raw": ("logFC", "raw"),
    "direction": ("logFC_direction", "raw"),
    "moderated-t": ("moderated_t", "raw"),
    "z-score-dataset": ("logFC", DATASET_SCALE),
    "z-score-dataset-cell-type": ("logFC", CELL_TYPE_SCALE),
}
SIMILARITIES = ("negative_l2", "cosine", "spearman")
GROUP_COLS = ["dataset_a", "dataset_b", "representation", "retrieval_variant"]
INNER_COLS = ["direction", "cell_type", "time_key"]
IDENTITY_COLS = [
    *GROUP_COLS, *INNER_COLS, "query_pubchem_cid", "similarity_metric", "scale_variant",
]
OBSERVED_RANK = "observed_normalized_best_positive_rank"
STRICT_METRICS = {
    OBSERVED_RANK: "normalized_best_positive_rank",
    "baseline_normalized_best_positive_rank": "baseline_normalized_best_positive_rank",
    "delta_vs_baseline_normalized_best_positive_rank": "delta_vs_baseline_normalized_best_positive_rank",
}

# Saved P/PT metric set from the original reviewer notebooks and reseeder. Explicit
# names keep strict-only columns and diagnostics out of the peer family.
NOTEBOOK_PEER_METRICS = (
    "delta_observed_vs_source_centroid",
    "delta_observed_vs_source_individual_mean",
    "delta_observed_vs_source_individual_rank",
    "delta_observed_vs_target_centroid",
    "delta_observed_vs_target_decoy_null_auroc",
    "delta_observed_vs_target_decoy_null_rank",
    "delta_observed_vs_target_decoy_null_recall_at_1",
    "delta_observed_vs_target_individual_mean",
    "delta_observed_vs_target_individual_rank",
    "observed_auroc",
    "observed_best_positive_similarity",
    "observed_mean_positive_similarity",
    OBSERVED_RANK,
    "observed_recall_at_1",
    "observed_sd_positive_similarity",
    "source_centroid_similarity",
    "source_individual_auroc",
    "source_individual_corrected_percentile",
    "source_individual_empirical_p_upper",
    "source_individual_fraction_below_observed",
    "source_individual_mean_similarity",
    "source_individual_normalized_best_positive_rank",
    "source_individual_recall_at_1",
    "source_individual_sd_similarity",
    "target_centroid_similarity",
    "target_decoy_null_auroc",
    "target_decoy_null_normalized_best_positive_rank",
    "target_decoy_null_recall_at_1",
    "target_individual_auroc",
    "target_individual_corrected_percentile",
    "target_individual_empirical_p_upper",
    "target_individual_fraction_below_observed",
    "target_individual_mean_similarity",
    "target_individual_normalized_best_positive_rank",
    "target_individual_recall_at_1",
    "target_individual_sd_similarity",
)

# Verbatim metric order from elmella 46a72c5, build_retrieval_ci().
REVIEWER_PEER_METRICS = (
    OBSERVED_RANK,
    "observed_recall_at_1",
    "observed_auroc",
    "observed_best_positive_similarity",
    "random_normalized_best_positive_rank",
    "random_recall_at_1",
    "random_auroc",
    "delta_vs_random_normalized_best_positive_rank",
    "source_individual_n_peers",
    "source_individual_mean_similarity",
    "source_individual_sd_similarity",
    "source_individual_n_below_observed",
    "source_individual_fraction_below_observed",
    "source_individual_corrected_percentile",
    "target_individual_n_peers",
    "target_individual_mean_similarity",
    "target_individual_sd_similarity",
    "target_individual_n_below_observed",
    "target_individual_fraction_below_observed",
    "target_individual_corrected_percentile",
    "source_centroid_similarity",
    "source_centroid_normalized_rank",
    "source_centroid_recall_at_1",
    "source_centroid_auroc",
    "target_centroid_similarity",
    "target_centroid_normalized_rank",
    "target_centroid_recall_at_1",
    "target_centroid_auroc",
    "source_peer_normalized_rank",
    "target_peer_normalized_rank",
    "delta_vs_source_peer_normalized_rank",
    "delta_vs_target_peer_normalized_rank",
    "target_decoy_null_normalized_best_positive_rank",
    "delta_vs_target_decoy_null_normalized_best_positive_rank",
)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Retrieval scores are missing columns: {missing}")


def _canonical_cluster_ids(values: pd.Series) -> pd.Series:
    """Retain the original reseeder's raw/sign/t compound-ID handling."""
    numeric = pd.to_numeric(values, errors="coerce")
    integral = numeric.notna() & (numeric % 1 == 0)
    canonical = values.astype("string").str.strip()
    canonical.loc[integral] = numeric.loc[integral].astype("int64").astype("string")
    return canonical.fillna("")


def _bootstrap(frame, metrics, *, group_cols, summary_level, n_boot):
    return cluster_bca_nested_mean_ci_table(
        frame, group_cols=group_cols, metric_cols=metrics,
        cluster_col="query_pubchem_cid", inner_cols=INNER_COLS,
        outer_cols=["direction"], n_boot=n_boot, seed=BOOTSTRAP_RANDOM_SEED,
        summary_level=summary_level,
    )


def reviewer_summary_level(similarity: str, scale: str) -> str:
    if similarity == "negative_l2":
        return "reviewer_l2_retrieval_dataset_pair"
    if scale == CELL_TYPE_SCALE:
        return "reviewer_dataset_cell_type_retrieval_dataset_pair"
    return "reviewer_minimal_retrieval_dataset_pair"


def _family_rows(frame: pd.DataFrame, family: str) -> pd.DataFrame:
    marker = f"{family}_available"
    if marker in frame:
        present = frame[marker].eq(True) | frame[marker].astype("string").eq("True")
        frame = frame.loc[present.fillna(False)]
    order = f"{family}_row_order"
    if order in frame:
        frame = frame.sort_values(order, kind="mergesort")
    return frame.copy()


def summarize_frame(
    frame: pd.DataFrame, *, computations: Sequence[str],
    n_boot: int = BOOTSTRAP_ITERATIONS,
) -> pd.DataFrame:
    """Summarize selected representations without sorting or preaggregating rows.

    ``strict_*`` columns retain the notebook's distinct gene filtering/centroid.
    Peer columns use notebook names for raw/sign/t and reviewer names for z-scores.
    Optional family presence/order columns preserve each input geometry separately.
"""
    unknown = sorted(set(computations) - set(COMPUTATIONS))
    if not computations or unknown:
        raise ValueError(f"Select computations from {tuple(COMPUTATIONS)}; unknown={unknown}")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    _require_columns(frame, IDENTITY_COLS)
    if frame.empty:
        raise ValueError("No retrieval scores to summarize")
    tables = []
    for computation, (representation, scale) in COMPUTATIONS.items():
        if computation not in computations:
            continue
        selected = frame.loc[
            frame["representation"].eq(representation)
            & frame["scale_variant"].eq(scale)
            & frame["retrieval_variant"].eq("strict_matched_condition")
        ].copy()
        if selected.empty:
            raise ValueError(f"No strict retrieval scores for {computation}")
        invalid = set(selected["similarity_metric"].dropna()) - set(SIMILARITIES)
        if invalid:
            raise ValueError(f"Unknown similarities: {sorted(invalid)}")
        is_zscore = scale != "raw"
        if not is_zscore:
            selected["query_pubchem_cid"] = _canonical_cluster_ids(selected["query_pubchem_cid"])
        for similarity in SIMILARITIES:
            scored = selected.loc[selected["similarity_metric"].eq(similarity)]
            if scored.empty:
                continue
            strict = _family_rows(scored, "strict")
            peers = _family_rows(scored, "peer")
            families = []
            if not strict.empty:
                for metric, column in STRICT_METRICS.items():
                    _require_columns(strict, [f"strict_{metric}"])
                    strict[column] = strict[f"strict_{metric}"]
                if is_zscore:
                    strict["representation"] = (
                        "logFC_zscore_dataset" if scale == DATASET_SCALE
                        else "logFC_zscore_dataset_cell_type"
                    )
                strict_ci = _bootstrap(
                    strict, STRICT_METRICS, group_cols=GROUP_COLS,
                    summary_level="dataset_pair", n_boot=n_boot,
                )
                families.append(("strict", strict_ci))
            if not peers.empty:
                _require_columns(peers, [
                    OBSERVED_RANK, "source_individual_corrected_percentile",
                    "target_individual_corrected_percentile",
                ])
                if is_zscore:
                    metrics = [name for name in REVIEWER_PEER_METRICS if name in peers]
                    peer_ci = _bootstrap(
                        peers, metrics, group_cols=[*GROUP_COLS, "similarity_metric", "scale_variant"],
                        summary_level=reviewer_summary_level(similarity, scale), n_boot=n_boot,
                    )
                else:
                    metrics = {name: name for name in peers if name in NOTEBOOK_PEER_METRICS}
                    metrics[OBSERVED_RANK] = "normalized_best_positive_rank"
                    peers["normalized_best_positive_rank"] = peers[OBSERVED_RANK]
                    peer_ci = _bootstrap(
                        peers, metrics, group_cols=GROUP_COLS,
                        summary_level="dataset_pair", n_boot=n_boot,
                    )
                families.append(("peer", peer_ci))
            for family, table in families:
                table["similarity_metric"] = similarity
                table["scale_variant"] = scale
                table["computation"] = computation
                table["family"] = family
                tables.append(table)
    if not tables:
        raise ValueError("No supported retrieval similarities to summarize")
    return pd.concat(tables, ignore_index=True)


def summarize_run(
    metrics_path: Path, *, computations: Sequence[str],
    n_boot: int = BOOTSTRAP_ITERATIONS,
) -> Path:
    """Read scores with historical string identities and write adjacent CIs."""
    metrics_path = Path(metrics_path)
    header = pd.read_csv(metrics_path, sep="\t", nrows=0).columns
    frame = pd.read_csv(
        metrics_path, sep="\t", low_memory=False,
        dtype={column: "string" for column in IDENTITY_COLS if column in header},
    )
    table = summarize_frame(frame, computations=computations, n_boot=n_boot)
    output = metrics_path.with_name(
        metrics_path.stem.removesuffix("_scored_metrics") + "_cluster_bca_ci.tsv"
    )
    table.to_csv(output, sep="\t", index=False)
    print(f"[retrieval CI] wrote {len(table):,} rows -> {output}")
    return output
