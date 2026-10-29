"""Pure and process-local scoring helpers for the parallel cross-source commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from scripts import cross_source_core
from scripts.cross_source_strata import SignatureStratum
from scripts.peer_baselines import (
    empty_peer_score_summary,
    select_peer_indices,
    spearman_against_peers,
    summarize_peer_scores,
)


ADJ_PVALUE_LAYER_PREFERENCES = (
    "adj.P.Value.within_one_contrast",
    "adj.P.Value.across_all_contrasts",
)
DEG_P_THRESHOLD = 0.05
DEG_ABS_LOGFC_THRESHOLD = 0.2
DE_OVERLAP_K_VALUES = (50, 100, 200)
DEG_DEFINITIONS = {
    "p05": {"require_abs_logfc": False},
    "p05_lfc02": {"require_abs_logfc": True},
}
PEER_BASELINE_VARIANTS = (
    "source_centroid",
    "target_centroid",
    "source_peer",
    "target_peer",
)
RETRIEVAL_VARIANTS = (
    "compound_across_doses",
    "dose_aware_compound",
    "strict_matched_condition",
)
RETRIEVAL_SIMILARITIES = ("negative_l2", "cosine", "spearman")


def finite_values_mask(*arrays: np.ndarray) -> np.ndarray:
    if not arrays:
        raise ValueError("finite_values_mask requires at least one array")
    mask = np.ones(len(arrays[0]), dtype=bool)
    for array in arrays:
        mask &= np.isfinite(np.asarray(array, dtype=np.float64))
    return mask


def strict_symmetric_mean(left: float, right: float) -> float:
    if pd.isna(left) or pd.isna(right):
        return float("nan")
    return float(np.mean([left, right]))


def deg_mask(
    logfc: np.ndarray,
    adj_p: np.ndarray,
    definition_key: str,
) -> np.ndarray:
    mask = (
        finite_values_mask(logfc, adj_p)
        & (np.asarray(adj_p, dtype=np.float64) < DEG_P_THRESHOLD)
    )
    if DEG_DEFINITIONS[definition_key]["require_abs_logfc"]:
        mask &= (
            np.abs(np.asarray(logfc, dtype=np.float64))
            > DEG_ABS_LOGFC_THRESHOLD
        )
    return mask


def ranked_genes_from_mask(
    gene_keys: np.ndarray,
    values: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return np.empty(0, dtype=object)
    order = np.argsort(
        -np.abs(np.asarray(values, dtype=np.float64)[indices]),
        kind="stable",
    )
    return np.asarray(gene_keys, dtype=object)[indices[order]]


def ranked_genes_by_abs_values(
    gene_keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    return ranked_genes_from_mask(
        gene_keys,
        values,
        finite_values_mask(values),
    )


def top_overlap_fraction(
    ranked_left: np.ndarray,
    ranked_right: np.ndarray,
    n: int,
) -> float:
    n = int(n)
    if n <= 0 or len(ranked_left) < n or len(ranked_right) < n:
        return float("nan")
    left = {str(gene) for gene in ranked_left[:n].tolist()}
    right = {str(gene) for gene in ranked_right[:n].tolist()}
    return float(len(left & right) / n)


def top_overlap_at_k_strict(
    ranked_left: np.ndarray,
    ranked_right: np.ndarray,
    k: int,
) -> float:
    return top_overlap_fraction(ranked_left, ranked_right, int(k))


def direction_agreement_with_masks(
    logfc_left: np.ndarray,
    logfc_right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    mask = (
        np.asarray(left_mask, dtype=bool)
        & np.asarray(right_mask, dtype=bool)
        & finite_values_mask(logfc_left, logfc_right)
    )
    if not mask.any():
        return float("nan")
    return float(
        np.mean(
            np.sign(np.asarray(logfc_left, dtype=np.float64)[mask])
            == np.sign(np.asarray(logfc_right, dtype=np.float64)[mask])
        )
    )


def deg_restricted_spearman(
    logfc_left: np.ndarray,
    logfc_right: np.ndarray,
    mask: np.ndarray,
) -> float:
    eval_mask = np.asarray(mask, dtype=bool) & finite_values_mask(
        logfc_left,
        logfc_right,
    )
    if int(eval_mask.sum()) < 2:
        return float("nan")
    return cross_source_core.signed_spearman(
        np.asarray(logfc_left)[eval_mask],
        np.asarray(logfc_right)[eval_mask],
    )


def observed_deg_metrics(
    *,
    gene_keys: np.ndarray,
    logfc_left: np.ndarray,
    logfc_right: np.ndarray,
    adj_p_left: np.ndarray,
    adj_p_right: np.ndarray,
    definition_key: str,
    prefix: str = "observed",
) -> dict[str, float]:
    left_mask = deg_mask(logfc_left, adj_p_left, definition_key)
    right_mask = deg_mask(logfc_right, adj_p_right, definition_key)
    ranked_left = ranked_genes_from_mask(gene_keys, logfc_left, left_mask)
    ranked_right = ranked_genes_from_mask(gene_keys, logfc_right, right_mask)
    left_spearman = deg_restricted_spearman(
        logfc_left,
        logfc_right,
        left_mask,
    )
    right_spearman = deg_restricted_spearman(
        logfc_left,
        logfc_right,
        right_mask,
    )
    n_left = int(left_mask.sum())
    n_right = int(right_mask.sum())
    values: dict[str, float] = {
        f"n_deg_left_{definition_key}": n_left,
        f"n_deg_right_{definition_key}": n_right,
        f"n_deg_overlap_{definition_key}": int(
            (
                left_mask
                & right_mask
                & finite_values_mask(logfc_left, logfc_right)
            ).sum()
        ),
        f"{prefix}_deg_lfc_spearman_left_ref_{definition_key}": left_spearman,
        f"{prefix}_deg_lfc_spearman_right_ref_{definition_key}": right_spearman,
        f"{prefix}_deg_lfc_spearman_sym_{definition_key}": strict_symmetric_mean(
            left_spearman,
            right_spearman,
        ),
        f"{prefix}_direction_agreement_{definition_key}": (
            direction_agreement_with_masks(
                logfc_left,
                logfc_right,
                left_mask,
                right_mask,
            )
        ),
    }
    left_overlap = top_overlap_fraction(
        ranked_left,
        ranked_right,
        n_left,
    )
    right_overlap = top_overlap_fraction(
        ranked_left,
        ranked_right,
        n_right,
    )
    values[f"{prefix}_de_overlap_refn_left_ref_{definition_key}"] = left_overlap
    values[f"{prefix}_de_overlap_refn_right_ref_{definition_key}"] = right_overlap
    values[f"{prefix}_de_overlap_refn_sym_{definition_key}"] = strict_symmetric_mean(
        left_overlap,
        right_overlap,
    )
    for k in DE_OVERLAP_K_VALUES:
        values[f"{prefix}_de_overlap_k{k}_{definition_key}"] = (
            top_overlap_at_k_strict(ranked_left, ranked_right, k)
        )
    return values


def sample_baseline_deg_metrics(
    *,
    prefix: str,
    gene_keys: np.ndarray,
    sample_logfc: np.ndarray,
    baseline_logfc: Optional[np.ndarray],
    sample_adj_p: np.ndarray,
    definition_key: str,
) -> dict[str, float]:
    names = [
        "deg_lfc_spearman",
        "de_overlap_refn",
        "direction_agreement",
        *[f"de_overlap_k{k}" for k in DE_OVERLAP_K_VALUES],
    ]
    if baseline_logfc is None:
        return {
            f"{prefix}_baseline_{name}_{definition_key}": float("nan")
            for name in names
        }
    mask = deg_mask(sample_logfc, sample_adj_p, definition_key)
    ranked_sample = ranked_genes_from_mask(gene_keys, sample_logfc, mask)
    ranked_baseline = ranked_genes_by_abs_values(gene_keys, baseline_logfc)
    values = {
        f"{prefix}_baseline_deg_lfc_spearman_{definition_key}": (
            deg_restricted_spearman(sample_logfc, baseline_logfc, mask)
        ),
        f"{prefix}_baseline_de_overlap_refn_{definition_key}": (
            top_overlap_fraction(
                ranked_sample,
                ranked_baseline,
                int(mask.sum()),
            )
        ),
        f"{prefix}_baseline_direction_agreement_{definition_key}": (
            direction_agreement_with_masks(
                sample_logfc,
                baseline_logfc,
                mask,
                finite_values_mask(baseline_logfc),
            )
        ),
    }
    for k in DE_OVERLAP_K_VALUES:
        values[f"{prefix}_baseline_de_overlap_k{k}_{definition_key}"] = (
            top_overlap_at_k_strict(ranked_sample, ranked_baseline, k)
        )
    return values


def row_lookup(
    row: Mapping[str, Any],
    *,
    side: str,
) -> dict[str, str]:
    return {
        "pubchem_cid": str(row["pubchem_cid"]),
        "dose_key": str(row[f"{side}_dose_key"]),
        "time_key": str(row["time_key"]),
    }


@dataclass(frozen=True)
class SelectedPeers:
    stratum: SignatureStratum
    row_indices: np.ndarray
    total_count: int

    @property
    def scored_count(self) -> int:
        return int(self.row_indices.size)


def select_source_peers(
    source: cross_source_core.LineSource,
    obs_id: str,
    lookup: Mapping[str, str],
    *,
    max_peers: Optional[int],
    sampling_seed: int,
) -> SelectedPeers:
    selected = cross_source_core.select_line_source_peers(
        source,
        obs_id,
        pubchem_cid=str(lookup["pubchem_cid"]),
        dose_key=str(lookup["dose_key"]),
        time_key=str(lookup["time_key"]),
        max_peers=max_peers,
        sampling_seed=sampling_seed,
    )
    return SelectedPeers(
        stratum=selected.stratum,
        row_indices=selected.row_indices,
        total_count=selected.total_count,
    )


def score_signature_baselines(
    *,
    observed: float,
    left_sample: np.ndarray,
    right_sample: np.ndarray,
    left_centroid: Optional[np.ndarray],
    right_centroid: Optional[np.ndarray],
    left_peer_matrix: np.ndarray,
    right_peer_matrix: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {}
    sides = {
        "left": {
            "sample": left_sample,
            "source_centroid": left_centroid,
            "target_centroid": right_centroid,
            "source_peer": left_peer_matrix,
            "target_peer": right_peer_matrix,
        },
        "right": {
            "sample": right_sample,
            "source_centroid": right_centroid,
            "target_centroid": left_centroid,
            "source_peer": right_peer_matrix,
            "target_peer": left_peer_matrix,
        },
    }
    for variant in PEER_BASELINE_VARIANTS:
        means: list[float] = []
        sds: list[float] = []
        fractions: list[float] = []
        percentiles: list[float] = []
        for side, spec in sides.items():
            stem = f"{prefix}_{variant}_spearman_logfc_{side}"
            if variant.endswith("_centroid"):
                centroid = spec[variant]
                value = (
                    float("nan")
                    if centroid is None
                    else cross_source_core.signed_spearman(
                        spec["sample"],
                        centroid,
                    )
                )
                record[stem] = value
                means.append(value)
                continue
            peer_matrix = np.asarray(spec[variant], dtype=np.float64)
            if peer_matrix.shape[0] == 0:
                summary = empty_peer_score_summary(stem)
            else:
                summary = summarize_peer_scores(
                    observed,
                    spearman_against_peers(spec["sample"], peer_matrix),
                    stem,
                )
            record.update(summary)
            means.append(float(summary[f"{stem}_mean_score"]))
            sds.append(float(summary[f"{stem}_sd_score"]))
            fractions.append(
                float(summary[f"{stem}_fraction_below_observed"])
            )
            percentiles.append(
                float(summary[f"{stem}_corrected_percentile"])
            )
        pair_stem = f"{prefix}_{variant}_spearman_logfc_pair"
        pair_value = cross_source_core.mean_available(means)
        record[pair_stem] = pair_value
        record[f"{prefix}_delta_vs_{variant}_spearman_logfc"] = (
            cross_source_core.difference_if_both_defined(observed, pair_value)
        )
        if not variant.endswith("_centroid"):
            record[f"{pair_stem}_sd_score"] = cross_source_core.mean_available(sds)
            record[f"{pair_stem}_fraction_below_observed"] = (
                cross_source_core.mean_available(fractions)
            )
            record[f"{pair_stem}_corrected_percentile"] = (
                cross_source_core.mean_available(percentiles)
            )
    return record


def score_deg_peer_distribution(
    *,
    observed: float,
    sample: np.ndarray,
    comparison_matrix: np.ndarray,
    sample_adj_p: np.ndarray,
    metric: str,
    stem: str,
) -> dict[str, Any]:
    matrix = np.atleast_2d(np.asarray(comparison_matrix, dtype=np.float64))
    if matrix.shape[0] == 0:
        return empty_peer_score_summary(stem)
    mask = deg_mask(sample, sample_adj_p, "p05")
    if metric == "deg_lfc_spearman":
        scores = spearman_against_peers(sample, matrix, mask)
    elif metric == "direction_agreement":
        scores = np.asarray(
            [
                direction_agreement_with_masks(
                    sample,
                    peer,
                    mask,
                    finite_values_mask(peer),
                )
                for peer in matrix
            ],
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unsupported DEG peer metric: {metric}")
    return summarize_peer_scores(observed, scores, stem)


def signed_significance(logfc: np.ndarray, adj_p: np.ndarray) -> np.ndarray:
    return -np.log10(
        np.clip(np.asarray(adj_p, dtype=np.float64), 1e-300, None)
    ) * np.sign(np.asarray(logfc, dtype=np.float64))


def negative_l2_similarity_matrix(
    query_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
) -> np.ndarray:
    return -cdist(
        np.asarray(query_matrix, dtype=np.float64),
        np.asarray(candidate_matrix, dtype=np.float64),
        metric="euclidean",
    )


def _normalized_rows(
    matrix: np.ndarray,
    *,
    rank_rows: bool,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    transformed = np.zeros_like(matrix, dtype=np.float64)
    if finite.any():
        values = matrix[finite]
        if rank_rows:
            values = np.atleast_2d(
                rankdata(values, method="average", axis=1)
            ).astype(np.float64)
            values = values - values.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(values, axis=1)
        valid_local = norms > 0.0
        finite_indices = np.flatnonzero(finite)
        valid = np.zeros(matrix.shape[0], dtype=bool)
        valid[finite_indices[valid_local]] = True
        transformed[finite_indices[valid_local]] = (
            values[valid_local] / norms[valid_local, None]
        )
        return transformed, valid
    return transformed, np.zeros(matrix.shape[0], dtype=bool)


def similarity_matrix(
    query_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    query_matrix = np.asarray(query_matrix, dtype=np.float64)
    candidate_matrix = np.asarray(candidate_matrix, dtype=np.float64)
    if metric == "negative_l2":
        valid_query = np.isfinite(query_matrix).all(axis=1)
        valid_candidate = np.isfinite(candidate_matrix).all(axis=1)
        scores = np.full(
            (len(query_matrix), len(candidate_matrix)),
            np.nan,
            dtype=np.float64,
        )
        if valid_query.any() and valid_candidate.any():
            scores[np.ix_(valid_query, valid_candidate)] = (
                negative_l2_similarity_matrix(
                    query_matrix[valid_query],
                    candidate_matrix[valid_candidate],
                )
            )
        return scores, valid_query, valid_candidate
    if metric not in {"cosine", "spearman"}:
        raise ValueError(f"Unsupported similarity metric: {metric}")
    queries, valid_query = _normalized_rows(
        query_matrix,
        rank_rows=metric == "spearman",
    )
    candidates, valid_candidate = _normalized_rows(
        candidate_matrix,
        rank_rows=metric == "spearman",
    )
    # NumPy 2.0 on macOS may surface stale Accelerate floating-point flags for a
    # finite normalized matrix product. The product itself remains finite; suppress
    # only those spurious warnings while preserving the notebook's dot-product formula.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = queries @ candidates.T
    return (
        np.clip(scores, -1.0, 1.0),
        valid_query,
        valid_candidate,
    )


def score_vector_against_matrix(
    query: np.ndarray,
    matrix: np.ndarray,
    metric: str,
) -> np.ndarray:
    scores, valid_query, valid_candidates = similarity_matrix(
        np.asarray(query)[None, :],
        matrix,
        metric,
    )
    result = scores[0]
    if not valid_query[0]:
        return np.full(len(matrix), np.nan)
    return result


def score_vector_pair(
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
) -> float:
    scores, valid_left, valid_right = similarity_matrix(
        np.asarray(left)[None, :],
        np.asarray(right)[None, :],
        metric,
    )
    if not valid_left[0] or not valid_right[0]:
        return float("nan")
    return float(scores[0, 0])


def normalized_best_positive_rank(
    scores: np.ndarray,
    positive_mask: np.ndarray,
) -> tuple[float, float]:
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    if scores.size < 2 or not positive_mask.any():
        return float("nan"), float("nan")
    ranks = rankdata(-scores, method="min")
    best = float(np.min(ranks[positive_mask]))
    return best, float(1.0 - ((best - 1.0) / (len(scores) - 1.0)))


def recall_at_1(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    best, _ = normalized_best_positive_rank(scores, positive_mask)
    return float(best == 1.0) if np.isfinite(best) else float("nan")


def auroc_from_scores(scores: np.ndarray, positive_mask: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    positives = scores[positive_mask]
    negatives = scores[~positive_mask]
    if positives.size == 0 or negatives.size == 0:
        return float("nan")
    wins = 0.0
    for score in positives:
        wins += float(np.sum(score > negatives))
        wins += 0.5 * float(
            np.sum(np.isclose(score, negatives, rtol=0.0, atol=1e-12))
        )
    return float(wins / (positives.size * negatives.size))


def summarize_retrieval_scores(
    scores: np.ndarray,
    positive_mask: np.ndarray,
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    positive_mask = np.asarray(positive_mask, dtype=bool)
    n_positive = int(positive_mask.sum())
    n_negative = int(len(positive_mask) - n_positive)
    empty = {
        "n_positives": n_positive,
        "n_negatives": n_negative,
        "best_rank": float("nan"),
        "normalized_rank": float("nan"),
        "recall_at_1": float("nan"),
        "auroc": float("nan"),
    }
    if (
        n_positive == 0
        or n_negative == 0
        or scores.size == 0
        or not np.isfinite(scores).all()
    ):
        return empty
    best, normalized = normalized_best_positive_rank(scores, positive_mask)
    return {
        "n_positives": n_positive,
        "n_negatives": n_negative,
        "best_rank": best,
        "normalized_rank": normalized,
        "recall_at_1": recall_at_1(scores, positive_mask),
        "auroc": auroc_from_scores(scores, positive_mask),
    }


def raw_dose_fold_differences(
    query_dose: float,
    candidate_doses: np.ndarray,
) -> np.ndarray:
    candidate_doses = np.asarray(candidate_doses, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.maximum(
            candidate_doses / float(query_dose),
            float(query_dose) / candidate_doses,
        )


def primary_positive_mask(
    compound: str,
    candidate_compounds: np.ndarray,
) -> np.ndarray:
    return np.asarray(candidate_compounds).astype(str) == str(compound)


def dose_aware_positive_mask(
    compound: str,
    query_dose: float,
    candidate_compounds: np.ndarray,
    candidate_doses: np.ndarray,
    *,
    max_fold: float,
) -> np.ndarray:
    return primary_positive_mask(
        compound,
        candidate_compounds,
    ) & (
        raw_dose_fold_differences(query_dose, candidate_doses)
        <= float(max_fold) + 1e-12
    )


def strict_positive_mask(
    query_obs_id: str,
    candidate_obs_ids: np.ndarray,
    strict_map: Mapping[str, set[str]],
) -> np.ndarray:
    positives = strict_map.get(str(query_obs_id), set())
    return np.asarray(
        [str(obs_id) in positives for obs_id in candidate_obs_ids],
        dtype=bool,
    )


def strict_positive_maps(
    matched_pairs: pd.DataFrame,
) -> dict[str, dict[str, set[str]]]:
    forward: dict[str, set[str]] = {}
    reverse: dict[str, set[str]] = {}
    for row in matched_pairs.itertuples(index=False):
        forward.setdefault(str(row.left_obs_id), set()).add(
            str(row.right_obs_id)
        )
        reverse.setdefault(str(row.right_obs_id), set()).add(
            str(row.left_obs_id)
        )
    return {"A_to_B": forward, "B_to_A": reverse}


def exact_cross_assay_null(
    *,
    n_candidates: int,
    n_positives: int,
) -> dict[str, float]:
    n = int(n_candidates)
    k = int(n_positives)
    if n < 2 or k < 1 or k >= n:
        return {
            "best_rank": float("nan"),
            "normalized_rank": float("nan"),
            "recall_at_1": float("nan"),
            "auroc": 0.5 if k > 0 and n > k else float("nan"),
        }
    expected_best_rank = (n + 1.0) / (k + 1.0)
    normalized = 1.0 - ((expected_best_rank - 1.0) / (n - 1.0))
    return {
        "best_rank": float(expected_best_rank),
        "normalized_rank": float(normalized),
        "recall_at_1": float(k / n),
        "auroc": 0.5,
    }


def null_best_rank_survival(
    n_candidates: int,
    n_positives: int,
    rank: int,
) -> float:
    """P(best rank >= rank) under uniform positive-label placement."""
    from math import comb

    n = int(n_candidates)
    k = int(n_positives)
    rank = int(rank)
    if not (1 <= k <= n) or rank < 1:
        return float("nan")
    if n - rank + 1 < k:
        return 0.0
    return float(comb(n - rank + 1, k) / comb(n, k))


def exact_null_mid_p(
    *,
    n_candidates: int,
    n_positives: int,
    best_rank: int,
) -> float:
    upper_at_rank = null_best_rank_survival(
        n_candidates,
        n_positives,
        best_rank,
    )
    upper_after = null_best_rank_survival(
        n_candidates,
        n_positives,
        best_rank + 1,
    )
    if not np.isfinite(upper_at_rank) or not np.isfinite(upper_after):
        return float("nan")
    probability_equal = upper_at_rank - upper_after
    return float(1.0 - upper_after - 0.5 * probability_equal)


def exact_null_rank_calibration(
    *,
    n_candidates: int,
    n_positives: int,
    best_rank: float,
) -> dict[str, float]:
    if not np.isfinite(best_rank):
        return {"null_p_value": float("nan"), "null_pit": float("nan")}
    rank = int(best_rank)
    survival_at_rank = null_best_rank_survival(
        n_candidates,
        n_positives,
        rank,
    )
    survival_after_rank = null_best_rank_survival(
        n_candidates,
        n_positives,
        rank + 1,
    )
    if not (
        np.isfinite(survival_at_rank)
        and np.isfinite(survival_after_rank)
    ):
        return {"null_p_value": float("nan"), "null_pit": float("nan")}
    probability_equal = max(0.0, survival_at_rank - survival_after_rank)
    p_at_most = 1.0 - survival_after_rank
    p_below = 1.0 - survival_at_rank
    return {
        "null_p_value": float(np.clip(p_at_most, 0.0, 1.0)),
        "null_pit": float(
            np.clip(p_below + 0.5 * probability_equal, 0.0, 1.0)
        ),
    }


def expected_single_signature_metrics(
    target_scores: np.ndarray,
    peer_scores: np.ndarray,
    *,
    chunk_size: int = 512,
) -> dict[str, float]:
    del chunk_size
    target_scores = np.asarray(target_scores, dtype=np.float64)
    peer_scores = np.asarray(peer_scores, dtype=np.float64)
    target_scores = target_scores[np.isfinite(target_scores)]
    peer_scores = peer_scores[np.isfinite(peer_scores)]
    if target_scores.size < 1 or peer_scores.size < 1:
        return {
            "n_peers": int(peer_scores.size),
            "best_rank": float("nan"),
            "normalized_rank": float("nan"),
            "recall_at_1": float("nan"),
            "auroc": float("nan"),
        }
    sorted_targets = np.sort(target_scores)
    n_targets = int(sorted_targets.size)
    insertion_left = np.searchsorted(
        sorted_targets,
        peer_scores,
        side="left",
    )
    insertion_right = np.searchsorted(
        sorted_targets,
        peer_scores,
        side="right",
    )
    n_better_targets = n_targets - insertion_right
    ranks = 1.0 + n_better_targets.astype(np.float64)

    close_left = np.searchsorted(
        sorted_targets,
        peer_scores - 1e-12,
        side="left",
    )
    close_right = np.searchsorted(
        sorted_targets,
        peer_scores + 1e-12,
        side="right",
    )
    close_counts = close_right - close_left
    wins = insertion_left.astype(np.float64)
    wins += 0.5 * close_counts.astype(np.float64)
    return {
        "n_peers": int(peer_scores.size),
        "best_rank": float(np.mean(ranks)),
        "normalized_rank": float(
            np.mean(1.0 - n_better_targets.astype(np.float64) / n_targets)
        ),
        "recall_at_1": float(np.mean(ranks == 1.0)),
        "auroc": float(np.mean(wins / float(n_targets))),
    }


def retrieval_peer_summary(
    observed_score: float,
    scores: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    values = summarize_peer_scores(observed_score, scores, prefix)
    # Preserve the retrieval notebook's public terminology alongside the common helper.
    renamed: dict[str, Any] = {}
    for key, value in values.items():
        renamed[
            key.replace("_mean_score", "_mean_similarity").replace(
                "_sd_score",
                "_sd_similarity",
            )
        ] = value
    return renamed


def pool_frame_for_context(
    dataset_frame: pd.DataFrame,
    cell_type: str,
    time_key: str,
) -> pd.DataFrame:
    return dataset_frame.loc[
        (dataset_frame["cell_type"].astype(str) == str(cell_type))
        & (dataset_frame["time_key"].astype(str) == str(time_key))
    ].copy().reset_index(drop=True)


@dataclass
class PoolMatrices:
    frame: pd.DataFrame
    logfc: np.ndarray
    t: np.ndarray
    adj_p: np.ndarray
    centroid_logfc: np.ndarray
    centroid_t: np.ndarray
    centroid_adj_p: np.ndarray
    centroid_peer_counts: np.ndarray
    diagnostics: list[dict[str, Any]]


@dataclass
class LogFCPoolMatrices:
    frame: pd.DataFrame
    logfc: np.ndarray
    diagnostics: list[dict[str, Any]]


def build_logfc_pool_matrices(
    pool_frame: pd.DataFrame,
    source: cross_source_core.LineSource,
    gene_positions: np.ndarray,
) -> LogFCPoolMatrices:
    resolved: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    logfc_rows: list[np.ndarray] = []
    for _, row in pool_frame.iterrows():
        lookup = {
            "pubchem_cid": str(row["pubchem_cid"]),
            "dose_key": str(row["dose_key"]),
            "time_key": str(row["time_key"]),
        }
        obs_id = str(row["obs_id"])
        try:
            logfc_rows.append(
                source.get_vector(obs_id, "logFC", **lookup)[gene_positions]
            )
            resolved.append(row.to_dict())
        except KeyError as exc:
            if "Could not resolve obs_id=" not in str(exc):
                raise
            diagnostics.append(
                {
                    **row.to_dict(),
                    "stage": "pool_resolution",
                    "reason": "unresolved_source_row",
                    "error": str(exc),
                }
            )
    matrix = (
        np.asarray(logfc_rows, dtype=np.float64)
        if logfc_rows
        else np.empty((0, len(gene_positions)), dtype=np.float64)
    )
    return LogFCPoolMatrices(
        frame=pd.DataFrame(resolved, columns=pool_frame.columns),
        logfc=matrix,
        diagnostics=diagnostics,
    )


def build_pool_matrices(
    pool_frame: pd.DataFrame,
    source: cross_source_core.LineSource,
    gene_positions: np.ndarray,
    *,
    adj_layer_name: str,
) -> PoolMatrices:
    resolved: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    logfc_rows: list[np.ndarray] = []
    t_rows: list[np.ndarray] = []
    adj_rows: list[np.ndarray] = []
    centroid_logfc_rows: list[np.ndarray] = []
    centroid_t_rows: list[np.ndarray] = []
    centroid_adj_rows: list[np.ndarray] = []
    peer_counts: list[int] = []
    n_genes = int(len(gene_positions))
    nan_row = np.full(n_genes, np.nan, dtype=np.float64)
    for _, row in pool_frame.iterrows():
        lookup = {
            "pubchem_cid": str(row["pubchem_cid"]),
            "dose_key": str(row["dose_key"]),
            "time_key": str(row["time_key"]),
        }
        obs_id = str(row["obs_id"])
        try:
            logfc_rows.append(
                source.get_vector(obs_id, "logFC", **lookup)[gene_positions]
            )
            t_rows.append(source.get_vector(obs_id, "t", **lookup)[gene_positions])
            adj_rows.append(
                source.get_vector(obs_id, adj_layer_name, **lookup)[gene_positions]
            )
            centroid_logfc = source.get_baseline_vector(
                obs_id,
                "logFC",
                **lookup,
            )
            centroid_t = source.get_baseline_vector(obs_id, "t", **lookup)
            centroid_adj = source.get_baseline_vector(
                obs_id,
                adj_layer_name,
                **lookup,
            )
            peer_counts.append(source.baseline_peer_count(obs_id, **lookup))
            centroid_logfc_rows.append(
                nan_row.copy()
                if centroid_logfc is None
                else np.asarray(centroid_logfc)[gene_positions]
            )
            centroid_t_rows.append(
                nan_row.copy()
                if centroid_t is None
                else np.asarray(centroid_t)[gene_positions]
            )
            centroid_adj_rows.append(
                nan_row.copy()
                if centroid_adj is None
                else np.asarray(centroid_adj)[gene_positions]
            )
            resolved.append(row.to_dict())
        except KeyError as exc:
            if "Could not resolve obs_id=" not in str(exc):
                raise
            diagnostics.append(
                {
                    **row.to_dict(),
                    "stage": "pool_resolution",
                    "reason": "unresolved_source_row",
                    "error": str(exc),
                }
            )

    def matrix(rows: Sequence[np.ndarray]) -> np.ndarray:
        if not rows:
            return np.empty((0, n_genes), dtype=np.float64)
        return np.asarray(rows, dtype=np.float64)

    return PoolMatrices(
        frame=pd.DataFrame(resolved, columns=pool_frame.columns),
        logfc=matrix(logfc_rows),
        t=matrix(t_rows),
        adj_p=matrix(adj_rows),
        centroid_logfc=matrix(centroid_logfc_rows),
        centroid_t=matrix(centroid_t_rows),
        centroid_adj_p=matrix(centroid_adj_rows),
        centroid_peer_counts=np.asarray(peer_counts, dtype=np.int64),
        diagnostics=diagnostics,
    )


def peer_matrix_for_query(
    source: cross_source_core.LineSource,
    *,
    dose_key: str,
    time_key: str,
    excluded_compound: str,
    gene_positions: np.ndarray,
    max_peers: Optional[int],
    sampling_seed: int,
) -> tuple[np.ndarray, Optional[np.ndarray], int, int]:
    compounds, full_matrix = cross_source_core.line_source_stratum_arrays(
        source,
        dose_key=dose_key,
        time_key=time_key,
    )
    compounds = np.asarray(compounds).astype(str)
    all_peer_rows = np.flatnonzero(compounds != str(excluded_compound))
    n_total = int(all_peer_rows.size)
    if n_total == 0:
        return (
            np.empty((0, len(gene_positions)), dtype=np.float64),
            None,
            0,
            0,
        )
    offsets = select_peer_indices(
        n_total,
        max_peers,
        "|".join(
            [
                source.dataset_name,
                source.cell_type,
                str(dose_key),
                str(time_key),
                str(excluded_compound),
            ]
        ),
        sampling_seed=sampling_seed,
    )
    matrix = np.asarray(full_matrix, dtype=np.float64)[:, gene_positions]
    selected = matrix[all_peer_rows[offsets]]
    centroid = matrix[all_peer_rows].mean(axis=0, dtype=np.float64)
    return selected, centroid, n_total, int(len(offsets))


__all__ = [
    "ADJ_PVALUE_LAYER_PREFERENCES",
    "DEG_DEFINITIONS",
    "DEG_P_THRESHOLD",
    "DE_OVERLAP_K_VALUES",
    "PEER_BASELINE_VARIANTS",
    "LogFCPoolMatrices",
    "PoolMatrices",
    "RETRIEVAL_SIMILARITIES",
    "RETRIEVAL_VARIANTS",
    "SelectedPeers",
    "auroc_from_scores",
    "build_logfc_pool_matrices",
    "build_pool_matrices",
    "deg_mask",
    "deg_restricted_spearman",
    "direction_agreement_with_masks",
    "dose_aware_positive_mask",
    "exact_cross_assay_null",
    "exact_null_mid_p",
    "exact_null_rank_calibration",
    "expected_single_signature_metrics",
    "finite_values_mask",
    "negative_l2_similarity_matrix",
    "normalized_best_positive_rank",
    "null_best_rank_survival",
    "observed_deg_metrics",
    "peer_matrix_for_query",
    "pool_frame_for_context",
    "primary_positive_mask",
    "raw_dose_fold_differences",
    "recall_at_1",
    "retrieval_peer_summary",
    "row_lookup",
    "sample_baseline_deg_metrics",
    "score_deg_peer_distribution",
    "score_signature_baselines",
    "score_vector_against_matrix",
    "score_vector_pair",
    "select_source_peers",
    "signed_significance",
    "similarity_matrix",
    "strict_positive_maps",
    "strict_positive_mask",
    "strict_symmetric_mean",
    "summarize_retrieval_scores",
    "top_overlap_at_k_strict",
    "top_overlap_fraction",
]
