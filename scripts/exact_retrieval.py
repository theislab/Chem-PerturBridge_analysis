"""Scoring for ``--exact`` Tables 12/13 retrieval runs.

``--exact`` reproduces the paper notebooks' raw logFC, sign(logFC) and moderated-t scores
bitwise. Identical signatures (wells with the same compound and dose) have exactly equal
true scores, so the last bit of each score decides how their peer ranks break ties:

- Peers and centroids use dot-then-divide cosine, so sign vectors with equal true cosine
  get bitwise-equal scores.
- The cosine score matrix, which target peers are read from, takes row norms on
  column-major copies.

Bitwise agreement also needs one process with eight BLAS threads (set by the runner), the
Python 3.12.13 environment in ``retrieval_requirements.txt`` and the paper's CPU type
(Intel Xeon Gold 6134), since BLAS kernels differ between CPUs.
"""
from __future__ import annotations

import numpy as np

from scripts.cross_source_scoring import (
    score_vector_against_matrix,
    score_vector_pair,
    similarity_matrix as default_similarity_matrix,
)


def scorers(exact: bool):
    """Matrix, peer and pair scorers: this module's with ``--exact``, the defaults otherwise."""
    if exact:
        return similarity_matrix, peer_scores, peer_pair
    return default_similarity_matrix, score_vector_against_matrix, score_vector_pair


def similarity_matrix(
    query_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
    metric: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score matrix with cosine row norms on column-major copies; invalid rows score NaN."""
    if metric != "cosine":
        return default_similarity_matrix(query_matrix, candidate_matrix, metric)
    query_matrix = np.asfortranarray(query_matrix, dtype=np.float64)
    candidate_matrix = np.asfortranarray(candidate_matrix, dtype=np.float64)
    query_norms = np.linalg.norm(query_matrix, axis=1)
    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    valid_query = np.isfinite(query_norms) & (query_norms > 0.0)
    valid_candidate = np.isfinite(candidate_norms) & (candidate_norms > 0.0)
    scores = np.full((len(query_matrix), len(candidate_matrix)), np.nan, dtype=np.float64)
    scores[np.ix_(valid_query, valid_candidate)] = np.clip(
        (query_matrix[valid_query] / query_norms[valid_query, None])
        @ (candidate_matrix[valid_candidate] / candidate_norms[valid_candidate, None]).T,
        -1.0,
        1.0,
    )
    return scores, valid_query, valid_candidate


def peer_scores(
    query: np.ndarray,
    matrix: np.ndarray,
    metric: str,
) -> np.ndarray:
    """Single-query scores with dot-then-divide cosine; invalid candidates score NaN."""
    if metric != "cosine":
        return score_vector_against_matrix(query, matrix, metric)
    query = np.asarray(query, dtype=np.float64)
    matrix = np.atleast_2d(np.asarray(matrix, dtype=np.float64))
    scores = np.full(len(matrix), np.nan, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    if not np.isfinite(query_norm) or query_norm <= 0.0:
        return scores
    norms = np.linalg.norm(matrix, axis=1)
    valid = np.isfinite(norms) & (norms > 0.0)
    scores[valid] = np.clip((matrix[valid] @ query) / (norms[valid] * query_norm), -1.0, 1.0)
    return scores


def peer_pair(
    left: np.ndarray,
    right: np.ndarray,
    metric: str,
) -> float:
    return float(peer_scores(left, np.asarray(right, dtype=np.float64)[None, :], metric)[0])


__all__ = [
    "peer_pair",
    "peer_scores",
    "scorers",
    "similarity_matrix",
]
