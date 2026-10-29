from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def cosine_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    nx = np.linalg.norm(x64)
    if nx <= 1e-12:
        return np.full(y64.shape[0], np.nan, dtype=np.float64)
    ny = np.linalg.norm(y64, axis=1)
    denom = nx * ny
    out = np.full(y64.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = (y64[valid] @ x64) / denom[valid]
    return out


def mrrmse_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    diff = y64 - x64[None, :]
    return np.sqrt(np.mean(diff * diff, axis=1, dtype=np.float64))


def pearson_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    x_centered = x64 - np.mean(x64)
    nx = np.linalg.norm(x_centered)
    if nx <= 1e-12:
        return np.full(y64.shape[0], np.nan, dtype=np.float64)

    y_centered = y64 - np.mean(y64, axis=1, keepdims=True)
    ny = np.linalg.norm(y_centered, axis=1)
    denom = nx * ny
    out = np.full(y64.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = (y_centered[valid] @ x_centered) / denom[valid]
    return out


def rank_center_norm_rows(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rank-transform each row for Spearman correlation.
    Returns rank-centered matrix, row norms, and finite-row mask.
    """
    n_rows, n_cols = x.shape
    rank_center = (n_cols + 1.0) / 2.0

    row_valid = np.isfinite(x).all(axis=1)
    ranked = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    norms = np.full(n_rows, np.nan, dtype=np.float32)

    if row_valid.any():
        x_valid = x[row_valid]
        r_valid = rankdata(x_valid, axis=1, method="average").astype(np.float32, copy=False)
        r_valid -= np.float32(rank_center)
        ranked[row_valid] = r_valid
        norms[row_valid] = np.linalg.norm(r_valid, axis=1).astype(np.float32, copy=False)

    return ranked, norms, row_valid


def spearman_scores_precomputed(
    x_ranked: np.ndarray,
    x_norm: float,
    y_ranked: np.ndarray,
    y_norms: np.ndarray,
) -> np.ndarray:
    if not np.isfinite(x_norm) or x_norm <= 1e-12:
        return np.full(y_ranked.shape[0], np.nan, dtype=np.float64)

    dots = np.asarray(y_ranked) @ np.asarray(x_ranked)
    denom = np.asarray(y_norms, dtype=np.float64) * float(x_norm)
    out = np.full(y_ranked.shape[0], np.nan, dtype=np.float64)
    valid = denom > 1e-12
    out[valid] = dots[valid] / denom[valid]
    return out


def compute_signed_adjusted(
    logfc: np.ndarray,
    adj_pvalues: np.ndarray,
    p_floor: float,
) -> np.ndarray:
    adj_p = np.asarray(adj_pvalues, dtype=np.float64)
    l = np.asarray(logfc, dtype=np.float64)

    signed = -np.log10(np.clip(adj_p, p_floor, None)) * np.sign(l)
    return signed.astype(np.float32, copy=False)


def compute_signed_and_z(
    logfc: np.ndarray,
    pvalues: np.ndarray,
    p_floor: float,
    p_clip_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvalues, dtype=np.float64)
    l = np.asarray(logfc, dtype=np.float64)
    p = np.clip(p, p_floor, p_clip_high)

    signed = -np.log10(p) * np.sign(l)
    z_score = np.sign(l) * norm.ppf(1.0 - p / 2.0)
    return signed.astype(np.float32, copy=False), z_score.astype(np.float32, copy=False)
