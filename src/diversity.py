"""Diversity measures over categorical distributions (used from Notebook 02 on).

Entropy uses the natural log. Two normalisations are provided because they
answer different questions:

* ``normalized_entropy(..., k=None)`` divides by log(K_observed): evenness among
  the categories a user actually touched (undefined for K_observed = 1).
* ``normalized_entropy(..., k=K_total)`` divides by log(K_total): breadth
  relative to the whole category space, comparable across all users.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def shannon_entropy(counts: np.ndarray | pd.Series) -> float:
    """H = -sum p_i log p_i over categories with p_i > 0 (natural log)."""
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return float("nan")
    p = c / c.sum()
    return float(-(p * np.log(p)).sum())


def normalized_entropy(counts: np.ndarray | pd.Series, k: int | None = None) -> float:
    """Entropy divided by log(k); ``k`` defaults to the number of non-zero categories."""
    c = np.asarray(counts, dtype=float)
    k_eff = int((c > 0).sum()) if k is None else int(k)
    if k_eff <= 1:
        return float("nan")
    return shannon_entropy(c) / np.log(k_eff)


def hhi(counts: np.ndarray | pd.Series) -> float:
    """Herfindahl–Hirschman index sum p_i^2 (1 = all in one category)."""
    c = np.asarray(counts, dtype=float)
    if c.sum() == 0:
        return float("nan")
    p = c / c.sum()
    return float((p**2).sum())


def dominant_share(counts: np.ndarray | pd.Series) -> float:
    """Share of the single largest category."""
    c = np.asarray(counts, dtype=float)
    return float(c.max() / c.sum()) if c.sum() > 0 else float("nan")


def group_diversity(df: pd.DataFrame, group_col: str, cat_col: str, k_total: int) -> pd.DataFrame:
    """Per-group entropy, normalised entropy (by K_total), HHI, distinct categories, n."""
    counts = df.groupby([group_col, cat_col]).size().unstack(fill_value=0)
    arr = counts.to_numpy(dtype=float)
    out = pd.DataFrame(index=counts.index)
    out["n"] = arr.sum(axis=1).astype(int)
    out["n_categories"] = (arr > 0).sum(axis=1)
    out["entropy"] = [shannon_entropy(r) for r in arr]
    out["entropy_norm_total"] = out["entropy"] / np.log(k_total)
    out["hhi"] = [hhi(r) for r in arr]
    out["dominant_share"] = [dominant_share(r) for r in arr]
    return out
