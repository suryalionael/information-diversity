"""Diversity / concentration measures over categorical distributions.

Entropy uses the natural log. Two normalisations are provided because they
answer different questions:

* ``normalized_entropy(..., k=None)`` divides by log(K_observed): evenness among
  the categories a user actually touched (undefined for K_observed = 1).
* ``normalized_entropy(..., k=K_total)`` divides by log(K_total): breadth
  relative to the whole category space, comparable across all users.

Plug-in entropy and plug-in HHI are biased by sample size (a 3-click history
looks artificially narrow). The **unbiased Simpson index**

    S = sum_i n_i (n_i - 1) / (N (N - 1)),   N >= 2

is the probability that two items drawn *without replacement* from the user's
list share a category. Its expectation equals the population HHI for any N, so
it is the primary concentration measure for comparing users with different
history lengths. ``1 / S`` is read as the *effective number of categories*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Scalar measures (one distribution)
# --------------------------------------------------------------------------- #
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
    """Plug-in Herfindahl–Hirschman index sum p_i^2 (1 = all in one category)."""
    c = np.asarray(counts, dtype=float)
    if c.sum() == 0:
        return float("nan")
    p = c / c.sum()
    return float((p**2).sum())


def simpson_unbiased(counts: np.ndarray | pd.Series) -> float:
    """Unbiased Simpson concentration sum n_i(n_i-1) / (N(N-1)); NaN if N < 2."""
    c = np.asarray(counts, dtype=float)
    n = c.sum()
    if n < 2:
        return float("nan")
    return float((c * (c - 1)).sum() / (n * (n - 1)))


def dominant_share(counts: np.ndarray | pd.Series) -> float:
    """Share of the single largest category."""
    c = np.asarray(counts, dtype=float)
    return float(c.max() / c.sum()) if c.sum() > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Vectorised per-group profiles
# --------------------------------------------------------------------------- #
def count_matrix(df: pd.DataFrame, group_col: str, cat_col: str,
                 categories: list[str] | None = None) -> pd.DataFrame:
    """Groups x categories count table (columns fixed to ``categories`` if given)."""
    counts = df.groupby([group_col, cat_col], observed=True).size().unstack(fill_value=0)
    if categories is not None:
        counts = counts.reindex(columns=categories, fill_value=0)
    return counts


def profile_metrics(counts: pd.DataFrame, k_norm: int, prefix: str = "") -> pd.DataFrame:
    """Per-row concentration/diversity measures from a groups x categories count table.

    Columns: n, n_categories, entropy, entropy_norm (H / log k_norm), hhi (plug-in),
    simpson (unbiased; NaN if n < 2), eff_categories (1 / simpson), dominant_category
    (ties broken by column order), dominant_share.
    """
    c = counts.to_numpy(dtype=float)
    n = c.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = c / n[:, None]
        plogp = np.where(p > 0, p * np.log(np.where(p > 0, p, 1.0)), 0.0)
        ent = 0.0 - plogp.sum(axis=1) + 0.0  # +0.0 avoids printing -0.0
        simpson = np.where(n >= 2, (c * (c - 1)).sum(axis=1) / (n * (n - 1)), np.nan)
        out = pd.DataFrame({
            "n": n.astype(int),
            "n_categories": (c > 0).sum(axis=1),
            "entropy": np.where(n > 0, ent, np.nan),
            "entropy_norm": np.where(n > 0, ent / np.log(k_norm), np.nan),
            "hhi": np.where(n > 0, (p**2).sum(axis=1), np.nan),
            "simpson": simpson,
            "eff_categories": np.where(simpson > 0, 1.0 / simpson, np.nan),
            "dominant_category": np.asarray(counts.columns)[c.argmax(axis=1)],
            "dominant_share": np.where(n > 0, c.max(axis=1) / n, np.nan),
        }, index=counts.index)
    out.loc[out["n"] == 0, "dominant_category"] = None
    return out.add_prefix(prefix)


def group_diversity(df: pd.DataFrame, group_col: str, cat_col: str, k_total: int) -> pd.DataFrame:
    """Backward-compatible wrapper: per-group metrics normalised by K_total."""
    return profile_metrics(count_matrix(df, group_col, cat_col), k_total).rename(
        columns={"entropy_norm": "entropy_norm_total"})


# --------------------------------------------------------------------------- #
# Alignment of exposure with historical preference (permutation null)
# --------------------------------------------------------------------------- #
def permutation_alignment(shares: np.ndarray, dom: np.ndarray, strata: np.ndarray,
                          groups: np.ndarray | None = None, n_perm: int = 200,
                          seed: int = 42) -> dict:
    """Is a user's exposure tilted toward their *own* historically dominant category?

    Parameters
    ----------
    shares : (users x K) exposure category shares (rows sum to 1).
    dom    : (users,) column index of each user's historically dominant category.
    strata : (users,) labels; dominant categories are shuffled only *within* a stratum
             (e.g. split membership x activity level) so the null keeps period/activity mix.
    groups : optional (users,) integer codes for per-group summaries (0..G-1).

    Returns observed per-user values, observed mean, null mean, null 95% interval,
    one-sided p-value (null >= observed), and per-group observed/null means.

    Null hypothesis: exposure composition is unrelated to which category a user
    favoured historically (given their stratum).
    """
    shares = np.asarray(shares, dtype=float)
    dom = np.asarray(dom)
    n = dom.size
    rows = np.arange(n)
    obs = shares[rows, dom]
    rng = np.random.default_rng(seed)
    strata_idx = [np.flatnonzero(strata == s) for s in pd.unique(strata)]
    if groups is not None:
        groups = np.asarray(groups, dtype=int)
        g_n = np.bincount(groups)
        g_obs = np.bincount(groups, weights=obs) / g_n
        g_null = np.empty((n_perm, g_n.size))
    null = np.empty(n_perm)
    dom_p = dom.copy()
    for b in range(n_perm):
        for idx in strata_idx:
            dom_p[idx] = dom[idx][rng.permutation(idx.size)]
        vals = shares[rows, dom_p]
        null[b] = vals.mean()
        if groups is not None:
            g_null[b] = np.bincount(groups, weights=vals, minlength=g_n.size) / g_n
    out = {
        "observed": obs, "observed_mean": float(obs.mean()), "null_mean": float(null.mean()),
        "null_low": float(np.quantile(null, 0.025)), "null_high": float(np.quantile(null, 0.975)),
        "lift": float(obs.mean() / null.mean()),
        "p_one_sided": float((np.sum(null >= obs.mean()) + 1) / (n_perm + 1)), "n_users": int(n),
        "n_perm": n_perm,
    }
    if groups is not None:
        out.update({"group_observed": g_obs, "group_null_mean": g_null.mean(axis=0),
                    "group_null_low": np.quantile(g_null, 0.025, axis=0),
                    "group_null_high": np.quantile(g_null, 0.975, axis=0), "group_n": g_n})
    return out
