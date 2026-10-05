"""Exposure, engagement and concentration metrics.

Every function documents its denominator explicitly. Terminology:

* catalogue share   = articles in group / articles in the catalogue
* exposure share    = impressions of group / all impressions
* click share       = clicks in group / all clicks
* CTR               = clicks in group / impressions of group   (NOT click share)
* amplification     = exposure share / catalogue share
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------- #
# Category-level shares
# --------------------------------------------------------------------------- #
def category_table(
    catalogue: pd.DataFrame,
    exposures: pd.DataFrame,
    group_col: str = "category",
    exposed_pool: bool = True,
) -> pd.DataFrame:
    """Catalogue vs exposure vs click shares per group.

    Parameters
    ----------
    catalogue : one row per article with ``news_id`` and ``group_col``.
    exposures : exposure records (``news_id``, ``clicked``) already joined to
        ``group_col``.
    exposed_pool : also compute shares among the *exposed pool* — the unique
        articles shown at least once — as an alternative catalogue baseline.

    Returns columns
    ---------------
    n_articles, catalogue_share, impressions, exposure_share, clicks,
    click_share, ctr, ctr_ci_low, ctr_ci_high, amplification,
    click_amplification (click share / catalogue share), and optionally
    n_exposed_articles, pool_share, amplification_vs_pool.
    """
    cat = catalogue.groupby(group_col).size().rename("n_articles")
    exp = exposures.groupby(group_col).agg(
        impressions=("news_id", "size"), clicks=("clicked", "sum")
    )
    t = pd.concat([cat, exp], axis=1).fillna(0)
    t[["n_articles", "impressions", "clicks"]] = t[["n_articles", "impressions", "clicks"]].astype("int64")
    t["catalogue_share"] = t["n_articles"] / t["n_articles"].sum()
    t["exposure_share"] = t["impressions"] / t["impressions"].sum()
    t["click_share"] = t["clicks"] / t["clicks"].sum()
    t["ctr"] = t["clicks"] / t["impressions"].where(t["impressions"] > 0)
    lo, hi = wilson_interval(t["clicks"].to_numpy(), t["impressions"].to_numpy())
    t["ctr_ci_low"], t["ctr_ci_high"] = lo, hi
    t["amplification"] = t["exposure_share"] / t["catalogue_share"].where(t["catalogue_share"] > 0)
    t["click_amplification"] = t["click_share"] / t["catalogue_share"].where(t["catalogue_share"] > 0)
    if exposed_pool:
        pool = exposures.drop_duplicates("news_id").groupby(group_col).size()
        t["n_exposed_articles"] = pool.reindex(t.index).fillna(0).astype("int64")
        t["pool_share"] = t["n_exposed_articles"] / t["n_exposed_articles"].sum()
        t["amplification_vs_pool"] = t["exposure_share"] / t["pool_share"].where(t["pool_share"] > 0)
    t.index.name = group_col
    return t.sort_values("impressions", ascending=False)


def wilson_interval(successes: np.ndarray, trials: np.ndarray, conf: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Wilson score interval for a binomial proportion (NaN where trials == 0)."""
    s = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    z = stats.norm.ppf(0.5 + conf / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s / n
        denom = 1 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    lo, hi = centre - half, centre + half
    lo[n == 0] = np.nan
    hi[n == 0] = np.nan
    return lo, hi


# --------------------------------------------------------------------------- #
# Concentration
# --------------------------------------------------------------------------- #
def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative distribution (0 = equal, ->1 = concentrated).

    G = (2 * sum_i i * x_(i)) / (n * sum x) - (n + 1) / n, with x sorted ascending
    and i = 1..n. Zeros are included as given — callers decide the population.
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(2 * np.sum(i * x) / (n * x.sum()) - (n + 1) / n)


def lorenz_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative share of items (x) vs cumulative share of total (y), ascending order.

    Both arrays start at 0 and end at 1.
    """
    x = np.sort(np.asarray(values, dtype=float))
    cum = np.concatenate([[0.0], np.cumsum(x)]) / x.sum()
    pop = np.linspace(0.0, 1.0, x.size + 1)
    return pop, cum


def top_share(values: np.ndarray, top_fraction: float) -> dict:
    """Share of the total held by the top ``top_fraction`` of items.

    The number of items is ceil(top_fraction * n) so that small populations
    still include at least one item.
    """
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    n = x.size
    k = min(n, int(np.ceil(round(top_fraction * n, 9))))
    return {
        "top_fraction": top_fraction,
        "n_items_total": int(n),
        "n_items_top": k,
        "share_of_total": float(x[:k].sum() / x.sum()),
        "min_value_in_top": float(x[k - 1]) if k else float("nan"),
    }


def bottom_share(values: np.ndarray, bottom_fraction: float) -> dict:
    """Share of the total held by the bottom ``bottom_fraction`` of items (floor)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    k = min(n, int(np.floor(round(bottom_fraction * n, 9))))
    return {
        "bottom_fraction": bottom_fraction,
        "n_items_total": int(n),
        "n_items_bottom": k,
        "share_of_total": float(x[:k].sum() / x.sum()),
        "max_value_in_bottom": float(x[k - 1]) if k else float("nan"),
    }


def items_for_share(values: np.ndarray, share: float) -> dict:
    """Smallest number (and fraction) of top items needed to reach ``share`` of the total."""
    x = np.sort(np.asarray(values, dtype=float))[::-1]
    cum = np.cumsum(x) / x.sum()
    k = min(x.size, int(np.searchsorted(cum, share - 1e-12) + 1))
    return {"target_share": share, "n_items": k, "fraction_of_items": k / x.size}


# --------------------------------------------------------------------------- #
# Headline metric registry
# --------------------------------------------------------------------------- #
@dataclass
class Headline:
    """One reportable statistic with full provenance (CLAUDE.md, Rule 1)."""

    metric: str
    value: float | int | str
    denominator: str
    population: str
    filters: str
    source: str
    unit: str = ""
    notes: str = ""
    calculated_on: str = field(default_factory=lambda: date.today().isoformat())


class HeadlineRegistry:
    """Collects ``Headline`` records and writes them to JSON."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.records: list[Headline] = []

    def add(self, metric: str, value, denominator: str, population: str,
            filters: str = "none", unit: str = "", notes: str = "") -> Headline:
        if isinstance(value, (np.integer,)):
            value = int(value)
        elif isinstance(value, (np.floating,)):
            value = float(value)
        rec = Headline(metric, value, denominator, population, filters, self.source, unit, notes)
        self.records = [r for r in self.records if r.metric != metric] + [rec]
        return rec

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(r) for r in self.records])

    def save(self, path: Path) -> None:
        """Merge into an existing JSON file, replacing records from the same source."""
        path = Path(path)
        existing = []
        if path.exists():
            existing = [r for r in json.loads(path.read_text()) if r.get("source") != self.source]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(existing + [asdict(r) for r in self.records], indent=2))


# --------------------------------------------------------------------------- #
# Uncertainty helpers (deterministic: seeded)
# --------------------------------------------------------------------------- #
SEED = 42


def bootstrap_ci(values: np.ndarray, stat=np.mean, n_boot: int = 1000, conf: float = 0.95,
                 seed: int = SEED, batch: int = 50) -> tuple[float, float, float]:
    """(estimate, low, high) percentile bootstrap CI of ``stat`` over ``values`` (NaNs dropped).

    Resamples are drawn in batches so memory stays bounded for large inputs.
    """
    x = np.asarray(values, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for start in range(0, n_boot, batch):
        m = min(batch, n_boot - start)
        sample = x[rng.integers(0, x.size, size=(m, x.size))]
        boots[start:start + m] = sample.mean(axis=1) if stat is np.mean else np.apply_along_axis(stat, 1, sample)
    a = (1 - conf) / 2
    return float(stat(x)), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 200, conf: float = 0.95,
                seed: int = SEED) -> dict:
    """Spearman rho with p-value and a percentile bootstrap CI (pairs with NaN dropped)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"rho": float("nan"), "p": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": int(x.size)}
    res = stats.spearmanr(x, y)
    # rank once, then bootstrap Pearson-on-ranks (equivalent up to tie handling within resamples)
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, x.size, x.size)
        boots[b] = np.corrcoef(rx[i], ry[i])[0, 1]
    a = (1 - conf) / 2
    return {"rho": float(res.statistic), "p": float(res.pvalue), "ci_low": float(np.nanquantile(boots, a)),
            "ci_high": float(np.nanquantile(boots, 1 - a)), "n": int(x.size)}


def tiebreak_rank(values, seed: int = SEED) -> np.ndarray:
    """Ranks 0..n-1 by value with exact ties broken by a seeded random key (never by row order).

    Use before quantile binning of discrete measures so tied units are split at random.
    """
    v = np.asarray(values, dtype=float)
    key = np.random.default_rng(seed).random(v.size)
    order = np.lexsort((key, v))
    r = np.empty(v.size, dtype=int)
    r[order] = np.arange(v.size)
    return r


def cluster_bootstrap_mean(values: np.ndarray, clusters: np.ndarray, n_boot: int = 1000, conf: float = 0.95,
                           seed: int = SEED) -> tuple[float, float, float]:
    """Mean of ``values`` with a CI that resamples whole clusters (e.g. users with several impressions).

    The estimate is the plain mean over rows; each bootstrap replicate draws clusters with replacement.
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "c": np.asarray(clusters)}).dropna()
    if df.empty:
        return (float("nan"),) * 3
    g = df.groupby("c")["v"].agg(["sum", "size"])
    sums, sizes = g["sum"].to_numpy(), g["size"].to_numpy()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b0 in range(0, n_boot, 50):
        m = min(50, n_boot - b0)
        idx = rng.integers(0, sums.size, size=(m, sums.size))
        boots[b0:b0 + m] = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    a = (1 - conf) / 2
    return float(df["v"].mean()), float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))
