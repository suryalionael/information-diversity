"""Simulated recommendation strategies and their evaluation (Notebook 03).

These are *experiments* on the MIND candidate lists. They do not reproduce, and
must never be described as reproducing, MSN's proprietary recommender.

Leakage rules (see CLAUDE.md):
* popularity      = click counts from **train** impressions only;
* user profiles   = each user's pre-period ``history`` field (identical in train/dev);
* TF-IDF          = fitted on article titles only (no labels);
* dev clicks      = evaluation labels only, never model input.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 42


# --------------------------------------------------------------------------- #
# Ranking metrics (one impression)
# --------------------------------------------------------------------------- #
def ndcg_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    """nDCG@k with binary gains; ties in ``scores`` are resolved by position (break them first)."""
    order = np.argsort(-scores, kind="stable")
    gains = labels[order][:k]
    disc = 1.0 / np.log2(np.arange(2, gains.size + 2))
    dcg = float((gains * disc).sum())
    ideal = np.sort(labels)[::-1][:k]
    idcg = float((ideal * disc[: ideal.size]).sum())
    return dcg / idcg if idcg > 0 else float("nan")


def auc_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via the rank-sum formula (ties count one half). NaN without both classes."""
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(scores)
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mrr_score(labels: np.ndarray, scores: np.ndarray) -> float:
    """MIND-style MRR: mean reciprocal rank over clicked items."""
    order = np.argsort(-scores, kind="stable")
    rr = labels[order] / np.arange(1, labels.size + 1)
    return float(rr.sum() / labels.sum()) if labels.sum() > 0 else float("nan")


def recall_at_k(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> float:
    order = np.argsort(-scores, kind="stable")
    return float(labels[order][:k].sum() / labels.sum()) if labels.sum() > 0 else float("nan")


def rank_metrics(labels: np.ndarray, scores: np.ndarray, k: int = 10) -> tuple[float, float, float, float]:
    """(nDCG@k, AUC, MRR, recall@k) from a single sort. Requires *unique* scores (use ``break_ties``).

    Equivalent to the individual functions above when scores have no ties (verified in tests).
    """
    n = labels.size
    order = np.argsort(-scores, kind="stable")
    lab = labels[order]
    n_pos = lab.sum()
    if n_pos == 0:
        return (float("nan"),) * 4
    pos_rank = np.flatnonzero(lab) + 1                      # 1-based ranks of positives
    disc = 1.0 / np.log2(np.arange(2, n + 2))
    dcg = float(disc[pos_rank[pos_rank <= k] - 1].sum())
    idcg = float(disc[: int(min(n_pos, k))].sum())
    n_neg = n - n_pos
    # AUC: for each positive, count negatives ranked below it
    auc = float(((n - pos_rank) - (n_pos - np.arange(1, n_pos + 1))).sum() / (n_pos * n_neg)) if n_neg else float("nan")
    return dcg / idcg, auc, float((1.0 / pos_rank).sum() / n_pos), float((pos_rank <= k).sum() / n_pos)


def break_ties(scores: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return rank-valued scores (higher = better) with exact ties broken by seeded random jitter.

    MIND does not document the order of articles inside an impression, so list position must
    never act as a tie-breaker.
    """
    jitter = rng.random(scores.size)
    order = np.lexsort((jitter, scores))          # ascending: last = best
    out = np.empty(scores.size)
    out[order] = np.arange(scores.size)
    return out


# --------------------------------------------------------------------------- #
# MMR re-ranking
# --------------------------------------------------------------------------- #
def mmr_rerank(relevance: np.ndarray, sim: np.ndarray, lam: float, k: int) -> np.ndarray:
    """Greedy Maximal Marginal Relevance.

    Picks, one at a time, argmax_i [ lam * rel_i - (1 - lam) * max_{j in selected} sim(i, j) ].
    ``relevance`` should be scaled to [0, 1] within the list. Exact ties (e.g. lam = 0 on the
    first pick) are broken by relevance. Returns the indices of the k selected items in order.
    """
    n = relevance.size
    k = min(k, n)
    chosen = np.empty(k, dtype=int)
    max_sim = np.zeros(n)
    avail = np.ones(n, dtype=bool)
    tie = 1e-9 * relevance
    for t in range(k):
        mmr = lam * relevance - (1 - lam) * max_sim + tie
        mmr[~avail] = -np.inf
        i = int(np.argmax(mmr))
        chosen[t] = i
        avail[i] = False
        max_sim = np.maximum(max_sim, sim[i])
    return chosen


def order_to_scores(top: np.ndarray, base_scores: np.ndarray) -> np.ndarray:
    """Full ranking scores: the re-ranked top-k first, then the rest by base score."""
    n = base_scores.size
    mask = np.ones(n, dtype=bool)
    mask[top] = False
    rest = np.flatnonzero(mask)
    rest = rest[np.argsort(-base_scores[rest], kind="stable")]
    full = np.concatenate([top, rest])
    out = np.empty(n)
    out[full] = np.arange(n, 0, -1)
    return out


# --------------------------------------------------------------------------- #
# Model inputs
# --------------------------------------------------------------------------- #
@dataclass
class Context:
    """Article-level arrays indexed by an integer article index."""

    news_ids: np.ndarray            # index -> news_id
    index: dict                     # news_id -> index
    X: sparse.csr_matrix            # L2-normalised TF-IDF title vectors
    cat: np.ndarray                 # index -> category code
    categories: list                # code -> category name
    pop: np.ndarray                 # train-period clicks per article
    vectorizer: TfidfVectorizer


def build_context(catalogue: pd.DataFrame, train_exposures: pd.DataFrame) -> Context:
    """TF-IDF over titles (fit on titles only) + train-only click popularity."""
    news_ids = catalogue["news_id"].to_numpy()
    index = {n: i for i, n in enumerate(news_ids)}
    vec = TfidfVectorizer(lowercase=True, stop_words="english", min_df=2, sublinear_tf=True,
                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
    X = normalize(vec.fit_transform(catalogue["title"].fillna("")), norm="l2").tocsr()
    cats = sorted(catalogue["category"].unique())
    cat = catalogue["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    clicks = train_exposures.loc[train_exposures["clicked"] == 1, "news_id"].value_counts()
    pop = np.zeros(len(news_ids))
    idx = clicks.index.map(index)
    pop[idx.to_numpy(dtype=int)] = clicks.to_numpy()
    return Context(news_ids, index, X, cat, cats, pop, vec)


def user_profiles(history: pd.DataFrame, ctx: Context) -> tuple[dict, dict, np.ndarray]:
    """Per-user TF-IDF centroid (L2-normalised) and category counts from history clicks.

    Returns (content_profile, category_counts, global_category_prior).
    """
    h = history.assign(idx=history["news_id"].map(ctx.index)).dropna(subset=["idx"])
    h["idx"] = h["idx"].astype(int)
    content, counts = {}, {}
    K = len(ctx.categories)
    for u, g in h.groupby("user_id", sort=False):
        ii = g["idx"].to_numpy()
        centroid = np.asarray(ctx.X[ii].mean(axis=0)).ravel()
        nrm = np.linalg.norm(centroid)
        content[u] = centroid / nrm if nrm > 0 else centroid
        counts[u] = np.bincount(ctx.cat[ii], minlength=K).astype(float)
    prior = np.bincount(ctx.cat[h["idx"].to_numpy()], minlength=K).astype(float)
    return content, counts, prior / prior.sum()


# --------------------------------------------------------------------------- #
# Scoring functions: (candidate indices, user) -> raw scores
# --------------------------------------------------------------------------- #
def score_popularity(cand: np.ndarray, ctx: Context, **_) -> np.ndarray:
    return np.log1p(ctx.pop[cand])


def score_content(cand: np.ndarray, ctx: Context, profile: np.ndarray, **_) -> np.ndarray:
    return ctx.X[cand] @ profile


def score_category(cand: np.ndarray, ctx: Context, cat_counts: np.ndarray, prior: np.ndarray,
                   alpha: float = 1.0, **_) -> np.ndarray:
    """Smoothed share of the user's history in the candidate's category, popularity as tie-break.

    p(c | u) = (n_uc + alpha * prior_c) / (n_u + alpha). The popularity term is scaled to
    < 1e-6 so it only orders articles within the same category.
    """
    p = (cat_counts + alpha * prior) / (cat_counts.sum() + alpha)
    pop = np.log1p(ctx.pop[cand])
    return p[ctx.cat[cand]] + 1e-6 * pop / (pop.max() + 1.0)


def score_random(cand: np.ndarray, ctx: Context, rng: np.random.Generator, **_) -> np.ndarray:
    return rng.random(cand.size)


# --------------------------------------------------------------------------- #
# List-level diversity
# --------------------------------------------------------------------------- #
def list_diversity(top: np.ndarray, cand_cat: np.ndarray, title_sim: np.ndarray) -> dict:
    """Diversity of one recommended list (``top`` = local indices into the candidate list).

    * distinct_categories  number of categories in the list
    * category_entropy     Shannon entropy (nats) of the list's categories
    * ild                  intra-list diversity: mean pairwise (1 - title cosine similarity)
    """
    c = np.bincount(cand_cat[top])
    c = c[c > 0]
    p = c / c.sum()
    n = top.size
    if n > 1:
        sub = title_sim[top][:, top]
        ild = float(1.0 - (sub.sum() - np.trace(sub)) / (n * (n - 1)))   # symmetric: mean of off-diagonal pairs
    else:
        ild = float("nan")
    return {"distinct_categories": int(c.size), "category_entropy": float(-(p * np.log(p)).sum() + 0.0),
            "ild": ild}


# --------------------------------------------------------------------------- #
# Evaluation loop
# --------------------------------------------------------------------------- #
SCORERS = {"popularity": score_popularity, "content": score_content,
           "category": score_category, "random": score_random}


def _minmax(x: np.ndarray) -> np.ndarray:
    span = x.max() - x.min()
    return (x - x.min()) / span if span > 0 else np.zeros_like(x, dtype=float)


def evaluate(impressions: pd.DataFrame, ctx: Context, content: dict, cat_counts: dict, prior: np.ndarray,
             head: np.ndarray, models: list[str], mmr_bases: list[str], lambdas: list[float],
             k: int = 10, beta: float = 0.5, min_hist_amp: int = 5, seed: int = SEED) -> tuple[pd.DataFrame, dict]:
    """Score, rank and measure every impression for every model (and MMR variant).

    impressions : exposure records with imp_key, user_id, news_id, clicked (one row per candidate).
    head        : boolean per article index, True for "head" (most-exposed) articles.
    MMR redundancy sim(i, j) = beta * [same category] + (1 - beta) * title cosine.

    Returns (records, recommended) where records has one row per (impression, model, lambda)
    and recommended maps (model, lambda) -> set of recommended article indices (for coverage).
    """
    rng = np.random.default_rng(seed)
    K = len(ctx.categories)
    zero_profile = np.zeros(ctx.X.shape[1])
    recs, recommended = [], {}
    imp = impressions.assign(idx=impressions["news_id"].map(ctx.index))
    if imp["idx"].isna().any():
        raise ValueError("candidate articles missing from catalogue context")
    for key, g in imp.groupby("imp_key", sort=False):
        cand = g["idx"].to_numpy(dtype=int)
        labels = g["clicked"].to_numpy(dtype=float)
        user = g["user_id"].iat[0]
        n = cand.size
        cand_cat = ctx.cat[cand]
        Xc = ctx.X[cand]
        title_sim = (Xc @ Xc.T).toarray()
        red_sim = beta * (cand_cat[:, None] == cand_cat[None, :]) + (1 - beta) * title_sim
        tail = ~head[cand]
        counts = cat_counts.get(user, np.zeros(K))
        if counts.sum() >= min_hist_amp:
            dom = int(np.argmax(counts)); hist_dom = counts[dom] / counts.sum()
            pool_dom = float(np.mean(cand_cat == dom))
        else:
            dom, hist_dom, pool_dom = -1, np.nan, np.nan
        kw = {"ctx": ctx, "profile": content.get(user, zero_profile), "cat_counts": counts, "prior": prior, "rng": rng}

        def measure(model, lam, scores, top):
            d = list_diversity(top, cand_cat, title_sim)
            rec_dom = float(np.mean(cand_cat[top] == dom)) if dom >= 0 else np.nan
            nd, au, mr, rc = rank_metrics(labels, scores, k)
            recs.append((key, user, model, lam, n, labels.sum(), nd, au, mr, rc,
                         d["distinct_categories"], d["category_entropy"], d["ild"], float(tail[top].mean()),
                         rec_dom, hist_dom, pool_dom))
            recommended.setdefault((model, lam), set()).update(cand[top].tolist())

        for model in models:
            raw = np.asarray(SCORERS[model](cand, **kw), dtype=float)
            s = break_ties(raw, rng)
            top = np.argsort(-s, kind="stable")[:k]
            measure(model, 1.0, s, top)
            if model in mmr_bases:
                rel = _minmax(raw) + 1e-6 * _minmax(s)
                rel = _minmax(rel)
                for lam in lambdas:
                    if lam == 1.0:
                        continue          # identical to the base ranking (verified in tests)
                    top_m = mmr_rerank(rel, red_sim, lam, k)
                    measure(f"{model}+MMR", lam, order_to_scores(top_m, s), top_m)
    cols = ["imp_key", "user_id", "model", "lam", "n_candidates", "n_clicks", "ndcg10", "auc", "mrr", "recall10",
            "distinct_categories", "category_entropy", "ild", "tail_share", "rec_dom_share", "hist_dom_share",
            "pool_dom_share"]
    out = pd.DataFrame(recs, columns=cols)
    out["amp_vs_history"] = out["rec_dom_share"] / out["hist_dom_share"]
    out["amp_vs_pool"] = out["rec_dom_share"] / out["pool_dom_share"].where(out["pool_dom_share"] > 0)
    return out, recommended
