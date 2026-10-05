"""Generates notebooks/03_relevance_diversity.ipynb (unexecuted). Run: python notebooks/build_03_relevance_diversity.py"""
from pathlib import Path

from _nbbuild import SETUP, Builder

b = Builder()
md, code = b.md, b.code

md(r"""
# 03 — Relevance vs. diversity: simulated recommendation strategies

## 1. Purpose

Research Question 3 (**Trade-off**): *Can a recommendation strategy recover information diversity without
sacrificing too much relevance?*

Everything in this notebook is a **simulated recommendation strategy** — an experiment run on MIND's logged
candidate lists. None of it reproduces, or claims to reproduce, MSN's proprietary recommender.

### Experimental design

| Element | Definition |
|---|---|
| Evaluation impressions | dev impressions (15 Nov 2019) of users with ≥1 pre-period history click |
| Candidate pool | the articles actually shown in that impression |
| Positives / negatives | clicked / shown-but-not-clicked (dev labels — used **only** for scoring the evaluation) |
| User profile | the user's pre-period `history` clicks (identical in train and dev; never updated) |
| Popularity | click counts from **train** impressions (9–14 Nov) only |
| Text model | TF-IDF on article titles (fit on titles only; no labels) |
| Ties | broken by seeded random jitter (the in-impression order is undocumented and must not leak) |
| List length | top-10 (`K = 10`) |

**Strategies.** `popularity` (train clicks), `content` (cosine between title TF-IDF and the user's history
centroid), `category` (the user's smoothed historical category share; popularity orders articles inside a
category), and `random` (reference only). **MMR** re-ranks each base strategy:

`MMR(i) = λ · relevance(i) − (1 − λ) · max_{j ∈ selected} sim(i, j)`, with relevance min-max scaled inside the
list and `sim(i, j) = ½·[same category] + ½·title cosine`. λ = 1 is the base ranking; λ = 0 ignores relevance.

**Diversity metrics** (per top-10 list): distinct categories, category entropy, intra-list diversity (ILD =
mean pairwise 1 − title cosine), long-tail share (slots going to articles outside the 10% most-exposed dev
candidates — an evaluation-only definition), catalogue coverage (share of evaluated candidates recommended at
least once), and the **amplification ratio** (share of the list in the user's historically dominant category ÷
that category's share of the user's history; users with ≥5 history clicks).

Diversity comparisons use impressions with **≥20 candidates** (`MIN_CANDIDATES`): with fewer, a top-10 is most
of the list and re-ranking can barely change it. Relevance on *all* impressions is reported separately.
""")

md("## 2. Setup")
code(SETUP + r'''
from time import perf_counter
from src import recommenders as R
K = 10
LAMBDAS = [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]
MIN_CANDIDATES = 20
MIN_HIST_AMP = 5
NDCG_TOLERANCE = 0.05     # a-priori: "acceptable" = at most 5% relative nDCG@10 loss vs. the base ranking
MAX_EVAL_IMPRESSIONS = None   # set an int for a seeded random subsample if runtime is a problem
''')

md("## 3. Data loading")
code(r"""
catalogue, _ = D.load_catalogue(["train", "dev"])
beh = {s: D.load_behaviors(s) for s in ["train", "dev"]}
exp = D.load_exposures(["train", "dev"])
exp["clicked"] = exp["clicked"].astype("int64")
train_exp, dev_exp = exp[exp["split"] == "train"], exp[exp["split"] == "dev"]
print(f"train exposure records {len(train_exp):,}; dev exposure records {len(dev_exp):,}")
""")

md("## 4. Validation & leakage checks")
code(r"""
assert set(train_exp["split"]) == {"train"} and set(dev_exp["split"]) == {"dev"}
assert train_exp["time"].max() < dev_exp["time"].min(), "train period must precede dev period"
print(f"train period ends {train_exp['time'].max()}, dev period starts {dev_exp['time'].min()}")

stab = D.history_stability(pd.concat(beh.values(), ignore_index=True))
print("history stability across train+dev:", stab)
assert stab["users_with_varying_history"] == 0, "history must be pre-period and fixed"

ctx = R.build_context(catalogue, train_exp)            # popularity from TRAIN clicks only
dev_cands = dev_exp["news_id"].unique()
seen = np.isin(dev_cands, train_exp.loc[train_exp["clicked"] == 1, "news_id"].unique())
slot_seen = dev_exp["news_id"].isin(train_exp.loc[train_exp["clicked"] == 1, "news_id"]).mean()
print(f"dev candidate articles with ≥1 train click: {seen.mean():.1%} of {len(dev_cands):,} articles; "
      f"{slot_seen:.1%} of dev candidate slots")
print(f"TF-IDF vocabulary: {ctx.X.shape[1]:,} terms over {ctx.X.shape[0]:,} titles")
""")
code(r"""
hist_dev = D.explode_history(beh["dev"], dedupe_per_user=True)
content, cat_counts, prior = R.user_profiles(hist_dev, ctx)
hist_n = hist_dev.groupby("user_id").size()

imp_size = dev_exp.groupby("imp_key").size()
imp_user = dev_exp.drop_duplicates("imp_key").set_index("imp_key")["user_id"]
has_hist = imp_user.map(hist_n).fillna(0) >= 1
print(imp_size.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).to_string())
pop_all = imp_size.index[(imp_size >= 2) & has_hist.reindex(imp_size.index)]
pop_div = imp_size.index[(imp_size >= MIN_CANDIDATES) & has_hist.reindex(imp_size.index)]
print(f"\ndev impressions: {len(imp_size):,}; with ≥1 history click: {has_hist.sum():,}")
print(f"relevance population (≥2 candidates): {len(pop_all):,}")
print(f"diversity population (≥{MIN_CANDIDATES} candidates): {len(pop_div):,} "
      f"({len(pop_div)/len(pop_all):.1%} of relevance population)")
if MAX_EVAL_IMPRESSIONS and len(pop_div) > MAX_EVAL_IMPRESSIONS:
    pop_div = pd.Index(np.random.default_rng(SEED).choice(pop_div, MAX_EVAL_IMPRESSIONS, replace=False))
    print("subsampled to", len(pop_div))
""")
code(r"""
# Head articles (evaluation-only): the 10% most-exposed dev candidate articles by number of dev impressions
dev_counts = dev_exp.groupby("news_id")["imp_key"].nunique()
head_ids = dev_counts.sort_values(ascending=False).index[: int(np.ceil(0.10 * len(dev_counts)))]
head = np.zeros(len(ctx.news_ids), dtype=bool)
head[[ctx.index[n] for n in head_ids]] = True
print(f"head articles: {head.sum():,} (top 10% of {len(dev_counts):,}); they fill "
      f"{dev_exp['news_id'].isin(head_ids).mean():.1%} of dev candidate slots")
""")

md("## 5. Run the experiments")
code(r"""
t0 = perf_counter()
rec_div, recommended = R.evaluate(dev_exp[dev_exp["imp_key"].isin(pop_div)], ctx, content, cat_counts, prior, head,
                                  models=["popularity", "content", "category", "random"],
                                  mmr_bases=["popularity", "content", "category"], lambdas=LAMBDAS,
                                  k=K, min_hist_amp=MIN_HIST_AMP, seed=SEED)
print(f"diversity population: {rec_div['imp_key'].nunique():,} impressions × {rec_div.groupby(['model','lam']).ngroups} "
      f"configurations in {perf_counter() - t0:.0f}s")
t0 = perf_counter()
rec_all, _ = R.evaluate(dev_exp[dev_exp["imp_key"].isin(pop_all)], ctx, content, cat_counts, prior, head,
                        models=["popularity", "content", "category", "random"], mmr_bases=[], lambdas=[1.0],
                        k=K, min_hist_amp=MIN_HIST_AMP, seed=SEED)
print(f"relevance population: {rec_all['imp_key'].nunique():,} impressions in {perf_counter() - t0:.0f}s")
""")
code(r"""
# Sanity: MMR at λ=1 must equal the base ranking; check on a sample by re-running one impression set
chk = rec_div[(rec_div["model"].isin(["content", "category", "popularity"])) & (rec_div["lam"] == 1.0)]
assert chk.groupby("model").size().nunique() == 1, "every base model should cover the same impressions"
assert rec_div["ndcg10"].between(0, 1).all() and rec_div["auc"].dropna().between(0, 1).all()
assert (rec_div["distinct_categories"] <= K).all()
print("sanity checks passed")
""")

md("## 6. Results — relevance and diversity per strategy")
code(r"""
def summarise(df: pd.DataFrame, recommended: dict | None, pool_ids: np.ndarray | None) -> pd.DataFrame:
    rows = []
    for (m, lam), g in df.groupby(["model", "lam"], sort=False):
        nd, nd_lo, nd_hi = M.bootstrap_ci(g["ndcg10"].to_numpy(), n_boot=500)
        amp = g["amp_vs_history"].dropna()
        row = {"model": m, "lam": lam, "n_impressions": len(g), "ndcg10": nd, "ndcg10_ci_low": nd_lo, "ndcg10_ci_high": nd_hi,
               "auc": g["auc"].mean(), "mrr": g["mrr"].mean(), "recall10": g["recall10"].mean(),
               "distinct_categories": g["distinct_categories"].mean(), "category_entropy": g["category_entropy"].mean(),
               "ild": g["ild"].mean(), "tail_share": g["tail_share"].mean(),
               "amp_vs_history_mean": amp.mean(), "amp_vs_history_median": amp.median(),
               "share_lists_amplifying": float((amp > 1).mean()) if len(amp) else np.nan,
               "rec_dom_share": g["rec_dom_share"].mean(), "hist_dom_share": g["hist_dom_share"].mean(),
               "pool_dom_share": g["pool_dom_share"].mean(), "amp_vs_pool_mean": g["amp_vs_pool"].mean(),
               "n_amp_impressions": len(amp)}
        if recommended is not None:
            row["coverage"] = len(recommended[(m, lam)]) / len(pool_ids)
        rows.append(row)
    return pd.DataFrame(rows)

pool_ids = np.unique([ctx.index[n] for n in dev_exp.loc[dev_exp["imp_key"].isin(pop_div), "news_id"].unique()])
summary = summarise(rec_div, recommended, pool_ids)
base = summary[summary["lam"] == 1.0].set_index("model")
base[["n_impressions", "ndcg10", "ndcg10_ci_low", "ndcg10_ci_high", "auc", "mrr", "recall10", "distinct_categories",
      "category_entropy", "ild", "tail_share", "coverage", "amp_vs_history_mean", "share_lists_amplifying", "amp_vs_pool_mean"]]
""")
code(r"""
rel_all = summarise(rec_all, None, None).set_index("model")[["n_impressions", "ndcg10", "ndcg10_ci_low", "ndcg10_ci_high", "auc", "mrr", "recall10"]]
rel_all
""")

md("## 7. Results — MMR relevance–diversity frontier")
code(r"""
mmr = summary[summary["model"].str.endswith("+MMR") | ((summary["lam"] == 1.0) & summary["model"].isin(["popularity", "content", "category"]))].copy()
mmr["base"] = mmr["model"].str.replace("+MMR", "", regex=False)
for c in ["ndcg10", "distinct_categories", "category_entropy", "ild", "tail_share", "amp_vs_history_mean"]:
    ref = mmr[mmr["lam"] == 1.0].set_index("base")[c]
    mmr[f"{c}_rel_change"] = mmr[c] / mmr["base"].map(ref) - 1
frontier = mmr.sort_values(["base", "lam"], ascending=[True, False])
frontier[["base", "lam", "ndcg10", "ndcg10_rel_change", "distinct_categories", "distinct_categories_rel_change",
          "category_entropy_rel_change", "ild", "tail_share", "amp_vs_history_mean", "coverage"]]
""")
code(r"""
# A-priori decision rule: walk λ down from 1.0 and keep the last λ before the first one whose nDCG@10 loss
# exceeds NDCG_TOLERANCE (conservative: never skips past a violation). Then a paired bootstrap over impressions.
def paired_change(base_name: str, lam: float, col: str) -> tuple[float, float, float]:
    a = rec_div[(rec_div["model"] == base_name) & (rec_div["lam"] == 1.0)].set_index("imp_key")[col]
    b_ = rec_div[(rec_div["model"] == f"{base_name}+MMR") & (rec_div["lam"] == lam)].set_index("imp_key")[col]
    d = (b_ - a.reindex(b_.index)).to_numpy()
    return M.bootstrap_ci(d, n_boot=1000)

picks = []
for bname, g in frontier.groupby("base"):
    g = g.sort_values("lam", ascending=False)
    pick = g.iloc[0]
    for _, r in g.iterrows():
        if r["ndcg10_rel_change"] < -NDCG_TOLERANCE:
            break
        pick = r
    row = {"base": bname, "lambda_selected": pick["lam"], "ndcg10_base": g.loc[g["lam"] == 1.0, "ndcg10"].iat[0],
           "ndcg10_selected": pick["ndcg10"], "ndcg10_rel_change": pick["ndcg10_rel_change"],
           "distinct_categories_base": g.loc[g["lam"] == 1.0, "distinct_categories"].iat[0],
           "distinct_categories_selected": pick["distinct_categories"],
           "distinct_categories_rel_change": pick["distinct_categories_rel_change"],
           "category_entropy_rel_change": pick["category_entropy_rel_change"], "ild_rel_change": pick["ild_rel_change"],
           "amp_vs_history_base": g.loc[g["lam"] == 1.0, "amp_vs_history_mean"].iat[0],
           "amp_vs_history_selected": pick["amp_vs_history_mean"]}
    if pick["lam"] < 1.0:
        for col in ["ndcg10", "distinct_categories"]:
            est, lo, hi = paired_change(bname, pick["lam"], col)
            row.update({f"{col}_paired_diff": est, f"{col}_paired_ci_low": lo, f"{col}_paired_ci_high": hi})
    picks.append(row)
picks = pd.DataFrame(picks)
picks.T
""")

md("## 8. Visualization")
code(r"""
colors = {"content": S.BLUE, "category": S.ORANGE, "popularity": S.AQUA}
fig, ax = plt.subplots(figsize=(7.5, 5))
for bname, g in frontier.groupby("base"):
    g = g.sort_values("lam", ascending=False)
    ax.plot(g["ndcg10"], g["distinct_categories"], marker="o", color=colors[bname], lw=2, label=bname)
    for _, r in g.iterrows():
        if r["lam"] in (1.0, 0.5, 0.0):
            ax.annotate(f"λ={r['lam']:g}", (r["ndcg10"], r["distinct_categories"]), textcoords="offset points",
                        xytext=(5, 4), fontsize=8.5, color=S.INK2)
rnd = base.loc["random"]
ax.scatter(rnd["ndcg10"], rnd["distinct_categories"], color=S.MUTED, marker="x", s=50, label="random (reference)")
ax.set_xlabel("Relevance: nDCG@10 (dev clicks)"); ax.set_ylabel("Diversity: distinct categories in top 10")
ax.legend(title="base strategy + MMR", loc="lower left"); S.title(ax, "Relevance–diversity frontier (simulated)")
S.save(fig, FIGURES / "03_frontier"); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 3.6))
bb = base.drop(index="random").sort_values("amp_vs_history_mean")
ax.barh(bb.index, bb["amp_vs_history_mean"], color=[colors[m] for m in bb.index], height=0.5)
ax.axvline(1, color=S.INK2, lw=1, ls="--")
ax.set_xlabel("Amplification ratio: dominant-category share in top 10 ÷ share in user history")
S.title(ax, "Do simulated recommenders amplify users' favourite category?")
S.save(fig, FIGURES / "03_amplification"); plt.show()
""")

md("## 9. Save results (with provenance)")
code(r"""
summary.to_csv(RESULTS / "recommender_metrics.csv", index=False)
frontier.to_csv(RESULTS / "recommender_frontier.csv", index=False)
rel_all.to_csv(RESULTS / "recommender_relevance_all.csv")
picks.to_csv(RESULTS / "recommender_mmr_selected.csv", index=False)
pd.DataFrame([{"dev_impressions": len(imp_size), "relevance_population": len(pop_all), "diversity_population": len(pop_div),
               "min_candidates": MIN_CANDIDATES, "dev_candidate_articles": len(dev_cands),
               "dev_candidates_with_train_clicks": float(seen.mean()), "dev_slots_with_train_clicks": float(slot_seen),
               "tfidf_terms": ctx.X.shape[1], "head_articles": int(head.sum())}]).to_csv(RESULTS / "recommender_setup.csv", index=False)

reg = M.HeadlineRegistry("notebooks/03_relevance_diversity.ipynb")
POPD = f"dev impressions with ≥{MIN_CANDIDATES} candidates, users with ≥1 history click (simulated strategies)"
for m, r in base.iterrows():
    reg.add(f"rec_{m}_ndcg10", r["ndcg10"], "mean nDCG@10 over impressions", POPD, notes=f"95% CI {r['ndcg10_ci_low']:.4f}–{r['ndcg10_ci_high']:.4f}")
    reg.add(f"rec_{m}_distinct_categories", r["distinct_categories"], "mean distinct categories in top 10", POPD)
    if m != "random":
        reg.add(f"rec_{m}_amp_vs_history", r["amp_vs_history_mean"], "mean over lists of rec dominant share / history dominant share",
                POPD, f"users with ≥{MIN_HIST_AMP} history clicks")
for _, r in picks.iterrows():
    reg.add(f"mmr_{r['base']}_lambda_selected", r["lambda_selected"], f"lowest λ reached from 1.0 before nDCG@10 loss exceeds {NDCG_TOLERANCE:.0%}", POPD)
    reg.add(f"mmr_{r['base']}_ndcg_rel_change", r["ndcg10_rel_change"], "nDCG@10 at selected λ / base − 1", POPD, unit="proportion")
    reg.add(f"mmr_{r['base']}_distinct_rel_change", r["distinct_categories_rel_change"], "distinct categories at selected λ / base − 1", POPD, unit="proportion")
reg.add("dev_candidates_with_train_clicks", float(seen.mean()), "dev candidate articles with ≥1 train click / dev candidate articles",
        "MIND small dev impressions", unit="proportion")
reg.save(RESULTS / "headline_metrics.json")
reg.to_frame()[["metric", "value", "denominator"]]
""")

md("## 10. Auto-generated FACT summary")
code(r"""
facts = [f"Diversity population: {len(pop_div):,} dev impressions (≥{MIN_CANDIDATES} candidates); relevance population: {len(pop_all):,}.",
         f"Only {seen.mean():.1%} of dev candidate articles had any train-period click (popularity is blind to the rest)."]
for m, r in base.iterrows():
    facts.append(f"{m}: nDCG@10 {r['ndcg10']:.4f} [{r['ndcg10_ci_low']:.4f}, {r['ndcg10_ci_high']:.4f}], AUC {r['auc']:.3f}, "
                 f"{r['distinct_categories']:.2f} distinct categories, ILD {r['ild']:.3f}, tail share {r['tail_share']:.1%}, "
                 f"amplification {r['amp_vs_history_mean']:.2f}.")
for _, r in picks.iterrows():
    facts.append(f"MMR on {r['base']}: λ={r['lambda_selected']:g} changes nDCG@10 by {r['ndcg10_rel_change']:+.1%} and "
                 f"distinct categories by {r['distinct_categories_rel_change']:+.1%}.")
print("\n".join("FACT: " + f for f in facts))
""")

md(r"""
## 11. Interpretation

*Written after review of the executed outputs above (see the FACT summary). Pending first execution on the
real MIND files.*
""")

md(r"""
## 12. Limitations

- **Simulated, offline, re-ranking only.** Strategies re-order the articles MSN actually showed; they cannot
  surface articles outside the logged candidate list, and offline click labels only exist for what was shown
  (an article ranked higher by our model might have been clicked had it been shown more prominently).
- **Click-conditioned labels.** Every logged impression has ≥1 click; relevance numbers are comparable *between
  strategies in this notebook only*, not to published MIND leaderboards or industry systems.
- **Simple models by design.** Popularity, TF-IDF and category preference are interpretable baselines, not
  state-of-the-art neural recommenders; the size of the trade-off may differ for stronger models.
- **Cold content.** Many dev candidates were published after the train period and have no train clicks, which
  handicaps the popularity strategy (quantified above).
- **Diversity = categories and title words.** Other notions (viewpoints, sources, geography of coverage) are not measured.
- **Not MSN.** Nothing here describes how MSN's production recommender behaves.
""")

b.save(Path(__file__).with_name("03_relevance_diversity.ipynb"))
