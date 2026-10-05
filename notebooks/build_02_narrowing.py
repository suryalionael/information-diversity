"""Generates notebooks/02_narrowing.ipynb (unexecuted). Run: python notebooks/build_02_narrowing.py"""
from pathlib import Path

from _nbbuild import SETUP, Builder

b = Builder()
md, code = b.md, b.code

md(r"""
# 02 — The Narrowing: does past engagement go with narrower later exposure?

## 1. Purpose

Research Question 2 (**Narrowing**): *Is greater historical engagement — or more concentrated historical
interest — associated with narrower subsequent information exposure?*

**Design (fixed by Notebook 01 validation).** A user's `history` field never changes across their logged
impressions (0 of 47,827 multi-impression users), and it is identical in train and dev. It records clicks
made **before** the logged period (9–15 Nov 2019). That gives a defensible ordering:

```
historical clicks (pre-period)  →  user profile  →  logged exposure (9–15 Nov)  →  exposure diversity
```

This is **associative**, not causal: history has no timestamps, and logged exposure comes from MSN's whole
ecosystem (editors, layout, recommendation, navigation), not only from personalization.

**Unit of analysis:** user (train ∪ dev). **Exposure:** all articles shown to the user in all their logged
impressions. **Categories:** the 18 MIND categories (4 tiny ones are kept in counts but are negligible).

### Measures

| Measure | Definition | Why |
|---|---|---|
| **Simpson concentration** *(primary)* | Σ nᵢ(nᵢ−1) / N(N−1): chance two random items share a category | Unbiased for any N ≥ 2, so short and long histories are comparable |
| Effective categories | 1 / Simpson | Plain-English reading: "equivalent number of equally-used categories" |
| Normalized entropy | H / log(18) | Classic diversity measure (secondary: biased low for small N) |
| Dominant share | share of the largest category | Simple, interpretable |
| **Alignment** | share of a user's exposure that falls in *their own* historically dominant category | Directly tests personalization of exposure |

Alignment is compared with a **permutation null**: dominant categories are shuffled among users in the same
stratum (split membership × activity level × history-concentration group), which keeps the period and
activity mix fixed and asks only "does *your* favourite category show up more for *you*?".
""")

md("## 2. Setup")
code(SETUP)

md("## 3. Data loading")
code(r"""
SPLITS = ["train", "dev"]
catalogue, _ = D.load_catalogue(SPLITS)
beh = pd.concat([D.load_behaviors(s) for s in SPLITS], ignore_index=True)
exp = D.load_exposures(SPLITS).merge(catalogue[["news_id", "category"]], on="news_id", how="left")
exp["clicked"] = exp["clicked"].astype("int64")
print(f"behaviors rows: {len(beh):,}   exposure records: {len(exp):,}   catalogue: {len(catalogue):,}")
""")

md("## 4. Data validation")
code(r"""
stab = D.history_stability(beh)
print("history stability (train+dev):", stab)
assert stab["users_with_varying_history"] == 0, "history varies within user: temporal design must change"
n_unjoined = int(exp["category"].isna().sum())
print("exposure records without category:", n_unjoined)
assert n_unjoined == 0

CATS = catalogue["category"].value_counts().index.tolist()          # fixed column order
K_NORM = len(CATS)
cat_size = catalogue["category"].value_counts()
MAIN_CATS = cat_size[cat_size >= 50].index.tolist()
print(f"K = {K_NORM} categories; {len(MAIN_CATS)} with ≥50 catalogue articles")
""")

md(r"""
## 5. Analysis A — Historical click profiles
""")
code(r"""
hist = D.explode_history(beh, dedupe_per_user=True).merge(catalogue[["news_id", "category"]], on="news_id", how="left")
assert hist["category"].notna().all()
all_users = pd.Index(beh["user_id"].unique(), name="user_id")
hcounts = V.count_matrix(hist, "user_id", "category", CATS).reindex(all_users, fill_value=0)
hp = V.profile_metrics(hcounts, K_NORM, "hist_")
print(f"users: {len(all_users):,}")
print(hp["hist_n"].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(1).to_string())
for t in (0, 1, 2, 5, 10, 20):
    print(f"users with < {t:>2} history clicks: {(hp['hist_n'] < t).mean():.1%}" if t else
          f"users with no history: {(hp['hist_n'] == 0).mean():.1%}")
""")
md(r"""
**Minimum history threshold.** The primary analysis uses users with **≥ 5 historical clicks** (`MIN_HIST = 5`).
Simpson concentration is defined from 2 clicks, but with fewer than 5 the dominant share can only take a
handful of discrete values (1, ¾, ⅔, ½ …) and a single click moves the profile drastically. The threshold is
fixed *a priori*; the robustness section re-runs the core tests at 2, 10 and 20.
""")
code(r"""
MIN_HIST = 5
SENS_THRESHOLDS = [2, 5, 10, 20]
print(f"eligible users (≥{MIN_HIST} history clicks): {(hp['hist_n'] >= MIN_HIST).sum():,} "
      f"of {len(hp):,} ({(hp['hist_n'] >= MIN_HIST).mean():.1%})")
hp.loc[hp["hist_n"] >= MIN_HIST, ["hist_n", "hist_n_categories", "hist_entropy_norm", "hist_simpson",
                                  "hist_eff_categories", "hist_dominant_share"]].describe().round(3)
""")
code(r"""
dom_tbl = (hp.loc[hp["hist_n"] >= MIN_HIST, "hist_dominant_category"].value_counts(normalize=True)
           .rename("share_of_eligible_users").to_frame())
dom_tbl
""")

md("## 6. Analysis B — Logged exposure profiles")
code(r"""
exp["day"] = exp["time"].dt.normalize()
ecounts = V.count_matrix(exp, "user_id", "category", CATS).reindex(all_users, fill_value=0)
ep = V.profile_metrics(ecounts, K_NORM, "exp_")
act = exp.groupby("user_id").agg(n_impressions=("imp_key", "nunique"), n_exposures=("news_id", "size"),
                                 n_unique_articles=("news_id", "nunique"), n_logged_clicks=("clicked", "sum"),
                                 n_days=("day", "nunique"))
membership = beh.groupby("user_id")["split"].agg(lambda s: "+".join(sorted(set(s)))).rename("splits")
ccounts = V.count_matrix(exp[exp["clicked"] == 1], "user_id", "category", CATS).reindex(all_users, fill_value=0)
cp = V.profile_metrics(ccounts, K_NORM, "click_")

# Per-impression (session-level) concentration: independent of how many impressions a user has
icounts = V.count_matrix(exp, "imp_key", "category", CATS)
ip = V.profile_metrics(icounts, K_NORM)
imp_user = exp.drop_duplicates("imp_key").set_index("imp_key")["user_id"]
sess = ip.join(imp_user).groupby("user_id").agg(sess_simpson=("simpson", "mean"),
                                                sess_n_categories=("n_categories", "mean"))

users = hp.join(ep).join(act).join(membership).join(cp).join(sess)
print(f"user table: {len(users):,} rows")
users[["n_impressions", "n_exposures", "n_unique_articles", "n_logged_clicks", "n_days",
       "exp_n_categories", "exp_simpson", "exp_eff_categories", "exp_entropy_norm", "exp_dominant_share"]].describe().round(3)
""")
code(r"""
print("split membership:\n" + users["splits"].value_counts().to_string())
print("\nmost common dominant *exposure* category:\n" +
      users["exp_dominant_category"].value_counts(normalize=True).head(6).round(3).to_string())
""")

md(r"""
## 7. Analysis C — History concentration → later exposure concentration (core test)
""")
code(r"""
elig = users[users["hist_n"] >= MIN_HIST].copy()
elig["hist_conc_simpson"] = elig["hist_simpson"]
elig["hist_conc_entropy"] = 1 - elig["hist_entropy_norm"]
elig["exp_conc_simpson"] = elig["exp_simpson"]
elig["exp_conc_entropy"] = 1 - elig["exp_entropy_norm"]

pairs = [("hist_conc_simpson", "exp_conc_simpson"), ("hist_dominant_share", "exp_conc_simpson"),
         ("hist_conc_entropy", "exp_conc_entropy"), ("hist_conc_simpson", "sess_simpson")]
corr_rows = []
for x, y in pairs:
    r = M.spearman_ci(elig[x], elig[y])
    corr_rows.append({"history_measure": x, "exposure_measure": y, **r})
corr = pd.DataFrame(corr_rows)
corr
""")
code(r"""
# Quintiles of historical concentration (equal-sized groups; non-linear patterns visible)
elig["hist_q"] = pd.qcut(elig["hist_simpson"].rank(method="first"), 5, labels=[f"Q{i}" for i in range(1, 6)])
qrows = []
for q, g in elig.groupby("hist_q", observed=True):
    est, lo, hi = M.bootstrap_ci(g["exp_simpson"].to_numpy())
    qrows.append({"hist_quintile": q, "n_users": len(g),
                  "hist_simpson_range": f"{g['hist_simpson'].min():.3f}–{g['hist_simpson'].max():.3f}",
                  "hist_eff_categories_median": g["hist_eff_categories"].median(),
                  "hist_dominant_share_median": g["hist_dominant_share"].median(),
                  "exp_simpson_mean": est, "exp_simpson_ci_low": lo, "exp_simpson_ci_high": hi,
                  "exp_eff_categories_median": g["exp_eff_categories"].median(),
                  "sess_simpson_mean": g["sess_simpson"].mean(),
                  "n_impressions_median": g["n_impressions"].median()})
quint = pd.DataFrame(qrows)
quint
""")

md(r"""
## 8. Analysis D — Is exposure tilted toward each user's own favourite category?
""")
code(r"""
def impression_group(n_imp: pd.Series) -> pd.Series:
    # activity strata for the null: quantile cut points on impressions, ties kept together
    edges = np.unique(np.quantile(n_imp, [0, .25, .5, .75, 1]))
    return pd.cut(n_imp, edges, include_lowest=True, duplicates="drop").astype(str)

def alignment(frame: pd.DataFrame, shares_counts: pd.DataFrame, n_perm: int = 200) -> dict:
    cats = list(shares_counts.columns)
    E = shares_counts.loc[frame.index].to_numpy(float)
    E = E / E.sum(axis=1, keepdims=True)
    dom = frame["hist_dominant_category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    qgroup = pd.qcut(frame["hist_simpson"].rank(method="first"), 5, labels=False).to_numpy()
    strata = (frame["splits"] + "|" + impression_group(frame["n_impressions"]) + "|" + qgroup.astype(str)).to_numpy()
    return V.permutation_alignment(E, dom, strata, groups=qgroup, n_perm=n_perm, seed=SEED)

al = alignment(elig, ecounts)
elig["exp_share_own_dom"] = al["observed"]
print(f"users: {al['n_users']:,}")
print(f"observed mean share of exposure in user's own historical top category: {al['observed_mean']:.4f}")
print(f"permutation null mean: {al['null_mean']:.4f}  (95% null interval {al['null_low']:.4f}–{al['null_high']:.4f})")
print(f"lift = {al['lift']:.3f}   one-sided p = {al['p_one_sided']:.3g}  ({al['n_perm']} permutations)")
align_q = pd.DataFrame({"hist_quintile": [f"Q{i}" for i in range(1, 6)], "n_users": al["group_n"],
                        "observed_share_own_dom": al["group_observed"], "null_mean": al["group_null_mean"],
                        "null_low": al["group_null_low"], "null_high": al["group_null_high"]})
align_q["lift"] = align_q["observed_share_own_dom"] / align_q["null_mean"]
align_q
""")
code(r"""
# Category view: users whose history is dominated by category c vs. other eligible users
crow = []
shares = ecounts.loc[elig.index].div(ecounts.loc[elig.index].sum(axis=1), axis=0)
cshares = ccounts.loc[elig.index]
for c in MAIN_CATS:
    fans = elig["hist_dominant_category"] == c
    if fans.sum() < 30:
        continue
    crow.append({"category": c, "n_users_dominant": int(fans.sum()), "n_users_other": int((~fans).sum()),
                 "exposure_share_dominant_users": shares.loc[fans, c].mean(),
                 "exposure_share_other_users": shares.loc[~fans, c].mean(),
                 "hist_share_dominant_users": (hcounts.loc[elig.index[fans], c] / elig.loc[fans, "hist_n"]).mean()})
by_cat = pd.DataFrame(crow)
by_cat["exposure_ratio"] = by_cat["exposure_share_dominant_users"] / by_cat["exposure_share_other_users"]
by_cat.sort_values("exposure_ratio", ascending=False)
""")

md(r"""
## 9. Analysis E — Behavioral activity groups

Groups are **quartiles of historical click count** (users with ≥1 history click; users with no history
form their own group). These are *behavioral activity groups*, not types of people. Exposure diversity is
shown both over all of a user's exposure (grows with the number of impressions) and per impression
(session-level, independent of volume).
""")
code(r"""
grp = users.copy()
has = grp["hist_n"] >= 1
grp["activity_group"] = "No history"
grp.loc[has, "activity_group"] = pd.qcut(grp.loc[has, "hist_n"], 4,
                                         labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"]).astype(str)
order = ["No history", "Q1 lowest", "Q2", "Q3", "Q4 highest"]
arows = []
for g in order:
    d = grp[grp["activity_group"] == g]
    est, lo, hi = M.bootstrap_ci(d["exp_simpson"].to_numpy())
    arows.append({"activity_group": g, "n_users": len(d),
                  "hist_clicks_range": f"{int(d['hist_n'].min())}–{int(d['hist_n'].max())}",
                  "hist_clicks_median": d["hist_n"].median(), "n_impressions_median": d["n_impressions"].median(),
                  "hist_eff_categories_median": d["hist_eff_categories"].median(),
                  "exp_simpson_mean": est, "exp_simpson_ci_low": lo, "exp_simpson_ci_high": hi,
                  "exp_eff_categories_median": d["exp_eff_categories"].median(),
                  "exp_n_categories_median": d["exp_n_categories"].median(),
                  "sess_simpson_mean": d["sess_simpson"].mean(),
                  "click_simpson_mean": d["click_simpson"].mean()})
activity = pd.DataFrame(arows)
activity
""")
code(r"""
act_corr = pd.DataFrame([
    {"x": "hist_n", "y": y, **M.spearman_ci(users.loc[users["hist_n"] >= 1, "hist_n"], users.loc[users["hist_n"] >= 1, y])}
    for y in ["exp_simpson", "sess_simpson", "exp_n_categories"]])
act_corr
""")

md(r"""
## 10. Analysis F — Click diversity vs. exposure diversity
Two comparisons:
1. **Same period:** what users clicked in the logged impressions vs. what they were shown (users with ≥5 logged clicks).
2. **History vs. later exposure:** pre-period click diversity vs. logged exposure diversity (eligible users).
""")
code(r"""
from scipy.stats import wilcoxon
same = users[users["n_logged_clicks"] >= 5]
d = same["click_simpson"] - same["exp_simpson"]
w = wilcoxon(same["click_simpson"], same["exp_simpson"])
cve = {"population": "users with ≥5 logged clicks", "n_users": len(same),
       "click_eff_categories_median": same["click_eff_categories"].median(),
       "exp_eff_categories_median": same["exp_eff_categories"].median(),
       "share_clicks_more_concentrated": float((d > 0).mean()), "median_diff_simpson": float(d.median()),
       "wilcoxon_p": float(w.pvalue)}
d2 = elig["hist_simpson"] - elig["exp_simpson"]
w2 = wilcoxon(elig["hist_simpson"], elig["exp_simpson"])
cve2 = {"population": f"users with ≥{MIN_HIST} history clicks", "n_users": len(elig),
        "click_eff_categories_median": elig["hist_eff_categories"].median(),
        "exp_eff_categories_median": elig["exp_eff_categories"].median(),
        "share_clicks_more_concentrated": float((d2 > 0).mean()), "median_diff_simpson": float(d2.median()),
        "wilcoxon_p": float(w2.pvalue)}
click_vs_exp = pd.DataFrame([cve, cve2])
click_vs_exp
""")
code(r"""
# Four quadrants (split at medians) and a 2-D binned histogram for the final chart (aggregate only)
xm, ym = elig["hist_eff_categories"].median(), elig["exp_eff_categories"].median()
quad = pd.crosstab(np.where(elig["hist_eff_categories"] <= xm, "narrow history", "broad history"),
                   np.where(elig["exp_eff_categories"] <= ym, "narrow exposure", "broad exposure"), normalize="all")
print(f"medians: history eff. categories {xm:.2f}, exposure eff. categories {ym:.2f}")
display(quad.round(3))
XMAX = float(np.nanquantile(elig["hist_eff_categories"], .99)); YMAX = float(np.nanquantile(elig["exp_eff_categories"], .99))
H, xe, ye = np.histogram2d(elig["hist_eff_categories"].clip(upper=XMAX), elig["exp_eff_categories"].clip(upper=YMAX),
                           bins=[30, 30], range=[[1, XMAX], [1, YMAX]])
hexbin = pd.DataFrame([{"x_low": xe[i], "x_high": xe[i + 1], "y_low": ye[j], "y_high": ye[j + 1], "n_users": int(H[i, j])}
                       for i in range(H.shape[0]) for j in range(H.shape[1]) if H[i, j] > 0])
print(f"2-D histogram bins with users: {len(hexbin)} (values above the 99th percentile clipped to the edge)")
""")

md(r"""
## 11. Robustness

Each row re-runs the two core tests — (1) Spearman ρ between historical and exposure Simpson concentration,
(2) alignment lift vs. the permutation null — under one alternative specification.
""")
code(r"""
def core_tests(frame, counts, label, n_perm=100):
    frame = frame.copy()
    tot = counts.loc[frame.index].sum(axis=1)
    frame = frame[tot > 0]
    prof = V.profile_metrics(counts.loc[frame.index], K_NORM)
    r = M.spearman_ci(frame["hist_simpson"], prof["simpson"], n_boot=100)
    a = alignment(frame, counts, n_perm=n_perm)
    return {"specification": label, "n_users": len(frame), "rho": r["rho"], "rho_ci_low": r["ci_low"],
            "rho_ci_high": r["ci_high"], "observed_share_own_dom": a["observed_mean"], "null_mean": a["null_mean"],
            "lift": a["lift"], "p_one_sided": a["p_one_sided"]}

rob = [core_tests(elig, ecounts, f"primary: ≥{MIN_HIST} history clicks, train+dev exposure")]
for t in SENS_THRESHOLDS:
    if t != MIN_HIST:
        rob.append(core_tests(users[users["hist_n"] >= t], ecounts, f"≥{t} history clicks"))
for s in SPLITS:
    sc = V.count_matrix(exp[exp["split"] == s], "user_id", "category", CATS)
    fr = elig[elig.index.isin(sc.index)]
    rob.append(core_tests(fr, sc.reindex(fr.index, fill_value=0), f"exposure from {s} only"))
for g, fr in elig.groupby(impression_group(elig["n_impressions"])):
    if len(fr) >= 200:
        rob.append(core_tests(fr, ecounts, f"within impressions group {g}"))
robust = pd.DataFrame(rob)
alt = M.spearman_ci(elig["hist_dominant_share"], elig["exp_dominant_share"], n_boot=100)
robust = pd.concat([robust, pd.DataFrame([{"specification": "measure: dominant share (history) vs dominant share (exposure)",
                                           "n_users": alt["n"], "rho": alt["rho"], "rho_ci_low": alt["ci_low"],
                                           "rho_ci_high": alt["ci_high"]}])], ignore_index=True)
robust
""")

md("## 12. Visualization")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
x = np.arange(len(align_q))
ax.fill_between(x, align_q["null_low"] * 100, align_q["null_high"] * 100, color=S.GRID, label="If exposure ignored history (95% null)")
ax.plot(x, align_q["null_mean"] * 100, color=S.MUTED, lw=1.5, ls="--")
ax.plot(x, align_q["observed_share_own_dom"] * 100, color=S.ORANGE, lw=2, marker="o", label="Observed")
ax.set_xticks(x, ["Q1\nbroadest", "Q2", "Q3", "Q4", "Q5\nmost concentrated"])
ax.set_xlabel("Historical click concentration (quintiles)"); ax.set_ylabel("% of exposure in user's top history category")
ax.legend(loc="upper left"); S.title(ax, "Exposure vs. each user's favourite category")
ax = axes[1]
ax.errorbar(x, quint["exp_simpson_mean"], yerr=[quint["exp_simpson_mean"] - quint["exp_simpson_ci_low"],
            quint["exp_simpson_ci_high"] - quint["exp_simpson_mean"]], color=S.BLUE, marker="o", lw=2, capsize=3)
ax.set_xticks(x, ["Q1\nbroadest", "Q2", "Q3", "Q4", "Q5\nmost concentrated"])
ax.set_xlabel("Historical click concentration (quintiles)"); ax.set_ylabel("Exposure concentration (Simpson)")
S.title(ax, "Later exposure concentration")
fig.tight_layout(); S.save(fig, FIGURES / "02_history_vs_exposure"); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(6, 5))
hb = ax.hexbin(elig["hist_eff_categories"].clip(upper=XMAX), elig["exp_eff_categories"].clip(upper=YMAX),
               gridsize=35, cmap="Blues", mincnt=1, bins="log", linewidths=0)
lim = max(XMAX, YMAX)
ax.plot([1, lim], [1, lim], color=S.INK2, lw=1, ls="--"); ax.text(lim * .72, lim * .78, "equal diversity", color=S.INK2, rotation=0)
ax.set_xlabel("Historical click diversity (effective categories)"); ax.set_ylabel("Logged exposure diversity (effective categories)")
fig.colorbar(hb, ax=ax, label="users (log scale)"); S.title(ax, "Clicks vs. exposure diversity per user")
S.save(fig, FIGURES / "02_click_vs_exposure"); plt.show()
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 3.8))
x = np.arange(len(activity))
ax.errorbar(x, activity["exp_simpson_mean"], yerr=[activity["exp_simpson_mean"] - activity["exp_simpson_ci_low"],
            activity["exp_simpson_ci_high"] - activity["exp_simpson_mean"]], color=S.BLUE, marker="o", lw=2, capsize=3,
            label="all logged exposure")
ax.plot(x, activity["sess_simpson_mean"], color=S.ORANGE, marker="s", lw=2, label="per impression (session)")
ax.set_xticks(x, activity["activity_group"]); ax.set_ylabel("Exposure concentration (Simpson)")
ax.set_xlabel("Behavioral activity group (quartiles of historical clicks)"); ax.legend()
S.title(ax, "Exposure concentration by activity group")
S.save(fig, FIGURES / "02_activity_groups"); plt.show()
""")

md("## 13. Save results (with provenance)")
code(r"""
reg = M.HeadlineRegistry("notebooks/02_narrowing.ipynb")
POP = f"users with ≥{MIN_HIST} pre-period history clicks, exposure = all logged impressions (train+dev)"
corr.to_csv(RESULTS / "narrowing_correlations.csv", index=False)
quint.to_csv(RESULTS / "narrowing_hist_quintiles.csv", index=False)
align_q.to_csv(RESULTS / "narrowing_alignment_by_quintile.csv", index=False)
by_cat.to_csv(RESULTS / "narrowing_alignment_by_category.csv", index=False)
activity.to_csv(RESULTS / "diversity_metrics.csv", index=False)
act_corr.to_csv(RESULTS / "narrowing_activity_correlations.csv", index=False)
click_vs_exp.to_csv(RESULTS / "narrowing_click_vs_exposure.csv", index=False)
quad.to_csv(RESULTS / "narrowing_quadrants.csv")
hexbin.to_csv(RESULTS / "narrowing_hist2d.csv", index=False)
robust.to_csv(RESULTS / "narrowing_robustness.csv", index=False)

prim = corr.iloc[0]
reg.add("narrowing_users_total", len(users), "users", "train ∪ dev behaviors.tsv")
reg.add("narrowing_users_eligible", len(elig), "users", POP, f"hist_n ≥ {MIN_HIST}")
reg.add("narrowing_rho_hist_vs_exposure_simpson", prim["rho"], "Spearman rank correlation", POP,
        notes=f"95% bootstrap CI {prim['ci_low']:.3f}–{prim['ci_high']:.3f}; p={prim['p']:.2g}")
reg.add("alignment_observed_share_own_dom", al["observed_mean"], "mean over users of (exposure in own top history category / all exposure)", POP, unit="proportion")
reg.add("alignment_null_share_own_dom", al["null_mean"], "same, with top categories shuffled within strata", POP,
        "200 permutations within split × impressions-group × history-quintile", unit="proportion")
reg.add("alignment_lift", al["lift"], "observed / null", POP, notes=f"one-sided permutation p={al['p_one_sided']:.3g}")
for _, r in align_q.iterrows():
    reg.add(f"alignment_lift_{r['hist_quintile']}", r["lift"], "observed / null", POP, f"history-concentration quintile {r['hist_quintile']}")
reg.add("exp_eff_categories_median", elig["exp_eff_categories"].median(), "1 / unbiased Simpson of exposure categories", POP)
reg.add("hist_eff_categories_median", elig["hist_eff_categories"].median(), "1 / unbiased Simpson of history categories", POP)
reg.add("share_users_history_more_concentrated_than_exposure", cve2["share_clicks_more_concentrated"],
        "users with hist_simpson > exp_simpson / eligible users", POP, unit="proportion")
reg.add("share_users_logged_clicks_more_concentrated_than_exposure", cve["share_clicks_more_concentrated"],
        "users with click_simpson > exp_simpson / users with ≥5 logged clicks", "users with ≥5 logged clicks", unit="proportion")
reg.add("rho_hist_clicks_vs_exposure_simpson", act_corr.iloc[0]["rho"], "Spearman rank correlation",
        "users with ≥1 history click", notes="history length vs exposure concentration")
reg.save(RESULTS / "headline_metrics.json")
reg.to_frame()[["metric", "value", "denominator"]]
""")

md("## 14. Auto-generated FACT summary")
code(r"""
facts = [
    f"{len(elig):,} of {len(users):,} users have ≥{MIN_HIST} pre-period history clicks.",
    f"Spearman ρ(historical Simpson, exposure Simpson) = {prim['rho']:.3f} (95% CI {prim['ci_low']:.3f}–{prim['ci_high']:.3f}).",
    f"Users' own top history category takes {al['observed_mean']:.1%} of their logged exposure vs {al['null_mean']:.1%} "
    f"under the permutation null (lift {al['lift']:.2f}, p={al['p_one_sided']:.3g}).",
    "Lift by history-concentration quintile: " + ", ".join(f"{q} {l:.2f}" for q, l in zip(align_q['hist_quintile'], align_q['lift'])) + ".",
    f"Median effective categories: history {elig['hist_eff_categories'].median():.2f}, exposure {elig['exp_eff_categories'].median():.2f}.",
    f"{cve2['share_clicks_more_concentrated']:.1%} of eligible users have history more concentrated than their exposure; "
    f"{cve['share_clicks_more_concentrated']:.1%} of users with ≥5 logged clicks clicked more narrowly than they were shown.",
    "Activity groups (exposure Simpson mean): " + ", ".join(f"{g} {v:.3f}" for g, v in zip(activity['activity_group'], activity['exp_simpson_mean'])) + ".",
]
print("\n".join("FACT: " + f for f in facts))
""")

md(r"""
## 15. Interpretation

*Written after review of the executed outputs above (see the FACT summary). Pending first execution on the
real MIND files.*
""")

md(r"""
## 16. Limitations

- **Associative design.** History precedes the logged exposure, but has no timestamps; the analysis cannot show
  that past clicks *caused* later exposure. Users' interests may simply be stable, and MSN may show popular
  content regardless.
- **Exposure ≠ algorithm.** Logged impressions reflect MSN's whole ecosystem (editorial modules, layout, recommendation,
  navigation). The alignment test shows whether exposure is *tilted toward* past interests, not which component does it.
- **Click-conditioned logs.** Every logged impression contains ≥1 click, so logged sessions are engaged sessions.
- **Short window.** One week of exposure; per-user exposure counts are small for light users (handled with
  an unbiased concentration measure, a minimum-history threshold, session-level measures and robustness checks).
- **Category granularity.** 18 broad categories; narrowing *within* a category (e.g. one team, one politician) is
  invisible here.
- **No demographics.** Groups are behavioral activity groups only.
""")

b.save(Path(__file__).with_name("02_narrowing.ipynb"))
