"""Generates notebooks/01_big_picture.ipynb (unexecuted).

Kept as a script so notebook logic is diff-able in review. Re-run after edits:
    python notebooks/build_01_big_picture.py
"""
from pathlib import Path

import nbformat as nbf

cells: list = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# =========================================================================== #
md(r"""
# 01 — The Big Picture: catalogue, exposure, and clicks

## 1. Purpose

Research Question 1 (**Amplification**): *Which news categories receive disproportionately high
exposure compared with their presence in the available content catalogue?*

This notebook works only with the **observed system** — the MIND small impression logs. It:

1. loads and validates `news.tsv` and `behaviors.tsv` (train + dev);
2. builds one **exposure record** per article shown in an impression;
3. compares **catalogue share → exposure share → click share** by category (Analyses A–D);
4. measures **article-level exposure concentration** (Lorenz, Gini, top-X%) and the long tail (E–F).

**Definitions** (see `CLAUDE.md`)

| Term | Definition | Denominator |
|---|---|---|
| Catalogue share | articles in category | unique articles in train ∪ dev `news.tsv` |
| Exposed-pool share | articles in category shown ≥1 time | unique articles shown ≥1 time |
| Exposure share | impressions in category | all impressions (train + dev) |
| Click share | clicks in category | all clicks |
| CTR | clicks in category | impressions in category |
| Exposure amplification | exposure share / catalogue share | — (1 = proportional) |

An *impression* is a logged exposure proxy, not proof of attention. A *click* is an engagement signal.
""")

md("## 2. Setup")
code(r"""
import sys, json
from pathlib import Path

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src import data as D
from src import metrics as M

SOURCE = "notebooks/01_big_picture.ipynb"
OUT = D.output_dir()
RESULTS, FIGURES = OUT / "results", OUT / "figures"
RESULTS.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
reg = M.HeadlineRegistry(SOURCE)

pd.set_option("display.width", 160, "display.max_columns", 30, "display.float_format", "{:,.4f}".format)

# Visual system (validated categorical slots 1-3; text in ink, never series colour)
C_CAT, C_EXP, C_CLK = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False,
})
print("data dir:", D.data_dir()); print("output dir:", OUT)
""")

# =========================================================================== #
md("## 3. Data loading")
code(r"""
files = D.check_files()
files
""")
code(r"""
SPLITS = ["train", "dev"]
news = {s: D.load_news(s) for s in SPLITS}
beh = {s: D.load_behaviors(s) for s in SPLITS}
for s in SPLITS:
    print(f"{s:5s}  news.tsv rows = {len(news[s]):>9,}   behaviors.tsv rows = {len(beh[s]):>9,}")
""")
code(r"""
display(news["train"].head(3))
display(beh["train"].head(3))
""")

# =========================================================================== #
md(r"""
## 4. Data validation

Checks: schema, row counts, missing values, duplicate IDs, impression structure, label validity,
joinability of shown and history articles to `news.tsv`, user overlap across splits, and whether a
user's `history` changes over time (relevant to Notebook 02's temporal design).
""")
code(r"""
news_val = pd.DataFrame({s: D.validate_news(news[s]) for s in SPLITS})
news_val
""")
code(r"""
catalogue, conflicts = D.load_catalogue(SPLITS)
print(f"Catalogue (train ∪ dev news.tsv): {len(catalogue):,} unique articles")
print(catalogue["in_splits"].value_counts().to_string())
print(f"Articles with conflicting category/subcategory/title across splits: {conflicts['news_id'].nunique()}")
""")
code(r"""
exp_by_split = {s: D.explode_impressions(beh[s]) for s in SPLITS}
beh_val = pd.DataFrame({s: D.validate_behaviors(beh[s], exp_by_split[s]) for s in SPLITS})
beh_val
""")
code(r"""
# Join coverage: shown articles and history articles vs. (a) same-split news.tsv, (b) the union catalogue
cov_rows = []
for s in SPLITS:
    hist = D.explode_history(beh[s], dedupe_per_user=False)
    for label, ids in [("impression articles", exp_by_split[s]["news_id"]), ("history articles", hist["news_id"])]:
        for ref_name, ref in [("same-split news.tsv", news[s]["news_id"]), ("union catalogue", catalogue["news_id"])]:
            cov_rows.append({"split": s, "ids": label, "reference": ref_name, **D.join_coverage(ids, ref)})
coverage = pd.DataFrame(cov_rows)
coverage
""")
code(r"""
users = {s: set(beh[s]["user_id"]) for s in SPLITS}
overlap = len(users["train"] & users["dev"])
all_users = users["train"] | users["dev"]
print(f"unique users train={len(users['train']):,}  dev={len(users['dev']):,}  "
      f"both={overlap:,}  union={len(all_users):,}")

beh_all = pd.concat([beh[s] for s in SPLITS], ignore_index=True)
hist_stab = {"within train": D.history_stability(beh["train"]),
             "within dev": D.history_stability(beh["dev"]),
             "across train+dev": D.history_stability(beh_all)}
pd.DataFrame(hist_stab)
""")
code(r"""
empty_hist = {s: int(beh[s]["history"].isna().sum()) for s in SPLITS}
print("impressions with empty history:", empty_hist)
for s in SPLITS:
    days = beh[s]["time"].dt.date.value_counts().sort_index()
    print(f"\n{s}: impressions per calendar day\n{days.to_string()}")
""")
code(r"""
# Collect anomalies programmatically so the list reflects what was actually observed.
anomalies = []
for s in SPLITS:
    nv, bv = news_val[s], beh_val[s]
    if nv["duplicate_news_ids"]: anomalies.append(f"{s}: {nv['duplicate_news_ids']} duplicate news_ids in news.tsv")
    if nv["bad_id_format"]: anomalies.append(f"{s}: {nv['bad_id_format']} news_ids not matching N<digits>")
    for col, n in nv["missing_by_column"].items():
        if n: anomalies.append(f"{s}: news.tsv column '{col}' missing in {n:,} rows ({n/nv['rows']:.1%})")
    for col, n in bv["missing_by_column"].items():
        if n: anomalies.append(f"{s}: behaviors.tsv column '{col}' missing in {n:,} rows ({n/bv['rows']:.1%})")
    for k in ["duplicate_impression_ids", "unparseable_labels", "labels_not_0_or_1",
              "impressions_with_no_articles", "duplicate_article_within_impression"]:
        if bv[k]: anomalies.append(f"{s}: {k} = {bv[k]:,}")
    if bv["impressions_with_zero_clicks"]:
        anomalies.append(f"{s}: {bv['impressions_with_zero_clicks']:,} impressions have zero clicks")
for _, r in coverage.iterrows():
    if r["records_unmatched"]:
        anomalies.append(f"{r['split']}: {r['ids']} not in {r['reference']}: "
                         f"{r['records_unmatched']:,} records / {r['unique_ids_unmatched']:,} unique ids")
if conflicts["news_id"].nunique():
    anomalies.append(f"{conflicts['news_id'].nunique()} articles have conflicting metadata across splits")
for k, v in hist_stab.items():
    if v["users_with_varying_history"]:
        anomalies.append(f"history varies across impressions for {v['users_with_varying_history']:,} users ({k})")
print("\n".join(f"- {a}" for a in anomalies) or "No anomalies detected.")
""")

# =========================================================================== #
md(r"""
## 5. Build the analysis table

Exposure records from train and dev are combined: they cover different days of the same collection
period, and this notebook describes the observed system as a whole. Shown articles are joined to the
union catalogue for category/subcategory. Rows that fail the join (if any — see validation) are
excluded from category analyses and counted.
""")
code(r"""
exp = pd.concat([exp_by_split[s] for s in SPLITS], ignore_index=True)
exp = exp.merge(catalogue[["news_id", "category", "subcategory"]], on="news_id", how="left")
n_unjoined = int(exp["category"].isna().sum())
print(f"exposure records: {len(exp):,}   unjoined to catalogue: {n_unjoined:,}")
exp = exp[exp["category"].notna()].copy()
exp["clicked"] = exp["clicked"].astype("int64")
K_CATEGORIES = catalogue["category"].nunique()
print("categories in catalogue:", K_CATEGORIES)
""")

# =========================================================================== #
md("## 6. Analysis A — Catalogue composition (baseline)")
code(r"""
cat_comp = (catalogue.groupby("category").size().rename("n_articles").to_frame()
            .assign(catalogue_share=lambda d: d["n_articles"] / d["n_articles"].sum())
            .sort_values("n_articles", ascending=False))
print(f"unique articles: {len(catalogue):,}   categories: {catalogue['category'].nunique()}   "
      f"subcategories: {catalogue['subcategory'].nunique()}")
cat_comp
""")
code(r"""
sub_comp = (catalogue.groupby(["category", "subcategory"]).size().rename("n_articles").reset_index()
            .assign(catalogue_share=lambda d: d["n_articles"] / d["n_articles"].sum())
            .sort_values("n_articles", ascending=False))
sub_comp.head(25)
""")
code(r"""
# How much of the catalogue was ever shown? (articles that only appear in click histories are older content)
shown_ids = set(exp["news_id"].unique())
hist_all = D.explode_history(beh_all, dedupe_per_user=False)
hist_ids = set(hist_all["news_id"].unique())
cat_ids = set(catalogue["news_id"])
membership = pd.Series({
    "catalogue articles": len(cat_ids),
    "shown in ≥1 impression": len(cat_ids & shown_ids),
    "only in click histories (never shown)": len((cat_ids & hist_ids) - shown_ids),
    "neither shown nor in histories": len(cat_ids - shown_ids - hist_ids),
})
membership.to_frame("n_articles").assign(share=lambda d: d["n_articles"] / len(cat_ids))
""")

# =========================================================================== #
md("## 7. Analyses B–D — Exposure, clicks, CTR and exposure amplification by category")
code(r"""
TOTAL_IMPRESSIONS = len(exp)
TOTAL_CLICKS = int(exp["clicked"].sum())
UNIQUE_EXPOSED = exp["news_id"].nunique()
print(f"total impressions (exposure records): {TOTAL_IMPRESSIONS:,}")
print(f"unique exposed articles:              {UNIQUE_EXPOSED:,}")
print(f"total clicks:                         {TOTAL_CLICKS:,}")
print(f"overall CTR (clicks / impressions):   {TOTAL_CLICKS / TOTAL_IMPRESSIONS:.4%}")
""")
code(r"""
cat_tbl = M.category_table(catalogue, exp, "category")
cols = ["n_articles", "catalogue_share", "n_exposed_articles", "pool_share", "impressions", "exposure_share",
        "clicks", "click_share", "ctr", "ctr_ci_low", "ctr_ci_high", "amplification", "amplification_vs_pool",
        "click_amplification"]
cat_tbl[cols]
""")
code(r"""
# Does exposure track engagement at category level? (categories with ≥50 catalogue articles)
from scipy.stats import spearmanr
big = cat_tbl[cat_tbl["n_articles"] >= 50]
r_ctr = spearmanr(big["amplification"], big["ctr"])
r_clk = spearmanr(big["exposure_share"], big["click_share"])
print(f"n categories = {len(big)}")
print(f"Spearman ρ(amplification, CTR)          = {r_ctr.statistic:.3f} (p = {r_ctr.pvalue:.2g})")
print(f"Spearman ρ(exposure share, click share) = {r_clk.statistic:.3f} (p = {r_clk.pvalue:.2g})")
""")
code(r"""
# Robustness: are category exposure shares stable across the two splits (different days)?
split_shares = (exp.groupby(["split", "category"]).size().unstack(0)
                .pipe(lambda d: d / d.sum()).rename(columns=lambda c: f"exposure_share_{c}"))
split_shares["abs_diff"] = (split_shares.iloc[:, 0] - split_shares.iloc[:, 1]).abs()
split_shares.sort_values("exposure_share_train", ascending=False)
""")
code(r"""
# Subcategory amplification — only subcategories with enough support to be meaningful.
MIN_ARTICLES = 50  # stated a-priori support threshold; tiny catalogue counts make ratios unstable
sub_tbl = M.category_table(catalogue, exp, "subcategory")
parent = catalogue.drop_duplicates("subcategory").set_index("subcategory")["category"]
sub_tbl.insert(0, "category", parent.reindex(sub_tbl.index))
sub_sup = sub_tbl[sub_tbl["n_articles"] >= MIN_ARTICLES]
print(f"subcategories: {len(sub_tbl)}  with ≥{MIN_ARTICLES} catalogue articles: {len(sub_sup)} "
      f"(covering {sub_sup['impressions'].sum()/TOTAL_IMPRESSIONS:.1%} of impressions)")
print("\nHighest exposure amplification:")
display(sub_sup.sort_values("amplification", ascending=False)[["category", "n_articles", "catalogue_share",
        "exposure_share", "amplification", "amplification_vs_pool", "ctr"]].head(10))
print("Lowest exposure amplification:")
display(sub_sup.sort_values("amplification")[["category", "n_articles", "catalogue_share",
        "exposure_share", "amplification", "amplification_vs_pool", "ctr"]].head(10))
""")

# =========================================================================== #
md(r"""
## 8. Analysis E — Article-level exposure concentration

Primary population: **articles shown at least once** (the exposed pool). Including never-shown catalogue
articles would mostly add older articles that only appear in click histories and were not candidates during
the logged period, inflating concentration; that variant is reported as a sensitivity check only.

Because news articles have short lifespans and the logs cover only a few days, concentration partly
reflects *timing* (when articles were published) as well as selection. We therefore also report it per day.
""")
code(r"""
art = (exp.groupby("news_id").agg(impressions=("clicked", "size"), clicks=("clicked", "sum"),
                                   category=("category", "first"))
       .sort_values("impressions", ascending=False))
imp = art["impressions"].to_numpy()
print(art["impressions"].describe(percentiles=[.25, .5, .75, .9, .99]).round(1).to_string())
""")
code(r"""
conc_rows = []
def add_conc(population, values, measure="impressions"):
    row = {"population": population, "measure": measure, "n_items": len(values),
           "total": float(np.sum(values)), "gini": M.gini(values)}
    for f in (0.01, 0.05, 0.10):
        row[f"top_{int(f*100)}pct_share"] = M.top_share(values, f)["share_of_total"]
    for f in (0.50, 0.90):
        row[f"bottom_{int(f*100)}pct_share"] = M.bottom_share(values, f)["share_of_total"]
    for s in (0.50, 0.80):
        row[f"frac_items_for_{int(s*100)}pct"] = M.items_for_share(values, s)["fraction_of_items"]
    conc_rows.append(row)

add_conc("exposed articles (train+dev)", imp)
zeros = np.zeros(len(catalogue) - len(art))
add_conc("full catalogue incl. never-shown (sensitivity)", np.concatenate([imp, zeros]))
add_conc("exposed articles (train+dev)", art["clicks"].to_numpy(), measure="clicks")
for s in SPLITS:
    add_conc(f"exposed articles ({s} only)", exp[exp["split"] == s].groupby("news_id").size().to_numpy())
exp["day"] = exp["time"].dt.date
for d, g in exp.groupby("day"):
    add_conc(f"exposed articles (day {d})", g.groupby("news_id").size().to_numpy())
conc = pd.DataFrame(conc_rows)
conc
""")
code(r"""
# Who is in the top 1%? Category mix of the most-exposed articles vs. all exposed articles.
k1 = M.top_share(imp, 0.01)["n_items_top"]
top1 = art.head(k1)
mix = pd.concat([top1["category"].value_counts(normalize=True).rename("share_of_top1pct_articles"),
                 art["category"].value_counts(normalize=True).rename("share_of_exposed_articles")], axis=1).fillna(0)
print(f"top 1% = {k1} articles; minimum impressions in top 1% = {top1['impressions'].min():,}")
mix.sort_values("share_of_top1pct_articles", ascending=False)
""")

# =========================================================================== #
md("## 9. Analysis F — Long-tail exposure")
code(r"""
long_tail = pd.Series({
    "exposed articles": len(art),
    "share of catalogue ever shown": len(art) / len(catalogue),
    "articles shown exactly once": int((imp == 1).sum()),
    "articles shown ≤10 times": int((imp <= 10).sum()),
    "share of impressions to bottom 50% of exposed articles": M.bottom_share(imp, .5)["share_of_total"],
    "share of impressions to bottom 90% of exposed articles": M.bottom_share(imp, .9)["share_of_total"],
    "fraction of exposed articles receiving 50% of impressions": M.items_for_share(imp, .5)["fraction_of_items"],
    "fraction of exposed articles receiving 80% of impressions": M.items_for_share(imp, .8)["fraction_of_items"],
})
long_tail.to_frame("value")
""")
code(r"""
# Is exposure concentration tracking engagement? Article CTR among the top 1% vs. the rest
# (restricted to articles with ≥100 impressions so CTR is not dominated by noise).
MIN_IMP = 100
supp = art[art["impressions"] >= MIN_IMP].copy()
supp["ctr"] = supp["clicks"] / supp["impressions"]
supp["group"] = np.where(supp.index.isin(top1.index), "top 1% by impressions", "rest (≥100 impressions)")
ctr_cmp = supp.groupby("group").agg(n_articles=("ctr", "size"), impressions=("impressions", "sum"),
                                    clicks=("clicks", "sum"), median_article_ctr=("ctr", "median"))
ctr_cmp["pooled_ctr"] = ctr_cmp["clicks"] / ctr_cmp["impressions"]
from scipy.stats import spearmanr
rho = spearmanr(supp["impressions"], supp["ctr"])
print(f"Spearman ρ(impressions, CTR) among articles with ≥{MIN_IMP} impressions: "
      f"{rho.statistic:.3f} (p={rho.pvalue:.2g}, n={len(supp):,})")
ctr_cmp
""")

# =========================================================================== #
md("## 10. Visualization")
code(r"""
order = cat_tbl.sort_values("exposure_share").index
y = np.arange(len(order)); h = 0.26
fig, ax = plt.subplots(figsize=(8, 0.42 * len(order) + 1.2))
for i, (col, c, lab) in enumerate([("catalogue_share", C_CAT, "Catalogue share (articles)"),
                                   ("exposure_share", C_EXP, "Exposure share (impressions)"),
                                   ("click_share", C_CLK, "Click share (clicks)")]):
    ax.barh(y + (1 - i) * h, cat_tbl.loc[order, col] * 100, height=h - 0.03, color=c, label=lab)
ax.set_yticks(y, order); ax.set_xlabel("% of total"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right")
ax.set_title("Catalogue vs. exposure vs. clicks by category — MIND small (train+dev)", loc="left", fontsize=11)
fig.savefig(FIGURES / "01_category_shares.png"); plt.show()
""")
code(r"""
amp = cat_tbl[cat_tbl["n_articles"] > 0].sort_values("amplification")
fig, ax = plt.subplots(figsize=(7, 0.36 * len(amp) + 1.2))
ax.hlines(amp.index, 1, amp["amplification"], color=GRID, lw=2)
ax.scatter(amp["amplification"], amp.index, color=C_EXP, s=40, zorder=3, label="vs. full catalogue")
ax.scatter(amp["amplification_vs_pool"], amp.index, facecolor="white", edgecolor=C_CAT, s=40, zorder=3,
           label="vs. exposed pool (robustness)")
ax.axvline(1, color=INK2, lw=1); ax.set_xscale("log")
ax.set_xlabel("Exposure amplification = exposure share / catalogue share  (log scale, 1 = proportional)")
ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
ax.set_title("Relative exposure by category", loc="left", fontsize=11)
fig.savefig(FIGURES / "01_amplification.png"); plt.show()
""")
code(r"""
x, yv = M.lorenz_curve(imp)
g = M.gini(imp)
fig, ax = plt.subplots(figsize=(5.5, 5.2))
ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--", label="Perfect equality")
ax.plot(x, yv, color=C_CAT, lw=2, label=f"Observed (Gini = {g:.2f})")
ax.set_xlabel("Cumulative share of exposed articles (fewest → most impressions)")
ax.set_ylabel("Cumulative share of impressions"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
t1 = M.top_share(imp, .01)["share_of_total"]
ax.annotate(f"Top 1% of articles:\n{t1:.1%} of impressions", xy=(0.99, 1 - t1), xytext=(0.45, 0.72),
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), fontsize=9, color=INK)
ax.legend(loc="upper left"); ax.set_title("Article-level exposure concentration", loc="left", fontsize=11)
fig.savefig(FIGURES / "01_lorenz.png"); plt.show()
""")
code(r"""
ctr_o = cat_tbl[cat_tbl["impressions"] > 0].sort_values("ctr")
fig, ax = plt.subplots(figsize=(7, 0.36 * len(ctr_o) + 1.2))
ax.errorbar(ctr_o["ctr"] * 100, ctr_o.index,
            xerr=[(ctr_o["ctr"] - ctr_o["ctr_ci_low"]) * 100, (ctr_o["ctr_ci_high"] - ctr_o["ctr"]) * 100],
            fmt="o", color=C_CLK, ecolor=INK2, elinewidth=1, ms=6)
ax.axvline(TOTAL_CLICKS / TOTAL_IMPRESSIONS * 100, color=INK2, lw=1, ls="--", label="overall CTR")
ax.set_xlabel("CTR = clicks / impressions in category (%), 95% Wilson CI"); ax.grid(axis="y", visible=False)
ax.legend(loc="lower right"); ax.set_title("Click-through rate by category", loc="left", fontsize=11)
fig.savefig(FIGURES / "01_ctr_by_category.png"); plt.show()
""")

# =========================================================================== #
md("## 11. Save results (with provenance)")
code(r"""
cat_tbl.to_csv(RESULTS / "category_metrics.csv")
sub_tbl.to_csv(RESULTS / "subcategory_metrics.csv")
conc.to_csv(RESULTS / "concentration_metrics.csv", index=False)
art.to_csv(RESULTS / "article_exposure.csv")
pd.DataFrame({"news": news_val.astype(str).to_dict(), "behaviors": beh_val.astype(str).to_dict()}).to_json(
    RESULTS / "validation_01.json", indent=2)
(RESULTS / "anomalies_01.txt").write_text("\n".join(anomalies) + "\n")

POP = "MIND small, train+dev behaviors.tsv, all impressions joined to union news.tsv"
reg.add("rows_news_train", len(news["train"]), "rows", "MIND small train news.tsv")
reg.add("rows_news_dev", len(news["dev"]), "rows", "MIND small dev news.tsv")
reg.add("rows_behaviors_train", len(beh["train"]), "rows (impressions logs)", "MIND small train behaviors.tsv")
reg.add("rows_behaviors_dev", len(beh["dev"]), "rows (impressions logs)", "MIND small dev behaviors.tsv")
reg.add("unique_users", len(all_users), "unique user_id", "train ∪ dev behaviors.tsv")
reg.add("catalogue_articles", len(catalogue), "unique news_id", "train ∪ dev news.tsv")
reg.add("total_impressions", TOTAL_IMPRESSIONS, "exposure records (article shown in an impression)", POP,
        f"excluded {n_unjoined} unjoined records")
reg.add("unique_exposed_articles", UNIQUE_EXPOSED, "unique news_id shown ≥1 time", POP)
reg.add("total_clicks", TOTAL_CLICKS, "exposure records with label 1", POP)
reg.add("overall_ctr", TOTAL_CLICKS / TOTAL_IMPRESSIONS, "clicks / impressions", POP, unit="proportion")
reg.add("share_of_catalogue_ever_shown", len(art) / len(catalogue), "exposed articles / catalogue articles", POP,
        unit="proportion")

top_amp = cat_tbl[cat_tbl["n_articles"] >= MIN_ARTICLES].sort_values("amplification", ascending=False)
if top_amp.empty:
    print(f"WARNING: no category has ≥{MIN_ARTICLES} catalogue articles; amplification headlines skipped")
for name, row in ([("highest", top_amp.iloc[0]), ("lowest", top_amp.iloc[-1])] if len(top_amp) else []):
    reg.add(f"category_amplification_{name}", f"{row.name}: {row['amplification']:.3f}",
            "exposure share / catalogue share", POP, f"categories with ≥{MIN_ARTICLES} catalogue articles",
            notes=f"exposure_share={row['exposure_share']:.4f}, catalogue_share={row['catalogue_share']:.4f}, "
                  f"amplification_vs_pool={row['amplification_vs_pool']:.3f}")

main = conc.iloc[0]
for k in ["gini", "top_1pct_share", "top_5pct_share", "top_10pct_share", "bottom_50pct_share",
          "bottom_90pct_share", "frac_items_for_50pct"]:
    reg.add(f"article_exposure_{k}", main[k], "impressions per article", f"{int(main['n_items']):,} exposed articles, {POP}",
            "articles with ≥1 impression", unit="proportion")
reg.add("article_exposure_gini_full_catalogue", conc.iloc[1]["gini"], "impressions per article",
        "full catalogue incl. never-shown articles", "sensitivity analysis", unit="proportion")
reg.save(RESULTS / "headline_metrics.json")
reg.to_frame()[["metric", "value", "denominator", "filters"]]
""")

# =========================================================================== #
md("## 12. Auto-generated FACT summary")
code(r"""
# Sentences generated directly from computed values — the basis for the written interpretation.
c = cat_tbl
facts = [f"{len(catalogue):,} catalogue articles across {K_CATEGORIES} categories; {UNIQUE_EXPOSED:,} "
         f"({UNIQUE_EXPOSED/len(catalogue):.1%}) were shown at least once.",
         f"{TOTAL_IMPRESSIONS:,} impressions and {TOTAL_CLICKS:,} clicks (CTR {TOTAL_CLICKS/TOTAL_IMPRESSIONS:.2%})."]
for cat_name, r in c.head(5).iterrows():
    facts.append(f"'{cat_name}': {r['catalogue_share']:.1%} of catalogue, {r['exposure_share']:.1%} of impressions, "
                 f"{r['click_share']:.1%} of clicks; amplification {r['amplification']:.2f} "
                 f"(vs. exposed pool {r['amplification_vs_pool']:.2f}); CTR {r['ctr']:.2%}.")
facts.append(f"Top 1% of exposed articles received {main['top_1pct_share']:.1%} of impressions; top 10%: "
             f"{main['top_10pct_share']:.1%}; Gini {main['gini']:.3f}.")
print("\n".join(f"FACT: {f}" for f in facts))
""")

md(r"""
## 13. Interpretation

Reviewed against the executed outputs of this notebook (MIND small, train + dev, 8,584,442 impressions).
Format: **FACT** (printed above) → **INTERPRETATION** → **LIMITATION**.

**1. Exposure is not proportional to the catalogue, and the pattern survives both baselines.**
- FACT: entertainment is 1.1% of catalogue articles but 5.4% of impressions (amplification 5.08; 4.55 vs. the
  exposed pool). Music 3.02, TV 2.77, movies 2.75, lifestyle 2.58 (lifestyle: 4.6% of catalogue → 11.8% of impressions).
  Sports is 29.7% of the catalogue but 11.0% of impressions (0.37; 0.39 vs. pool); weather 0.41, video 0.51, news 0.85.
- INTERPRETATION: in the observed logs, entertainment/lifestyle-type categories receive roughly 2.5–5× their
  catalogue presence, while sports, weather and video receive well under half. The ranking is nearly identical
  under the exposed-pool baseline, so it is not an artefact of history-only articles in `news.tsv`.
- LIMITATION: amplification describes the output of MSN's whole ecosystem (editorial placement, page layout,
  recommendation, user navigation). It does not show intent, and "catalogue share" counts articles, not
  their importance or the number of distinct stories.

**2. Exposure tracks clicks only loosely at category level.**
- FACT: sports has the highest CTR (5.75%) yet the lowest amplification among large categories; entertainment
  has one of the lowest CTRs (2.88%) yet the highest amplification. Across the 14 categories with ≥50 articles,
  Spearman ρ(amplification, CTR) = −0.08 (p = 0.78).
- INTERPRETATION: category-level exposure is not explained by category-level click-through in these logs —
  consistent with exposure being shaped by factors beyond observed engagement rates.
- LIMITATION: n = 14 categories; CTR here is conditional on the logging design (see Limitations: every logged
  impression contains ≥1 click). Not causal.

**3. Article-level exposure is extremely concentrated.**
- FACT: among the 22,771 articles shown at least once, Gini = 0.92; the top 1% (228 articles) received 34.7% of
  impressions and the top 10% received 91.2%; the bottom 50% received 0.4%. 1.95% of articles account for half
  of all impressions. Within single days, Gini is 0.89–0.93.
- INTERPRETATION: a very small set of articles captures most of the logged exposure. Because the concentration
  holds within each day, it is not only an effect of articles being published at different times.
- LIMITATION: the top-1% articles also have a higher pooled CTR (5.38% vs. 3.31% for other articles with
  ≥100 impressions), though the article-level rank correlation is weak (ρ = 0.06). Concentration may partly
  reflect demand; the observational data cannot separate the two.

**4. Most of the catalogue is never shown.**
- FACT: 34.9% of the 65,238 catalogue articles appear in at least one impression; the remaining 65.1% appear only
  in users' pre-period click histories.
- INTERPRETATION: `news.tsv` is mostly a record of older content, not the candidate pool during the logged week.
- LIMITATION: this is why the exposed-pool baseline is always reported next to catalogue share.
""")

md(r"""
## 14. Limitations

- **Click-conditioned logs.** Every logged impression in MIND small contains at least one click (validation:
  0 zero-click impressions in train or dev). CTR values are therefore conditional on sessions with engagement
  and overstate CTR across all MSN page views. Compare CTRs across categories only, never to external benchmarks.
- **Exposure proxy.** An impression means an article was logged as shown, not that it was seen or read.
  The order inside an impression list is not documented as display rank.
- **Catalogue definition.** 65.1% of `news.tsv` articles appear only in click histories (older content). Amplification
  vs. the full catalogue mixes availability with selection; the exposed-pool baseline (which conditions on being shown)
  is reported alongside, and both give the same ranking.
- **Short window.** Train covers 9–14 Nov 2019 and dev covers 15 Nov 2019; category shares shift between the two
  (largest: news 27.2% train vs. 23.4% dev), so single-week results may not generalise across time.
- **Sampled users.** Train and dev are largely different user samples (50,000 each, 5,943 in both).
- **Small categories.** kids (22 articles), middleeast (2), northamerica (1) and games (1) are too small for
  meaningful ratios and are excluded from headline claims.
- **Observed system ≠ algorithm.** Impressions result from MSN's whole ecosystem. Patterns are descriptive and do not
  establish intent or unfairness.
- **Scope.** US Microsoft News only; not representative of other platforms, countries, or Canadian audiences.
- **No demographics.** Users are anonymized; no demographic or geographic user analysis is possible or attempted.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = Path(__file__).with_name("01_big_picture.ipynb")
nbf.write(nb, out)
print("wrote", out)
