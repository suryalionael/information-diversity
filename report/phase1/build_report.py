"""Builds report/phase1/behind-the-feed-phase1.pdf from saved results and final figures.

Every number in the PDF is read from outputs/results/ (written by Notebooks 01–03);
nothing is typed in by hand. Pages whose inputs are missing render a visible
"PENDING" notice instead of content. Run after Notebook 04:

    python report/phase1/build_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src import style as S  # noqa: E402

import os  # noqa: E402
from src.data import output_dir  # noqa: E402

RES = output_dir() / "results"
FIG = output_dir() / "figures" / "final"
OUT = Path(os.environ.get("REPORT_PDF", Path(__file__).with_name("behind-the-feed-phase1.pdf")))
DATA_URL = "https://msnews.github.io/"
REPO_URL = "https://github.com/suryalionael/information-diversity"

# --------------------------------------------------------------------------- #
# Typography & layout
# --------------------------------------------------------------------------- #
for name, file in [("SS", "Regular"), ("SS-Semi", "Semibold"), ("SS-Bold", "Bold"), ("SS-Light", "Light"), ("SS-It", "It")]:
    pdfmetrics.registerFont(TTFont(name, str(ROOT / "assets" / "fonts" / f"SourceSans3-{file}.ttf")))
pdfmetrics.registerFontFamily("SS", normal="SS", bold="SS-Semi", italic="SS-It", boldItalic="SS-Semi")

W, H = letter
M = 46                       # page margin (pt)
CW = W - 2 * M               # content width
INK, INK2, MUTED, RULE = HexColor(S.INK), HexColor(S.INK2), HexColor(S.MUTED), HexColor(S.RULE)
ORANGE, BLUE, AQUA, TINT = HexColor(S.ORANGE), HexColor(S.BLUE), HexColor(S.AQUA), HexColor("#f6f5f1")

ST = {
    "kicker": ParagraphStyle("kicker", fontName="SS-Semi", fontSize=8.5, leading=11, textColor=ORANGE),
    "h1": ParagraphStyle("h1", fontName="SS-Semi", fontSize=25, leading=29, textColor=INK),
    "h2": ParagraphStyle("h2", fontName="SS-Semi", fontSize=20, leading=24, textColor=INK),
    "dek": ParagraphStyle("dek", fontName="SS", fontSize=12, leading=16.5, textColor=INK2),
    "body": ParagraphStyle("body", fontName="SS", fontSize=9.8, leading=13.6, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="SS", fontSize=8.6, leading=11.6, textColor=INK2),
    "cap": ParagraphStyle("cap", fontName="SS-Semi", fontSize=10, leading=13, textColor=INK),
    "note": ParagraphStyle("note", fontName="SS", fontSize=7.8, leading=10.2, textColor=MUTED),
    "statnum": ParagraphStyle("statnum", fontName="SS-Semi", fontSize=25, leading=27, textColor=ORANGE),
    "statlbl": ParagraphStyle("statlbl", fontName="SS", fontSize=8.8, leading=11.4, textColor=INK2),
}


def para(c, text: str, x: float, y_top: float, width: float, style: str) -> float:
    """Draw a wrapped paragraph with its top at y_top; returns the height used."""
    p = Paragraph(text, ST[style])
    _, h = p.wrap(width, H)
    p.drawOn(c, x, y_top - h)
    return h


def image(c, name: str, x: float, y_top: float, width: float, max_h: float | None = None) -> float:
    """Draw a figure scaled to ``width`` (or to ``max_h``); returns height used. Missing -> pending box."""
    path = FIG / f"{name}.png"
    if not path.exists():
        h = max_h or 120
        pending(c, x, y_top, width, h, f"figure {name} not generated yet")
        return h
    img = ImageReader(str(path))
    iw, ih = img.getSize()
    w, h = width, width * ih / iw
    if max_h and h > max_h:
        h, w = max_h, max_h * iw / ih
    c.drawImage(img, x, y_top - h, w, h, mask="auto")
    return h


def pending(c, x, y_top, w, h, why: str) -> None:
    c.setStrokeColor(ORANGE); c.setDash(4, 3); c.rect(x, y_top - h, w, h); c.setDash()
    para(c, f"<b>PENDING</b> — {why}. Run Notebooks 02–04 on the MIND data, then rebuild.", x + 10, y_top - 10, w - 20, "small")


def stat(c, x, y_top, w, number: str, label: str, color=ORANGE) -> float:
    st = ParagraphStyle("s", parent=ST["statnum"], textColor=color)
    p = Paragraph(number, st); _, h1 = p.wrap(w, H); p.drawOn(c, x, y_top - h1)
    h2 = para(c, label, x, y_top - h1 - 2, w, "statlbl")
    return h1 + h2 + 2


def rule(c, y, x0=M, x1=W - M, color=RULE, width=0.6):
    c.setStrokeColor(color); c.setLineWidth(width); c.line(x0, y, x1, y)


def frame(c, page: int, total: int = 5) -> None:
    rule(c, 34)
    c.setFont("SS", 7.6); c.setFillColor(MUTED)
    c.drawString(M, 23, "Behind the Feed · Phase 1  |  Data: Microsoft News Dataset (MIND), Microsoft Research — msnews.github.io")
    c.drawRightString(W - M, 23, f"{page} / {total}")


def kicker(c, text: str, y: float) -> float:
    return para(c, text.upper(), M, y, CW, "kicker")


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #
def load_headlines() -> dict:
    p = RES / "headline_metrics.json"
    return {r["metric"]: r for r in json.loads(p.read_text())} if p.exists() else {}


HL = load_headlines()


def v(metric: str):
    if metric not in HL:
        raise KeyError(f"headline metric missing: {metric}")
    return HL[metric]["value"]


def csv(name: str, **kw) -> pd.DataFrame | None:
    p = RES / name
    return pd.read_csv(p, **kw) if p.exists() else None


def period() -> str:
    val = json.loads((RES / "validation_01.json").read_text())
    t0 = pd.Timestamp(val["behaviors"]["train"]["time_min"])
    t1 = pd.Timestamp(val["behaviors"]["dev"]["time_max"])
    return f"{t0.day}–{t1.day} {t1:%b %Y}"


def robustness_sentence() -> str:
    rob = csv("narrowing_robustness.csv")
    if rob is None or "lift" not in rob:
        return ""
    r = rob.dropna(subset=["lift"]).iloc[1:]          # row 0 is the primary specification
    if r.empty:
        return ""
    same = int(((r["lift"] > 1) & (r["p_one_sided"] < 0.05)).sum())
    return (f" The tilt above the null held in {same} of {len(r)} alternative specifications (history thresholds, "
            f"train/dev-only exposure, activity strata).")


def pct(x: float, d: int = 0) -> str:
    return f"{x * 100:.{d}f}%"


def fmt_int(x) -> str:
    return f"{int(round(x)):,}"


def millions(x) -> str:
    return f"{x / 1e6:.1f} million"


def LBL(c):
    return S.category_label(c)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page1(c) -> None:
    cat = csv("category_metrics.csv", index_col=0)
    main = cat[cat["n_articles"] >= 50]
    hi, lo = main["amplification"].idxmax(), main["amplification"].idxmin()
    gain = (main["exposure_share"] - main["catalogue_share"]).idxmax()
    y = H - M
    y -= kicker(c, "Behind the Feed  ·  Phase 1 visual data story", y) + 6
    y -= para(c, "What gets shown — and what gets overlooked?", M, y, CW, "h1") + 8
    dek = (f"We traced <b>{millions(v('total_impressions'))}</b> logged article impressions from one week of Microsoft News "
           f"back to the <b>{fmt_int(v('catalogue_articles'))}</b> articles in the dataset. What readers were shown did not "
           f"mirror what was available: <b>{LBL(hi).lower()}</b> received {main.loc[hi, 'amplification']:.1f}× its share of the "
           f"catalogue, while <b>{LBL(lo).lower()}</b> — {pct(main.loc[lo, 'catalogue_share'])} of all articles — received "
           f"{main.loc[lo, 'amplification']:.2f}×.")
    y -= para(c, dek, M, y, CW, "dek") + 14

    tiles = [(pct(main.loc[lo, "catalogue_share"]), f"of articles were {LBL(lo).lower()}…", BLUE),
             (pct(main.loc[lo, "exposure_share"]), f"…but only this share of what users were shown", BLUE),
             (f"{main.loc[hi, 'amplification']:.1f}×", f"{LBL(hi)}: exposure relative to its catalogue share", ORANGE),
             (pct(main.loc[gain, "exposure_share"]), f"{LBL(gain)}: share of exposure, from {pct(main.loc[gain, 'catalogue_share'], 1)} of articles", ORANGE)]
    tw = (CW - 3 * 14) / 4
    hmax = 0
    for i, (n, lbl, col) in enumerate(tiles):
        hmax = max(hmax, stat(c, M + i * (tw + 14), y, tw, n, lbl, col))
    y -= hmax + 12
    rule(c, y); y -= 12

    y -= para(c, "Share of each news category: in the catalogue → in what users were shown → in what they clicked", M, y, CW, "cap") + 4
    y -= image(c, "fig1_slopegraph", M, y, CW, max_h=290) + 6
    y -= para(c, (f"Each line is one of the {len(main)} MIND categories with at least 50 articles. Orange: shown more than its catalogue "
                  "share; blue: shown less; grey: other categories."), M, y, CW, "note") + 12

    col = (CW - 20) / 2
    sp = main.loc[lo]
    ctr_rank = main["ctr"].rank(ascending=False)            # 1 = highest CTR
    if ctr_rank[lo] == 1:
        lead = (f"<b>{LBL(lo)} was clicked most often when shown.</b> Its click-through rate — clicks ÷ times shown — was "
                f"{pct(sp['ctr'], 2)}, the highest of any large category, yet it received the least exposure relative to its "
                f"catalogue presence.")
    else:
        lead = (f"<b>{LBL(lo)} was not ignored by readers.</b> Its click-through rate — clicks ÷ times shown — was "
                f"{pct(sp['ctr'], 2)} (rank {int(ctr_rank[lo])} of {len(main)}), yet it received the least exposure relative "
                f"to its catalogue presence.")
    hi_rank = int(main["ctr"].rank()[hi])                   # 1 = lowest CTR
    hi_txt = ("one of the lowest click-through rates" if hi_rank <= 3 else f"a click-through rate ranked {len(main) - hi_rank + 1} of {len(main)}")
    left = f"{lead} {LBL(hi)} had {hi_txt} ({pct(main.loc[hi, 'ctr'], 2)}) and the most amplified exposure."
    right = ("<b>What this is — and isn't.</b> An <i>impression</i> means an article was logged as shown; it is a proxy for "
             "exposure, not proof of reading. The pattern describes the output of MSN's whole system — editors, page layout, "
             "recommendation and user navigation — not the recommender alone, and it does not show intent or unfairness.")
    h1 = para(c, left, M, y, col, "body")
    h2 = para(c, right, M + col + 20, y, col, "body")
    frame(c, 1)


def page2(c) -> None:
    conc = csv("concentration_metrics.csv")
    cat = csv("category_metrics.csv", index_col=0)
    main = cat[cat["n_articles"] >= 50]
    m = conc.iloc[0]
    daily = conc[conc["population"].str.contains("day")]
    from src.metrics import top_share
    import numpy as np
    n_top1 = top_share(np.ones(int(m["n_items"])), 0.01)["n_items_top"]     # same rounding rule as the analysis
    y = H - M
    y -= kicker(c, "Who gets the spotlight?", y) + 6
    y -= para(c, f"{pct(m['top_1pct_share'], 0)} of all exposure went to 1% of articles", M, y, CW, "h2") + 8
    dek = (f"Of the {fmt_int(m['n_items'])} articles shown at least once, the most-shown {fmt_int(n_top1)} "
           f"received {pct(m['top_1pct_share'], 1)} of all impressions and the top 10% received {pct(m['top_10pct_share'], 1)}. "
           f"The bottom half of articles shared {pct(m['bottom_50pct_share'], 1)}.")
    y -= para(c, dek, M, y, CW, "dek") + 10
    y -= image(c, "fig4_concentration_bars", M, y, CW) + 4
    stable = daily["gini"].min() >= m["gini"] - 0.05
    tail = ("so the concentration is not just older and newer articles being mixed together." if stable else
            "so part of the overall concentration reflects articles published on different days.")
    y -= para(c, (f"Gini coefficient of impressions per article: {m['gini']:.2f} (0 = every article shown equally, 1 = one article "
                  f"shown everywhere). Within single days it ranges from {daily['gini'].min():.2f} to {daily['gini'].max():.2f}, "
                  f"{tail}"), M, y, CW, "small") + 14
    rule(c, y); y -= 14

    corr = csv("category_exposure_engagement.csv")
    r = corr.iloc[0] if corr is not None else None
    if r is None or r["p"] >= 0.05:
        h2txt = "Exposure tracks neither catalogue size nor clicks"
    else:
        h2txt = "Exposure only partly follows click rates" if r["rho"] > 0 else "Exposure runs against click rates"
    y -= para(c, h2txt, M, y, CW, "h2") + 6
    y -= para(c, ("Categories ranked by exposure relative to their catalogue share (left), and the same ratio against how often "
                  "each category was clicked when shown (right)."), M, y, CW, "dek") + 8
    col = (CW - 18) / 2
    h1 = image(c, "fig2_amplification", M, y, col, max_h=255)
    h2 = image(c, "fig3_ctr_vs_amplification", M + col + 18, y, col, max_h=255)
    y -= max(h1, h2) + 8
    if r is not None:
        verdict = "no detectable relationship" if r["p"] >= 0.05 else "a statistically detectable relationship"
        rtxt = (f"Across the {int(r['n'])} large categories, the rank correlation between exposure amplification and "
                f"click-through rate is ρ = {r['rho']:.2f} (p = {r['p']:.2f}): {verdict}.")
    else:
        rtxt = ""
    n_ex = len(cat) - len(main)
    words = {1: "One category", 2: "Two categories", 3: "Three categories", 4: "Four categories", 5: "Five categories"}
    note = (f"{rtxt} Click-through rates are only comparable inside this dataset: every logged impression contains at least one "
            f"click, so they describe engaged sessions. {words.get(n_ex, f'{n_ex} categories')} with fewer than 50 articles "
            f"{'is' if n_ex == 1 else 'are'} excluded.")
    para(c, note, M, y, CW, "small")
    frame(c, 2)


def page3(c) -> None:
    y = H - M
    y -= kicker(c, "The shrinking window?", y) + 6
    q = csv("narrowing_alignment_by_quintile.csv")
    need = ["alignment_lift", "alignment_observed_share_own_dom", "alignment_null_share_own_dom",
            "narrowing_users_eligible", "exp_eff_categories_median", "hist_eff_categories_median"]
    if q is None or any(k not in HL for k in need):
        y -= para(c, "Does past clicking narrow what users are shown later?", M, y, CW, "h2") + 10
        pending(c, M, y, CW, 200, "Notebook 02 results not found")
        frame(c, 3); return
    lift, obs, null = v("alignment_lift"), v("alignment_observed_share_own_dom"), v("alignment_null_share_own_dom")
    p_val = v("alignment_p")
    lift_all = v("alignment_lift_all_shown")
    sig = lift > 1 and p_val < 0.05
    if sig and lift >= 1.05:
        head = "Past clicks echo in what users are shown next"
    elif lift <= 0.95:
        head = "Users were shown less of their past favourites than chance"
    else:
        head = "Past clicks barely shape what users are shown next"
    q1 = q["lift"].iat[0]
    qmax = q.loc[q["lift"].idxmax()]
    if qmax["hist_quintile"] == q["hist_quintile"].iat[-1] and qmax["lift"] > q1:
        qtxt = (f" The tilt was largest for users whose past clicks were most concentrated ({qmax['lift']:.2f}× vs "
                f"{q1:.2f}× for the broadest).")
    elif qmax["lift"] > q1:
        qtxt = f" The tilt varied across groups ({q['lift'].min():.2f}×–{q['lift'].max():.2f}×) without a steady trend."
    else:
        qtxt = f" The tilt did not grow with how concentrated past clicks were ({q1:.2f}× for the broadest group)."
    y -= para(c, head, M, y, CW, "h2") + 8
    min_hist = HL["narrowing_users_eligible"]["filters"].split("≥")[-1].strip()
    dek = (f"For {fmt_int(v('narrowing_users_eligible'))} users with at least {min_hist} clicks <i>before</i> the logged week, their "
           f"most-clicked category made up <b>{pct(obs, 1)}</b> of the articles they were shown but did not click "
           f"<i>during</i> it — versus {pct(null, 1)} if exposure had ignored their history (a {lift:.2f}× tilt; "
           f"{lift_all:.2f}× counting every article shown).{qtxt}")
    y -= para(c, dek, M, y, CW, "dek") + 10
    col = (CW - 18) / 2
    para(c, "Exposure to each user's favourite category", M, y, col, "cap")
    bc = csv("narrowing_alignment_by_category.csv")
    more = bc is not None and (bc["exposure_ratio"] > 1).mean() > 0.5
    para(c, "Users who mostly clicked a category see more of it" if more else "Past favourites vs. later exposure, by category",
         M + col + 18, y, col, "cap")
    y -= 16
    h1 = image(c, "fig5_alignment", M, y, col, max_h=215)
    h2 = image(c, "fig5b_alignment_by_category", M + col + 18, y, col, max_h=215)
    y -= max(h1, h2) + 12
    rule(c, y); y -= 12
    he, ee = v("hist_eff_categories_median"), v("exp_eff_categories_median")
    share = v("share_users_history_more_concentrated_than_exposure")
    y -= para(c, "But exposure stays broader than clicks" if ee > he and share > 0.5 else "Exposure is not broader than clicks",
              M, y, CW, "h2") + 6
    act = csv("narrowing_activity_correlations.csv")
    arho = act.iloc[0]["rho"] if act is not None else float("nan")
    if abs(arho) < 0.1:
        atxt = "Users with longer click histories were not shown measurably narrower feeds"
    elif arho > 0:
        atxt = "Users with longer click histories were shown somewhat more concentrated feeds"
    else:
        atxt = "Users with longer click histories were shown somewhat broader feeds"
    body = (f"At the median, a user's past clicks covered the equivalent of <b>{he:.1f}</b> evenly-used categories; what they "
            f"were shown covered <b>{ee:.1f}</b>. For {pct(share)} of users, past clicks were more concentrated than later "
            f"exposure. {atxt} (rank correlation between history length and exposure concentration ρ = {arho:.2f}).")
    para(c, body, M, y, col, "body")
    h = image(c, "fig7_activity_groups", M + col + 18, y + 4, col, max_h=175)
    y -= max(h, 120) + 4
    para(c, (f"Associative design: history (clicks before the logs) precedes logged exposure ({period()}) but has no timestamps; "
             "this cannot show that past clicks <i>caused</i> later exposure. Logged impressions all contain a click, which favours "
             "users' interests, so the tilt is measured on articles shown but <i>not</i> clicked (a conservative choice). "
             "“Effective categories” = 1 ÷ the median chance that two random items share a category. Groups are behavioural, "
             "never demographic."),
         M, y, CW, "note")
    frame(c, 3)


def page4(c) -> None:
    y = H - M
    y -= kicker(c, "Relevance vs. diversity — a simulation", y) + 6
    sel = csv("recommender_mmr_selected.csv")
    sm = csv("recommender_metrics.csv")
    setup = csv("recommender_setup.csv")
    if sel is None or sm is None:
        y -= para(c, "Can a feed be relevant and diverse?", M, y, CW, "h2") + 10
        pending(c, M, y, CW, 200, "Notebook 03 results not found")
        frame(c, 4); return
    base = sm[sm["lam"] == 1.0].set_index("model")
    best = base.drop(index="random")["ndcg10"].idxmax()
    pick = sel.set_index("base").loc[best]
    names = {"content": "topic-match", "category": "favourite-category", "popularity": "most-clicked-last-week"}
    gain, loss = pick["distinct_categories_rel_change"], pick["ndcg10_rel_change"]
    good = pick["lambda_selected"] < 1 and gain >= 0.2
    if good:
        head = "A little diversity costs very little relevance"
    elif pick["lambda_selected"] < 1:
        head = "Cheap diversity was available, but only a little of it"
    else:
        head = "In this simulation, even a little diversity cost relevance"
    y -= para(c, head, M, y, CW, "h2") + 8
    dek = (f"We built three simple, transparent recommenders and re-ranked each one with <b>Maximal Marginal Relevance</b> "
           f"(MMR), which trades predicted relevance against similarity to articles already picked. For the most accurate "
           f"strategy ({names[best]}), moving the dial to λ = {pick['lambda_selected']:g} changed the number of categories in "
           f"a top-10 list by <b>{gain:+.0%}</b> ({pick['distinct_categories_base']:.1f} → "
           f"{pick['distinct_categories_selected']:.1f}) while relevance changed by <b>{loss:+.1%}</b>.")
    y -= para(c, dek, M, y, CW, "dek") + 10
    wl, wr = CW * 0.62, CW * 0.36
    para(c, "Each dot is one setting of the diversity dial", M, y, wl, "cap")
    para(c, "Do recommenders amplify favourites?", M + wl + CW * 0.02, y, wr, "cap")
    y -= 16
    h1 = image(c, "fig8_frontier", M, y, wl, max_h=245)
    h2 = image(c, "fig9_rec_amplification", M + wl + CW * 0.02, y, wr, max_h=245)
    y -= max(h1, h2) + 10
    amp_c = base.loc["category", "amp_vs_history_mean"]
    amp_t = base.loc["content", "amp_vs_history_mean"]
    seen = setup["dev_candidates_with_train_clicks"].iat[0] if setup is not None else float("nan")
    min_c = int(setup["min_candidates"].iat[0]) if setup is not None else "?"
    col = (CW - 20) / 2
    amp_r = base.loc["random", "amp_vs_history_mean"]
    if amp_c > 1 and amp_t > 1:
        verdict = "Personalisation concentrates."
    elif max(amp_c, amp_t) > 1:
        verdict = "One strategy concentrates."
    else:
        verdict = "Personalisation did not concentrate lists."
    left = (f"<b>{verdict}</b> Ranked purely by past behaviour, the favourite-category strategy filled "
            f"top-10 lists with the user's favourite category at {amp_c:.2f}× its share in their own history; topic-matching "
            f"did so at {amp_t:.2f}×, and a random order of the same articles at {amp_r:.2f}×. Above 1× means the list is "
            f"<i>narrower</i> than the user's own past clicks.")
    if good:
        right = ("<b>Phase 2 direction.</b> The dial is cheap, transparent and explainable. A user-facing “breadth” control — "
                 "with the relevance cost shown openly — is a concrete design we can prototype and test with people, rather "
                 "than a promise that diversity is free.")
    else:
        right = ("<b>Phase 2 direction.</b> Because extra breadth carried a visible relevance cost here, any “breadth” control "
                 "should let people choose the trade-off themselves, with the cost shown openly — a design to prototype and "
                 "test, not a promise that diversity is free.")
    h1 = para(c, left, M, y, col, "body"); h2 = para(c, right, M + col + 20, y, col, "body")
    y -= max(h1, h2) + 8
    para(c, (f"Simulated strategies — they do not reproduce MSN's recommender. Evaluated on {fmt_int(base.loc[best, 'n_impressions'])} "
             f"real dev-day impressions (≥{min_c} shown articles): re-ranking only the articles actually shown, scored against real "
             f"clicks (nDCG@10). Models used only information available beforehand (pre-period histories, train-week clicks); "
             f"{pct(seen)} of dev-day candidate articles had any train-week click."), M, y, CW, "note")
    frame(c, 4)


def page5(c) -> None:
    from scipy.stats import spearmanr
    cat = csv("category_metrics.csv", index_col=0)
    main = cat[cat["n_articles"] >= 50]
    rank_r = spearmanr(main["amplification"], main["amplification_vs_pool"]).statistic
    hist_only = 1 - v("share_of_catalogue_ever_shown")
    y = H - M
    y -= kicker(c, "Methods & limitations", y) + 6
    y -= para(c, "How we know — and what we can't conclude", M, y, CW, "h2") + 12
    col = (CW - 22) / 2
    left = [
        ("Data", f"Microsoft News Dataset (MIND-small), Microsoft Research: {fmt_int(v('catalogue_articles'))} news articles with "
                 f"categories, and anonymised logs of {fmt_int(v('unique_users'))} users — {fmt_int(v('rows_behaviors_train') + v('rows_behaviors_dev'))} "
                 f"impression logs ({period()}) containing {fmt_int(v('total_impressions'))} article impressions and "
                 f"{fmt_int(v('total_clicks'))} clicks."),
        ("Definitions", "<b>Exposure</b> = an article logged as shown in an impression (a proxy, not proof of reading). "
                        "<b>Engagement</b> = a click. <b>Catalogue share</b> = articles in a category ÷ all articles; "
                        "<b>exposure share</b> = impressions ÷ all impressions; <b>CTR</b> = clicks ÷ impressions in that category."),
        ("Amplification", f"Exposure share ÷ catalogue share (1× = proportional). Because {pct(hist_only)} of catalogue "
                          f"articles appear only in users' older click histories, we repeated it against articles actually shown "
                          f"at least once: {'the category ranking barely changes' if rank_r >= 0.9 else 'the category ranking shifts'} (rank correlation {rank_r:.2f})."),
        ("Concentration", "Top-1%/10% shares and the Gini coefficient of impressions per article, also computed day by day."),
        ("Narrowing", "Unbiased Simpson concentration (chance two items share a category), entropy, and a permutation test on "
                      "articles shown but not clicked (logged impressions all contain a click, which favours interests): "
                      "users' favourite categories are shuffled within groups of similar activity and period to estimate exposure "
                      "if it ignored history." + robustness_sentence()),
        ("Recommenders", "Popularity (train-week clicks), TF-IDF title similarity to the user's history, and favourite-category "
                         "preference; MMR re-ranking with λ from 1 to 0. Evaluated on the dev day with nDCG@10, AUC, distinct "
                         "categories, intra-list diversity, coverage and an amplification ratio. No dev clicks used as input."),
    ]
    right = [
        "Observational data: associations, not causes.",
        "One week of US Microsoft News; not the whole internet, other platforms, Canada, or other years.",
        "Users are anonymised; no demographic or location data exist, and none were inferred. Groups are behavioural.",
        "Impressions are an exposure proxy; they do not prove an article was seen or read.",
        "Every logged impression contains at least one click, so click-through rates describe engaged sessions and are "
        "comparable only within this dataset.",
        "Click histories pre-date the logs but have no timestamps and never update during the logged week.",
        "Observed exposure reflects MSN's whole ecosystem — editors, layout, recommendation, navigation — not one algorithm.",
        "Our recommenders are simulations on logged candidate lists; they do not reproduce MSN's proprietary system.",
        f"Diversity is measured with {len(cat)} broad categories and title words; viewpoint and source diversity are not measured.",
    ]
    yl = y
    for title, text in left:
        yl -= para(c, f"<b>{title}.</b> {text}", M, yl, col, "body") + 7
    xr = M + col + 22
    c.setFillColor(TINT); c.roundRect(xr - 10, yl - 4, col + 10, y - yl + 10, 4, stroke=0, fill=1)
    yr = y - 4
    yr -= para(c, "<b>Limitations</b>", xr, yr, col - 10, "body") + 4
    for t in right:
        yr -= para(c, f"•&nbsp;&nbsp;{t}", xr, yr, col - 14, "small") + 4
    y = min(yl, yr) - 14
    rule(c, y); y -= 12
    refs = (f"<b>Sources.</b> Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. <i>Proceedings of ACL 2020</i>, "
            f"3597–3606. Dataset: Microsoft News Dataset (MIND), Microsoft Research — {DATA_URL} (Microsoft Research License "
            f"Terms). Carbonell, J. &amp; Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents "
            f"and producing summaries. <i>SIGIR '98</i>. <b>Reproducibility.</b> All code, notebooks and the result files behind "
            f"every number: {REPO_URL}. The dataset itself is not redistributed.")
    para(c, refs, M, y, CW, "small")
    frame(c, 5)


def build() -> Path:
    if not HL:
        raise SystemExit("outputs/results/headline_metrics.json not found — run the notebooks first")
    c = canvas.Canvas(str(OUT), pagesize=letter)
    c.setTitle("Behind the Feed — Does personalized exposure come at the cost of diversity?")
    c.setAuthor("information-diversity project"); c.setSubject("TECHNATION Behind the Feed — Phase 1")
    for page in (page1, page2, page3, page4, page5):
        page(c)
        c.showPage()
    c.save()
    print("wrote", OUT)
    return OUT


if __name__ == "__main__":
    build()
