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

NAVY, ON_DARK, ON_DARK2 = HexColor("#0f1c2e"), HexColor("#ffffff"), HexColor("#c4cdd9")
HERO_ORANGE = HexColor("#ff7a45")

ST = {
    "kicker": ParagraphStyle("kicker", fontName="SS-Semi", fontSize=8.5, leading=11, textColor=ORANGE),
    "h1": ParagraphStyle("h1", fontName="SS-Semi", fontSize=25, leading=29, textColor=INK),
    "h2": ParagraphStyle("h2", fontName="SS-Semi", fontSize=20, leading=24, textColor=INK),
    "dek": ParagraphStyle("dek", fontName="SS", fontSize=12, leading=16.5, textColor=INK2),
    "body": ParagraphStyle("body", fontName="SS", fontSize=9.8, leading=13.6, textColor=INK, alignment=TA_LEFT),
    "small": ParagraphStyle("small", fontName="SS", fontSize=8.6, leading=11.6, textColor=INK2),
    "cap": ParagraphStyle("cap", fontName="SS-Semi", fontSize=10, leading=13, textColor=INK),
    "note": ParagraphStyle("note", fontName="SS", fontSize=7.8, leading=10.2, textColor=MUTED),
    "kickw": ParagraphStyle("kickw", fontName="SS-Semi", fontSize=8.8, leading=11, textColor=HERO_ORANGE),
    "h1w": ParagraphStyle("h1w", fontName="SS-Bold", fontSize=26, leading=29, textColor=ON_DARK),
    "h2w": ParagraphStyle("h2w", fontName="SS-Bold", fontSize=20, leading=23, textColor=ON_DARK),
    "dekw": ParagraphStyle("dekw", fontName="SS", fontSize=11, leading=14.6, textColor=ON_DARK2),
    "bylinew": ParagraphStyle("bylinew", fontName="SS-Semi", fontSize=10, leading=13, textColor=ON_DARK),
    "hero": ParagraphStyle("hero", fontName="SS-Bold", fontSize=50, leading=52, textColor=HERO_ORANGE),
    "herolbl": ParagraphStyle("herolbl", fontName="SS", fontSize=9.4, leading=12, textColor=ON_DARK2),
    "bodyw": ParagraphStyle("bodyw", fontName="SS", fontSize=9.8, leading=13.6, textColor=ON_DARK2),
    "capw": ParagraphStyle("capw", fontName="SS-Bold", fontSize=12.5, leading=15, textColor=HERO_ORANGE),
    "sub": ParagraphStyle("sub", fontName="SS-Semi", fontSize=18, leading=22, textColor=INK),
    "statnum": ParagraphStyle("statnum", fontName="SS-Bold", fontSize=30, leading=32, textColor=ORANGE),
    "statlbl": ParagraphStyle("statlbl", fontName="SS", fontSize=8.8, leading=11.4, textColor=INK2),
}


def para(c, text: str, x: float, y_top: float, width: float, style: str) -> float:
    """Draw a wrapped paragraph with its top at y_top; returns the height used."""
    p = Paragraph(mn(text), ST[style])
    _, h = p.wrap(width, H)
    p.drawOn(c, x, y_top - h)
    return h


def image(c, name: str, x: float, y_top: float, width: float, max_h: float | None = None, center: bool = False) -> float:
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
        if center:
            x += (width - w) / 2
    c.drawImage(img, x, y_top - h, w, h, mask="auto")
    return h


def pending(c, x, y_top, w, h, why: str) -> None:
    c.setStrokeColor(ORANGE); c.setDash(4, 3); c.rect(x, y_top - h, w, h); c.setDash()
    para(c, f"<b>PENDING:</b> {why}. Run Notebooks 02–04 on the MIND data, then rebuild.", x + 10, y_top - 10, w - 20, "small")


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
    c.drawString(M, 23, "Signal & Noise  ·  Behind the Feed, Phase 1  |  Data: Microsoft News Dataset (MIND), Microsoft Research, msnews.github.io")
    c.drawRightString(W - M, 23, f"{page} / {total}")


def header(c, kick: str, title: str, dek: str, hero: str | None = None, hero_lbl: str = "",
           byline: str | None = None, big: bool = False) -> float:
    """Full-bleed dark header band with white headline and optional hero number. Returns y below the band."""
    tw = CW * (0.66 if hero else 1.0)
    pieces = [(kick.upper(), "kickw", 7), (title, "h1w" if big else "h2w", 6 if byline else 9)]
    if byline:
        pieces.append((byline, "bylinew", 9))
    pieces.append((dek, "dekw", 0))
    heights = [Paragraph(mn(t), ST[st]).wrap(tw, H)[1] for t, st, _ in pieces]
    text_h = sum(heights) + sum(g for _, _, g in pieces)
    hero_h = 0
    if hero:
        hw = CW * 0.30
        hero_h = Paragraph(hero, ST["hero"]).wrap(hw, H)[1] + 6 + Paragraph(hero_lbl, ST["herolbl"]).wrap(hw, H)[1]
    band = max(text_h, hero_h) + 2 * 24
    c.setFillColor(NAVY); c.rect(0, H - band, W, band, stroke=0, fill=1)
    c.setFillColor(HERO_ORANGE); c.rect(0, H - band, W, 3.2, stroke=0, fill=1)
    y = H - 24
    for (t, st, gap), h in zip(pieces, heights):
        para(c, t, M, y, tw, st); y -= h + gap
    if hero:
        xh = M + CW * 0.70
        c.setStrokeColor(HexColor("#2c3b52")); c.setLineWidth(0.8)
        c.line(xh - 14, H - band + 20, xh - 14, H - 20)
        top = H - band / 2 + hero_h / 2
        hn = para(c, hero, xh, top, hw, "hero")
        para(c, hero_lbl, xh, top - hn - 6, hw, "herolbl")
    return H - band - 16


def subhead(c, text: str, y: float) -> float:
    """Section heading with a short orange accent bar."""
    c.setFillColor(ORANGE); c.rect(M, y - 3, 34, 3.2, stroke=0, fill=1)
    return para(c, text, M, y - 9, CW, "sub") + 12


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
    return (f" The tilt beat that baseline in {same} of {len(r)} alternative setups (history thresholds of 2 to 20 clicks, "
            f"exposure from 9–14 Nov only or 15 Nov only, and groups of similar activity).")


def mn(text: str) -> str:
    """Typographic minus for negative numbers."""
    import re
    return re.sub(r"(?<![\w])-(?=\d)", "−", text)


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
    dek = (f"We matched <b>{millions(v('total_impressions'))}</b> logged article impressions from one week of US Microsoft "
           f"News (MSN) against the <b>{fmt_int(v('catalogue_articles'))}</b> articles in the dataset, and the two don't line up. "
           f"<b>{LBL(hi)}</b> got {main.loc[hi, 'amplification']:.1f}× its share of the catalogue; <b>{LBL(lo).lower()}</b>, "
           f"{pct(main.loc[lo, 'catalogue_share'])} of all articles, got {main.loc[lo, 'amplification']:.2f}×.")
    y = header(c, "Signal &amp; Noise  ·  Behind the Feed, Phase 1", "What gets shown — and what gets overlooked?", dek,
               byline="Lionael Surya and Margil Parekh", big=True) - 2

    tiles = [(pct(main.loc[lo, "catalogue_share"]), f"of articles were {LBL(lo).lower()}…", BLUE),
             (pct(main.loc[lo, "exposure_share"]), f"…but only this share of what users saw", BLUE),
             (f"{main.loc[hi, 'amplification']:.1f}×", f"{LBL(hi)}: exposure relative to its catalogue share", ORANGE),
             (pct(main.loc[gain, "exposure_share"]), f"{LBL(gain)}'s share of exposure, from {pct(main.loc[gain, 'catalogue_share'], 1)} of articles", ORANGE)]
    tw = (CW - 3 * 14) / 4
    hmax = 0
    for i, (n, lbl, col) in enumerate(tiles):
        c.setFillColor(col); c.rect(M + i * (tw + 14), y - 1, tw, 2.4, stroke=0, fill=1)
        hmax = max(hmax, stat(c, M + i * (tw + 14), y - 8, tw, n, lbl, col))
    y -= hmax + 20

    y -= para(c, "Each category's share of the catalogue, of what users were shown, and of what they clicked", M, y, CW, "cap") + 4
    y -= image(c, "fig1_slopegraph", M, y, CW, max_h=212, center=True) + 6
    y -= para(c, (f"Each line is one of the {len(main)} MIND categories with 50+ articles. Orange means shown more than its catalogue "
                  "share, blue means shown less, and grey lines are the rest."), M, y, CW, "note") + 12

    col = (CW - 20) / 2
    sp = main.loc[lo]
    rob = csv("category_baseline_robustness.csv")
    lowest_both = rob is None or rob["lowest_vs_pool"].iat[0] == lo
    least = ("the least exposure relative to its catalogue presence" if lowest_both else
             f"the least exposure relative to the catalogue ({sp['amplification']:.2f}×; measured against articles that were "
             f"actually shown it's {sp['amplification_vs_pool']:.2f}×, second-lowest after "
             f"{LBL(rob['lowest_vs_pool'].iat[0]).lower()})")
    ctr_rank = main["ctr"].rank(ascending=False)            # 1 = highest CTR
    if ctr_rank[lo] == 1:
        lead = (f"<b>When {LBL(lo).lower()} did appear, people clicked it more readily than any other large category.</b> "
                f"Its click-through rate (clicks ÷ times shown) hit {pct(sp['ctr'], 2)}, and yet it got {least}.")
    else:
        lead = (f"<b>{LBL(lo)} still drew clicks.</b> Its click-through rate (clicks ÷ times shown) was "
                f"{pct(sp['ctr'], 2)}, rank {int(ctr_rank[lo])} of {len(main)}, and yet it got {least}.")
    hi_rank = int(main["ctr"].rank()[hi])                   # 1 = lowest CTR
    hi_txt = ("one of the lowest click-through rates" if hi_rank <= 3 else f"a click-through rate ranked {len(main) - hi_rank + 1} of {len(main)}")
    left = f"{lead} {LBL(hi)} went the other way, with {hi_txt} ({pct(main.loc[hi, 'ctr'], 2)}) and the most amplified exposure."
    right = ("<b>Read this carefully.</b> An <i>impression</i> means the log says an article was shown. We use it as a "
             "stand-in for exposure, and it doesn't prove anyone read the thing. These numbers come out of the whole Microsoft "
             "News system (editors, page layout, recommendation and user navigation together), so they can't tell you anyone "
             "intended this, or that it's unfair.")
    h1 = para(c, left, M, y, col, "body")
    h2 = para(c, right, M + col + 20, y, col, "body")
    y -= max(h1, h2) + 14
    q = ("<b>The question we set out to answer: does personalized news exposure cost people diversity?</b> A feed decides "
         "which stories millions of people run into every day. If what it shows drifts away from what's available, or keeps "
         "narrowing toward whatever each person already clicks, readers end up with a thinner slice of the news and have no way "
         "of noticing. Pages 2 and 3 measure that in the real logs; page 4 runs a simulation to see whether a feed can stay "
         "relevant while showing more.")
    hq = Paragraph(mn(q), ST["body"]).wrap(CW - 24, H)[1]
    c.setFillColor(TINT); c.roundRect(M, y - hq - 16, CW, hq + 16, 4, stroke=0, fill=1)
    para(c, q, M + 12, y - 8, CW - 24, "body")
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
    dek = (f"Of the {fmt_int(m['n_items'])} articles shown at least once, the {fmt_int(n_top1)} most-shown took "
           f"{pct(m['top_1pct_share'], 1)} of all impressions and the top tenth took {pct(m['top_10pct_share'], 1)}, leaving "
           f"the bottom half with {pct(m['bottom_50pct_share'], 1)} between them.")
    y = header(c, "Who gets the spotlight?", f"{pct(m['top_1pct_share'], 0)} of all exposure went to 1% of articles", dek,
               hero=f"{m['gini']:.2f}", hero_lbl="Gini coefficient of exposure across articles (1 would mean one article got it all)")
    y -= image(c, "fig4_concentration_bars", M, y, CW) + 4
    stable = daily["gini"].min() >= m["gini"] - 0.05
    tail = ("so this isn't just old and new articles being lumped together." if stable else
            "which means part of the overall figure comes from articles published on different days.")
    y -= para(c, (f"Gini coefficient of impressions per article: {m['gini']:.2f} (0 would mean every article got shown equally, "
                  f"1 that a single article got everything). Inside any single day it stays between {daily['gini'].min():.2f} "
                  f"and {daily['gini'].max():.2f}, {tail}"), M, y, CW, "small") + 14
    rule(c, y); y -= 14

    corr = csv("category_exposure_engagement.csv")
    r = corr.iloc[0] if corr is not None else None
    if r is None or r["p"] >= 0.05:
        h2txt = "Over-exposure has nothing to do with click-through rate"
    else:
        h2txt = "Exposure only partly follows click rates" if r["rho"] > 0 else "Exposure runs against click rates"
    y -= subhead(c, h2txt, y)
    y -= para(c, ("Left: categories ranked by how much exposure they got relative to their catalogue share. Right: that same "
                  "ratio against how often people clicked each category when it appeared."), M, y, CW, "dek") + 8
    col = (CW - 18) / 2
    h1 = image(c, "fig2_amplification", M, y, col, max_h=232)
    h2 = image(c, "fig3_ctr_vs_amplification", M + col + 18, y, col, max_h=232)
    y -= max(h1, h2) + 8
    if r is not None:
        verdict = "which is no detectable relationship at all" if r["p"] >= 0.05 else "a statistically detectable relationship"
        rtxt = (f"Across the {int(r['n'])} large categories the rank correlation between amplification and click-through "
                f"rate is ρ = {r['rho']:.2f} (p = {r['p']:.2f}), {verdict}.")
    else:
        rtxt = ""
    n_ex = len(cat) - len(main)
    words = {1: "One category", 2: "Two categories", 3: "Three categories", 4: "Four categories", 5: "Five categories"}
    nums = {1: "one category", 2: "two categories", 3: "three categories", 4: "four categories", 5: "five categories"}
    note = (f"{rtxt} Click-through rates only compare fairly inside this dataset, because every logged impression contains at "
            f"least one click and so the logs describe engaged sessions. We left out {nums.get(n_ex, f'{n_ex} categories')} "
            f"with fewer than 50 articles.")
    para(c, note, M, y, CW, "small")
    frame(c, 2)


def page3(c) -> None:
    q = csv("narrowing_alignment_by_quintile.csv")
    need = ["alignment_lift", "alignment_observed_share_own_dom", "alignment_null_share_own_dom",
            "narrowing_users_eligible", "exp_eff_categories_median", "hist_eff_categories_median"]
    if q is None or any(k not in HL for k in need):
        y = header(c, "The shrinking window?", "Does past clicking narrow what users are shown later?", "")
        pending(c, M, y, CW, 200, "Notebook 02 results not found")
        frame(c, 3); return
    lift, obs, null = v("alignment_lift"), v("alignment_observed_share_own_dom"), v("alignment_null_share_own_dom")
    p_val = v("alignment_p")
    lift_all = v("alignment_lift_all_shown")
    sig = lift > 1 and p_val < 0.05
    if sig and lift >= 1.05:
        head = "Past favourites show up more than chance predicts"
    elif lift <= 0.95:
        head = "Users saw less of their past favourites than chance"
    else:
        head = "Past clicks barely shape what users see next"
    q1 = q["lift"].iat[0]
    qmax = q.loc[q["lift"].idxmax()]
    if qmax["hist_quintile"] == q["hist_quintile"].iat[-1] and qmax["lift"] > q1:
        qtxt = (f" People with the most concentrated click histories saw the biggest tilt ({qmax['lift']:.2f}×, against "
                f"{q1:.2f}× for the broadest).")
    elif qmax["lift"] > q1:
        qtxt = f" The tilt moved between {q['lift'].min():.2f}× and {q['lift'].max():.2f}× across groups with no steady trend."
    else:
        qtxt = f" The tilt didn't grow with how concentrated people's past clicks were ({q1:.2f}× for the broadest group)."
    min_hist = HL["narrowing_users_eligible"]["filters"].split("≥")[-1].strip()
    dek = (f"We took {fmt_int(v('narrowing_users_eligible'))} users with at least {min_hist} clicks <i>before</i> the logged "
           f"week and looked at the articles they were shown but didn't click <i>during</i> it. Their most-clicked category made "
           f"up <b>{pct(obs, 1)}</b> of those, against roughly {pct(null, 1)} under a history-blind baseline: a {lift:.2f}× tilt "
           f"({lift_all:.2f}× if you count every article shown).{qtxt}")
    y = header(c, "The shrinking window?", head, dek, hero=f"{lift:.2f}×",
               hero_lbl="tilt toward each user's past favourite category, in articles shown but not clicked")
    col = (CW - 18) / 2
    para(c, "How much of each feed went to the user's favourite", M, y, col, "cap")
    bc = csv("narrowing_alignment_by_category.csv")
    more = bc is not None and (bc["exposure_ratio"] > 1).mean() > 0.5
    para(c, "Past fans of a category see more of it than others" if more else
         "Past favourites vs. later exposure, by category",
         M + col + 18, y, col, "cap")
    y -= 16
    h1 = image(c, "fig5_alignment", M, y, col, max_h=215)
    h2 = image(c, "fig5b_alignment_by_category", M + col + 18, y, col, max_h=215)
    y -= max(h1, h2) + 12
    rule(c, y); y -= 12
    he, ee = v("hist_eff_categories_median"), v("exp_eff_categories_median")
    share = v("share_users_history_more_concentrated_than_exposure")
    y -= subhead(c, "Still, the feed stayed wider than the clicks" if ee > he and share > 0.5 else
                 "The feed was no wider than the clicks", y)
    act = csv("narrowing_activity_correlations.csv")
    arow = act.iloc[0] if act is not None else None
    arho = arow["rho"] if arow is not None else float("nan")
    ci_has_zero = arow is not None and arow["ci_low"] <= 0 <= arow["ci_high"]
    if abs(arho) < 0.1 and ci_has_zero:
        atxt = "Users with longer click histories didn't get measurably narrower feeds"
    elif abs(arho) < 0.1:
        atxt = (f"Longer click histories went with feeds that were only negligibly "
                f"{'more' if arho > 0 else 'less'} concentrated")
    elif arho > 0:
        atxt = "Users with longer click histories were shown somewhat more concentrated feeds"
    else:
        atxt = "Users with longer click histories were shown somewhat broader feeds"
    body = (f"At the median, a user's past clicks spread across the equivalent of <b>{he:.1f}</b> evenly used categories, "
            f"while what they were shown spread across <b>{ee:.1f}</b>. For {pct(share)} of users, past clicks were more "
            f"concentrated than later exposure. {atxt} (rank correlation between history length and exposure "
            f"concentration, ρ = {arho:.2f}).")
    para(c, body, M, y, col, "body")
    h = image(c, "fig7_activity_groups", M + col + 18, y + 4, col, max_h=175)
    y -= max(h, 120) + 4
    para(c, (f"History (clicks from before the logs) comes ahead of the logged exposure ({period()}), but it carries no "
             "timestamps, so we can't claim past clicks <i>caused</i> what people saw later. Every logged impression contains a "
             "click, and clicks lean toward people's interests; that's why we measured the tilt on articles shown but <i>not</i> "
             "clicked, the cautious option. “Effective categories” is 1 ÷ the median chance that two random items share a "
             "category. Groups are based on behaviour only, never demographics."),
         M, y, CW, "note")
    frame(c, 3)


def page4(c) -> None:
    sel = csv("recommender_mmr_selected.csv")
    sm = csv("recommender_metrics.csv")
    setup = csv("recommender_setup.csv")
    if sel is None or sm is None:
        y = header(c, "Relevance vs. diversity: a simulation", "Can a feed be relevant and diverse?", "")
        pending(c, M, y, CW, 200, "Notebook 03 results not found")
        frame(c, 4); return
    base = sm[sm["lam"] == 1.0].set_index("model")
    best = base.drop(index="random")["ndcg10"].idxmax()
    pick = sel.set_index("base").loc[best]
    names = {"content": "topic-match", "category": "favourite-category", "popularity": "most-clicked"}
    second = base.drop(index=["random", best])["ndcg10"].idxmax()
    overlap = base.loc[second, "ndcg10_ci_high"] >= base.loc[best, "ndcg10_ci_low"]
    margin = (f", narrowly ahead of {names[second]} on ranking accuracy against real clicks, with nDCG@10 "
              f"{base.loc[best, 'ndcg10']:.3f} vs {base.loc[second, 'ndcg10']:.3f} and AUC {base.loc[best, 'auc']:.3f} vs "
              f"{base.loc[second, 'auc']:.3f}" if overlap else "")
    gain, loss = pick["distinct_categories_rel_change"], pick["ndcg10_rel_change"]
    good = pick["lambda_selected"] < 1 and gain >= 0.2
    if good:
        head = "A little diversity costs very little relevance"
    elif pick["lambda_selected"] < 1:
        head = "Cheap diversity was available, but only a little of it"
    else:
        head = "In this simulation, even a little diversity cost relevance"
    dek = (f"We built simple recommenders anyone can inspect, then re-ranked each with <b>Maximal Marginal Relevance</b> "
           f"(MMR), which weighs an article's predicted relevance against how similar it is to what's already in the list. "
           f"For the most accurate one ({names[best]}), turning the dial to λ = {pick['lambda_selected']:g} moved "
           f"the number of categories in a top-10 list by <b>{gain:+.0%}</b> ({pick['distinct_categories_base']:.1f} → "
           f"{pick['distinct_categories_selected']:.1f}) and relevance by <b>{loss:+.1%}</b>.")
    y = header(c, "Relevance vs. diversity: a simulation", head, dek, hero=f"{gain:+.0%}",
               hero_lbl=f"more categories in a top 10 for a {abs(loss):.1%} relevance cost (simulated)")
    wl, wr = CW * 0.62, CW * 0.36
    para(c, "Each dot is one setting of the diversity dial", M, y, wl, "cap")
    para(c, "Do recommenders amplify favourites?", M + wl + CW * 0.02, y, wr, "cap")
    y -= 16
    h1 = image(c, "fig8_frontier", M, y, wl, max_h=182)
    h2 = image(c, "fig9_rec_amplification", M + wl + CW * 0.02, y, wr, max_h=182)
    y -= max(h1, h2) + 10
    amp_c = base.loc["category", "amp_vs_history_mean"]
    amp_t = base.loc["content", "amp_vs_history_mean"]
    seen = setup["dev_candidates_with_train_clicks"].iat[0] if setup is not None else float("nan")
    min_c = int(setup["min_candidates"].iat[0]) if setup is not None else "?"
    col = (CW - 20) / 2
    amp_r = base.loc["random", "amp_vs_history_mean"]
    narrowest = base.drop(index="random")["distinct_categories"].idxmin()
    if narrowest == best and base.loc[best, "amp_vs_history_mean"] > max(1, amp_r):
        verdict = "The most accurate strategy was also the narrowest."
    elif amp_c > 1 and amp_t > 1:
        verdict = "Personalization concentrates."
    elif max(amp_c, amp_t) > 1:
        verdict = "One strategy concentrates."
    else:
        verdict = "Personalization did not concentrate lists."
    left = (f"<b>{verdict}</b> Ranking by past behaviour, the favourite-category strategy fit just "
            f"{base.loc['category', 'distinct_categories']:.1f} categories into a top 10 (a random order of the same articles "
            f"manages {base.loc['random', 'distinct_categories']:.1f}) and stacked the list with each user's favourite at "
            f"{amp_c:.2f}× its share of their own history. Topic-matching came in at {amp_t:.2f}× and random order at "
            f"{amp_r:.2f}×; anything above 1× means the list is <i>narrower</i> than the person's own clicking.")
    if good:
        right = ("<b>What we'd do about it.</b> (1) Platforms should publish exposure audits like pages 1 to 3 as a matter "
                 "of routine: catalogue share against exposure share, how concentrated exposure is, and how far it tilts toward "
                 "past clicks. (2) The diversity dial was cheap and easy to explain in our simulation, so for Phase 2 we want to "
                 "prototype a “breadth” control that users adjust themselves, show them plainly what it costs in relevance, and "
                 "test it with real people.")
    else:
        right = ("<b>What we'd do about it.</b> Platforms should publish exposure audits like pages 1 to 3 as a matter of "
                 "routine. Extra breadth cost noticeable relevance here, so a “breadth” control should let people pick the "
                 "trade-off themselves with the cost in plain view; it's something to prototype and test, because diversity "
                 "clearly isn't free.")
    h1 = para(c, left, M, y, col, "body"); h2 = para(c, right, M + col + 20, y, col, "body")
    y -= max(h1, h2) + 8
    acc = (f" Favourite-category beat topic-match only narrowly (nDCG@10 {base.loc[best, 'ndcg10']:.3f} vs "
           f"{base.loc[second, 'ndcg10']:.3f}; AUC {base.loc[best, 'auc']:.3f} vs {base.loc[second, 'auc']:.3f})." if overlap else "")
    y -= para(c, (f"These are simulated strategies and they don't reproduce MSN's recommender. We tested them on "
             f"{fmt_int(base.loc[best, 'n_impressions'])} real impressions from the last logged day, 15 Nov, each with {min_c} or "
             f"more articles shown, re-ranking only what was actually shown and scoring against real clicks (nDCG@10). The models "
             f"only had information from before that day (pre-period histories plus clicks from the six days prior), and just "
             f"{pct(seen)} of that day's candidate articles had any click in those six days.{acc}"), M, y, CW, "note")
    y -= 12
    conclusion(c, y, base, best, pick)
    frame(c, 4)


def conclusion(c, y_top: float, base, best, pick) -> None:
    """Answer to the central question, assembled from Notebook 02 and 03 results."""
    if "alignment_lift" not in HL:
        return
    lift, p_val = v("alignment_lift"), v("alignment_p")
    he, ee = v("hist_eff_categories_median"), v("exp_eff_categories_median")
    observed = (f"<b>In the real feed, a modest lean.</b> Among articles people were shown but didn't click, they saw "
                f"{lift:.2f}× as much of their past favourite category as a history-blind baseline predicts" if lift > 1 and p_val < 0.05 else
                "<b>In the real feed, no clear lean</b> toward people's past favourite categories")
    observed += (f", yet the feed stayed wider than their clicks ({ee:.1f} vs {he:.1f} effective categories).")
    simulated = (f"<b>In our simulation, the default narrows.</b> The most accurate strategy fit only "
                 f"{base.loc[best, 'distinct_categories']:.1f} categories into a top 10." if base.loc[best, "distinct_categories"]
                 < base.loc["random", "distinct_categories"] else
                 "<b>In our simulation, the most accurate strategy did not narrow lists.</b>")
    fix = (f"<b>Adding breadth is cheap.</b> A diversity dial raised categories per top 10 by "
           f"{pick['distinct_categories_rel_change']:.0%} at a {abs(pick['ndcg10_rel_change']):.1%} relevance cost, in "
           f"simulation." if pick["lambda_selected"] < 1 and
           pick["distinct_categories_rel_change"] >= 0.2 else
           "<b>Adding breadth wasn't cheap</b> in this simulation.")
    h = 0
    p = Paragraph("So does personalized exposure cost diversity?", ST["capw"]); _, h0 = p.wrap(CW - 28, H)
    items = [t.replace("<b>", '<font color="#ffffff"><b>').replace("</b>", "</b></font>") for t in [observed, simulated, fix]]
    w3 = (CW - 28 - 2 * 16) / 3
    hs = [Paragraph(mn(t), ST["bodyw"]).wrap(w3, H)[1] for t in items]
    box_h = h0 + 10 + max(hs) + 24
    c.setFillColor(NAVY); c.roundRect(M, y_top - box_h, CW, box_h, 5, stroke=0, fill=1)
    c.setFillColor(HERO_ORANGE); c.rect(M, y_top - box_h, CW, 3, stroke=0, fill=1)
    p.drawOn(c, M + 14, y_top - 12 - h0)
    for i, t in enumerate(items):
        para(c, t, M + 14 + i * (w3 + 16), y_top - 22 - h0, w3, "bodyw")


def page5(c) -> None:
    from scipy.stats import spearmanr
    cat = csv("category_metrics.csv", index_col=0)
    main = cat[cat["n_articles"] >= 50]
    rob = csv("category_baseline_robustness.csv")
    rank_r = rob["rho"].iat[0] if rob is not None else spearmanr(main["amplification"], main["amplification_vs_pool"]).statistic
    hist_only = 1 - v("share_of_catalogue_ever_shown")
    y = header(c, "Methods &amp; limitations", "How we know, and what we can't conclude",
               "One public dataset, code anyone can rerun, and the places where the evidence stops.") - 4
    col = (CW - 22) / 2
    left = [
        ("Data", f"Microsoft News Dataset (MIND-small) from Microsoft Research: {fmt_int(v('catalogue_articles'))} news "
                 f"articles with categories, plus anonymized logs from {fmt_int(v('unique_users'))} users. That's "
                 f"{fmt_int(v('rows_behaviors_train') + v('rows_behaviors_dev'))} impression logs ({period()}) holding "
                 f"{fmt_int(v('total_impressions'))} article impressions and {fmt_int(v('total_clicks'))} clicks."),
        ("Definitions", "<b>Exposure</b> is an article logged as shown in an impression, a stand-in that doesn't prove "
                        "anyone read it. <b>Engagement</b> is a click. <b>Catalogue share</b> = a category's articles ÷ all "
                        "articles; <b>exposure share</b> = its impressions ÷ all impressions; <b>CTR</b> = clicks ÷ impressions "
                        "within that category."),
        ("Amplification", f"Exposure share ÷ catalogue share, where 1× means proportional. {pct(hist_only)} of catalogue "
                          f"articles only appear in users' older click histories, so we reran it against just the articles shown "
                          f"at least once, and {'the category ranking barely moved' if rank_r >= 0.9 else 'the category ranking shifted'} "
                          f"(rank correlation {rank_r:.2f})."),
        ("Concentration", "Top-1% and top-10% shares plus the Gini coefficient of impressions per article, recalculated for "
                          "each day as well."),
        ("Narrowing", "We used an unbiased Simpson index (the chance two items share a category) alongside entropy, then ran "
                      "a permutation test on articles shown but not clicked, since every logged impression contains a click "
                      "and clicks favour people's interests. Shuffling users' favourite categories within groups of similar "
                      "activity and time period gives the history-blind baseline." + robustness_sentence()),
        ("Recommenders", "Most clicked over the six days before the test day, TF-IDF similarity between titles and the "
                         "user's history, and favourite-category preference, each re-ranked with MMR as λ went from 1 down to 0. "
                         "We scored them on 15 Nov using nDCG@10 and AUC (ranking accuracy against real clicks), distinct "
                         "categories, intra-list diversity, coverage and an amplification ratio. No 15 Nov click ever went "
                         "into a model."),
    ]
    right = [
        "Observational data, so these are associations rather than causes.",
        "One week of US Microsoft News. It says nothing direct about the wider internet, other platforms, Canada or other years.",
        "Users are anonymized. No demographic or location data exist and we didn't infer any; our groups are purely behavioural.",
        "An impression shows an article was logged as displayed, not that anyone saw or read it.",
        "Every logged impression contains a click, which makes these engaged sessions; click-through rates compare fairly "
        "only inside this dataset.",
        "Click histories come from before the logs, but they carry no timestamps and don't update during the logged week.",
        "Observed exposure is the output of MSN's whole operation (editors, layout, recommendation, navigation), not any "
        "single algorithm.",
        "Our recommenders are simulations run on logged candidate lists and don't reproduce MSN's own system.",
        f"We measured diversity with {len(cat)} broad categories, which misses viewpoint and source diversity. Title-word "
        "diversity (ILD) sat near its maximum for every strategy, so we didn't draw conclusions from it.",
        "MIND-small samples 50,000 users per split; it isn't a census of MSN traffic.",
    ]
    yl = y
    for title, text in left:
        yl -= para(c, f"<b>{title}.</b> {text}", M, yl, col, "body") + 5
    xr = M + col + 22
    c.setFillColor(TINT); c.roundRect(xr - 10, yl - 4, col + 10, y - yl + 10, 4, stroke=0, fill=1)
    yr = y - 4
    yr -= para(c, "<b>Limitations</b>", xr, yr, col - 10, "body") + 4
    for t in right:
        yr -= para(c, f"•&nbsp;&nbsp;{t}", xr, yr, col - 14, "small") + 4
    y = min(yl, yr) - 14
    rule(c, y); y -= 12
    refs = (f"<b>Sources.</b> Wu, F. et al. (2020). MIND: A Large-scale Dataset for News Recommendation. <i>Proceedings of ACL 2020</i>, "
            f"3597–3606. Dataset: Microsoft News Dataset (MIND), Microsoft Research, {DATA_URL} (Microsoft Research License "
            f"Terms). Carbonell, J. &amp; Goldstein, J. (1998). The use of MMR, diversity-based reranking for reordering documents "
            f"and producing summaries. <i>SIGIR '98</i>. <b>Reproducibility.</b> Every number traces back to code, notebooks and "
            f"result files at {REPO_URL}. We don't redistribute the dataset itself.")
    para(c, refs, M, y, CW, "small")
    frame(c, 5)


def build() -> Path:
    if not HL:
        raise SystemExit("outputs/results/headline_metrics.json not found — run the notebooks first")
    c = canvas.Canvas(str(OUT), pagesize=letter, initialFontName="SS", initialFontSize=10)
    c.setTitle("Behind the Feed — Does personalized exposure come at the cost of diversity?")
    c.setAuthor("Signal & Noise (Lionael Surya, Margil Parekh)"); c.setSubject("TECHNATION Behind the Feed — Phase 1")
    for page in (page1, page2, page3, page4, page5):
        page(c)
        c.showPage()
    c.save()
    print("wrote", OUT)
    return OUT


if __name__ == "__main__":
    build()
