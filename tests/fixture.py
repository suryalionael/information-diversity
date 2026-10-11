"""Synthetic MIND-format fixture for code-path testing ONLY.

Every value produced here is invented. It mimics the *structure* of MIND small
(schema, ≥1 click per impression, per-user histories that never change across
impressions or splits, dev users mostly distinct from train users, a few tiny
categories) so notebooks can be executed end-to-end without the real data.
Never report numbers derived from this fixture.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CATEGORIES = {  # name: (relative catalogue weight, subcategories)
    "news": (30, ["newsus", "newspolitics", "newsworld"]), "sports": (28, ["football_nfl", "tennis"]),
    "finance": (6, ["markets"]), "foodanddrink": (5, ["recipes"]), "travel": (5, ["traveltips"]),
    "lifestyle": (5, ["lifestyleroyals"]), "video": (4, ["viral"]), "weather": (4, ["weathertopstories"]),
    "health": (3, ["medical"]), "autos": (3, ["autosnews"]), "tv": (2, ["tvnews"]),
    "music": (2, ["musicnews"]), "entertainment": (1.5, ["celebrity"]), "movies": (1.5, ["movienews"]),
    "kids": (0.2, ["kidsfun"]), "middleeast": (0.05, ["middleeast"]), "games": (0.05, ["games"]),
}
FILLER = "report update says new week after over could year first people".split()


def write_fixture(root: Path, seed: int = 0, n_articles: int = 900, n_users: int = 500) -> None:
    rng = np.random.default_rng(seed)
    cats = list(CATEGORIES)
    w = np.array([CATEGORIES[c][0] for c in cats]); w = w / w.sum()
    art_cat = rng.choice(len(cats), n_articles, p=w)
    art_cat[: len(cats)] = np.arange(len(cats))  # every category present at least once
    ids = np.array([f"N{i}" for i in range(1, n_articles + 1)])
    titles = []
    for i, c in enumerate(art_cat):
        words = [cats[c]] * 2 + [f"{cats[c]}{rng.integers(5)}"] + list(rng.choice(FILLER, 3))
        rng.shuffle(words)
        titles.append(("Fixture \"quote " if i == 0 else "") + " ".join(words))
    old = np.arange(n_articles) < n_articles * 0.6          # history-only, older content
    pop = rng.pareto(1.2, n_articles) + 0.05                 # heavy-tailed attractiveness
    pop[old] = 0
    pop[art_cat >= len(cats) - 2] = 0                        # two tiny categories never shown

    users = np.array([f"U{u}" for u in range(1, n_users + 1)])
    pref = rng.dirichlet(np.full(len(cats), 0.3), n_users)
    hist_len = np.minimum(rng.geometric(1 / 15, n_users) - 1, 120)
    hist = []
    for u in range(n_users):
        p = pref[u][art_cat] * old; p = p / p.sum()
        hist.append(" ".join(rng.choice(ids, hist_len[u], replace=False, p=p)) if hist_len[u] else "")

    in_train = rng.random(n_users) < 0.75
    in_dev = (~in_train) | (rng.random(n_users) < 0.15)
    for split, days, members in [("train", range(9, 15), in_train), ("dev", [15], in_dev)]:
        d = root / f"MINDsmall_{split}"
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "news.tsv", "w") as f:
            for i in range(n_articles):
                c = cats[art_cat[i]]
                sub = CATEGORIES[c][1][i % len(CATEGORIES[c][1])]
                f.write("\t".join([ids[i], c, sub, titles[i], "" if i % 19 == 0 else "abstract",
                                   "https://example.invalid", "[]", "[]"]) + "\n")
        imp_id = 0
        with open(d / "behaviors.tsv", "w") as f:
            for u in np.flatnonzero(members):
                for _ in range(1 + rng.poisson(2)):
                    imp_id += 1
                    n = int(np.clip(rng.lognormal(3, 0.8), 2, 150))
                    p = pop * (0.5 + 3 * pref[u][art_cat]); p = p / p.sum()
                    shown = rng.choice(n_articles, min(n, int((p > 0).sum())), replace=False, p=p)
                    clk_p = 0.02 + 0.3 * pref[u][art_cat[shown]]
                    labels = (rng.random(len(shown)) < clk_p).astype(int)
                    if labels.sum() == 0:
                        labels[rng.integers(len(shown))] = 1
                    toks = " ".join(f"{ids[a]}-{l}" for a, l in zip(shown, labels))
                    day = rng.choice(list(days))
                    hh = rng.integers(1, 13)
                    f.write(f"{imp_id}\t{users[u]}\t11/{day}/2019 {hh}:{rng.integers(10, 60)}:07 "
                            f"{'AM' if rng.random() < .5 else 'PM'}\t{hist[u]}\t{toks}\n")
