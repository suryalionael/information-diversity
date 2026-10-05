"""Code-path tests on a tiny SYNTHETIC MIND-format fixture.

These numbers are fabricated test inputs, never findings. Results go to a temp
directory, never to outputs/. Run: python -m tests.test_pipeline
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CATS = {"news": ["newsus", "newspolitics"], "sports": ["football_nfl"], "lifestyle": ["lifestyleroyals"],
        "finance": ["markets"], "kids": ["kidsfun"]}


def write_fixture(root: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    ids = [f"N{i}" for i in range(1, 121)]
    for split, days in [("train", range(9, 14)), ("dev", [15])]:
        d = root / f"MINDsmall_{split}"
        d.mkdir(parents=True)
        with open(d / "news.tsv", "w") as f:
            for i, nid in enumerate(ids):
                cat = rng.choice(list(CATS))
                title = 'Fixture "quote' if i == 0 else f"Title {nid}"  # unbalanced quote on purpose
                f.write("\t".join([nid, cat, rng.choice(CATS[cat]), title, "" if i % 7 == 0 else "abs",
                                   "https://example.invalid", "[]", "[]"]) + "\n")
        with open(d / "behaviors.tsv", "w") as f:
            for imp in range(1, 61):
                user = f"U{rng.randint(1, 15)}"
                hist = "" if user == "U1" else " ".join(rng.sample(ids[:40], 5) + (["N9999"] if user == "U2" else []))
                shown = rng.sample(ids[20:], 8)
                labels = [1] + [0] * 7
                rng.shuffle(labels)
                toks = " ".join(f"{n}-{l}" for n, l in zip(shown, labels))
                f.write(f"{imp}\t{user}\t11/{rng.choice(list(days))}/2019 {rng.randint(1,12)}:05:58 AM\t{hist}\t{toks}\n")


def test_metrics() -> None:
    from src.diversity import normalized_entropy, shannon_entropy
    from src.metrics import gini, top_share
    assert abs(gini(np.ones(10))) < 1e-12
    assert abs(gini(np.r_[np.zeros(99), 1.0]) - 0.99) < 1e-12
    assert top_share(np.r_[np.ones(99), 101.0], 0.01)["share_of_total"] == 0.505
    assert abs(shannon_entropy([1, 1]) - np.log(2)) < 1e-12
    assert abs(normalized_entropy([5, 5, 5]) - 1.0) < 1e-12


def test_notebook_01() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        write_fixture(tmp / "raw")
        env = {**os.environ, "MIND_DATA_DIR": str(tmp / "raw"), "MIND_OUTPUT_DIR": str(tmp / "out")}
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--output-dir", str(tmp), str(ROOT / "notebooks" / "01_big_picture.ipynb")],
                       check=True, env=env, cwd=ROOT / "notebooks")
        for f in ["headline_metrics.json", "category_metrics.csv", "concentration_metrics.csv"]:
            assert (tmp / "out" / "results" / f).exists(), f


if __name__ == "__main__":
    test_metrics()
    test_notebook_01()
    print("all tests passed (synthetic fixture)")
