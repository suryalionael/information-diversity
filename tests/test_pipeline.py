"""Tests: metric unit tests + end-to-end notebook execution on a SYNTHETIC fixture.

Fixture numbers are fabricated test inputs, never findings. Notebook outputs go to a
temporary directory, never to outputs/. Run: python -m tests.test_pipeline [--keep DIR]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.fixture import write_fixture  # noqa: E402

NOTEBOOKS = ["01_big_picture", "02_narrowing", "03_relevance_diversity", "04_final_charts"]


# --------------------------------------------------------------------------- #
# Unit tests
# --------------------------------------------------------------------------- #
def test_metrics() -> None:
    from src.metrics import bottom_share, category_table, gini, items_for_share, top_share, wilson_interval
    assert abs(gini(np.ones(10))) < 1e-12
    assert abs(gini(np.r_[np.zeros(99), 1.0]) - 0.99) < 1e-12
    assert top_share(np.r_[np.ones(99), 101.0], 0.01)["share_of_total"] == 0.505
    assert abs(bottom_share(np.arange(1, 11), 0.5)["share_of_total"] - 15 / 55) < 1e-12
    assert items_for_share(np.array([50, 30, 20.]), 0.5)["n_items"] == 1
    lo, hi = wilson_interval(np.array([5, 0]), np.array([100, 0]))
    assert 0 < lo[0] < 0.05 < hi[0] and np.isnan(lo[1])
    cat = pd.DataFrame({"news_id": list("abcd"), "category": ["x", "x", "y", "z"]})
    exp = pd.DataFrame({"news_id": list("aaab"), "category": ["x"] * 4, "clicked": [1, 0, 0, 1]})
    t = category_table(cat, exp)
    assert t.loc["x", "amplification"] == 1 / 0.5 and t.loc["x", "ctr"] == 0.5
    assert t.loc["y", "impressions"] == 0 and np.isnan(t.loc["y", "ctr"])
    assert t.loc["x", "pool_share"] == 1.0  # only x articles were shown


def test_diversity() -> None:
    from src.diversity import (normalized_entropy, permutation_alignment, profile_metrics,
                               shannon_entropy, simpson_unbiased)
    assert abs(shannon_entropy([1, 1]) - np.log(2)) < 1e-12
    assert abs(normalized_entropy([5, 5, 5]) - 1.0) < 1e-12
    assert np.isnan(simpson_unbiased([1])) and simpson_unbiased([2, 2, 2]) == 0.2
    # unbiasedness: mean of Simpson over small samples ≈ population HHI
    rng = np.random.default_rng(0)
    p = np.array([.6, .3, .1])
    est = [simpson_unbiased(np.bincount(rng.choice(3, 4, p=p), minlength=3)) for _ in range(20000)]
    assert abs(np.mean(est) - (p**2).sum()) < 0.01
    pm = profile_metrics(pd.DataFrame([[3, 1, 0], [0, 0, 0]], columns=list("abc")), 3)
    assert pm.loc[0, "dominant_category"] == "a" and pm.loc[0, "dominant_share"] == 0.75
    assert pd.isna(pm.loc[1, "dominant_category"]) and np.isnan(pm.loc[1, "simpson"])
    n, K = 2000, 4
    dom = rng.integers(0, K, n)
    sh = rng.dirichlet(np.ones(K), n)
    sh[np.arange(n), dom] += 1
    sh /= sh.sum(1, keepdims=True)
    res = permutation_alignment(sh, dom, np.zeros(n), n_perm=50)
    assert res["lift"] > 1.3 and res["p_one_sided"] < 0.05


def test_recommenders() -> None:
    from src.recommenders import auc_score, mmr_rerank, mrr_score, ndcg_at_k
    assert ndcg_at_k(np.array([1, 0, 0]), np.array([3., 2, 1]), 10) == 1.0
    assert abs(ndcg_at_k(np.array([0, 1]), np.array([2., 1]), 10) - 1 / np.log2(3)) < 1e-12
    assert auc_score(np.array([1, 0, 0]), np.array([3., 2, 1])) == 1.0
    assert auc_score(np.array([1, 0]), np.array([1., 1])) == 0.5          # ties count half
    assert np.isnan(auc_score(np.array([1, 1]), np.array([1., 2])))
    assert mrr_score(np.array([0, 1]), np.array([2., 1])) == 0.5
    from src.recommenders import break_ties, list_diversity, rank_metrics, recall_at_k
    rng = np.random.default_rng(3)
    for _ in range(300):                                    # fast path == reference implementations
        n = rng.integers(2, 60)
        lab = (rng.random(n) < 0.15).astype(float)
        if lab.sum() == 0:
            lab[rng.integers(n)] = 1
        sc = break_ties(rng.integers(0, 5, n).astype(float), rng)
        ref = (ndcg_at_k(lab, sc, 10), auc_score(lab, sc), mrr_score(lab, sc), recall_at_k(lab, sc, 10))
        assert np.allclose(rank_metrics(lab, sc, 10), ref, equal_nan=True), (rank_metrics(lab, sc, 10), ref)
    ts = np.array([[1, .2, .4], [.2, 1, 0], [.4, 0, 1.]])
    d = list_diversity(np.array([0, 1, 2]), np.array([0, 0, 1]), ts)
    assert d["distinct_categories"] == 2 and abs(d["ild"] - (1 - (0.2 + 0.4 + 0) / 3)) < 1e-12
    rel = np.array([1.0, 0.9, 0.1])
    sim = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1.]])
    assert list(mmr_rerank(rel, sim, 1.0, 3)) == [0, 1, 2]
    assert list(mmr_rerank(rel, sim, 0.5, 3)) == [0, 2, 1]                 # redundancy pushes item 1 down


# --------------------------------------------------------------------------- #
# End-to-end notebook execution on the fixture
# --------------------------------------------------------------------------- #
def run_notebooks(workdir: Path, names: list[str] = NOTEBOOKS) -> Path:
    write_fixture(workdir / "raw")
    env = {**os.environ, "MIND_DATA_DIR": str(workdir / "raw"), "MIND_OUTPUT_DIR": str(workdir / "out")}
    for name in names:
        nb = ROOT / "notebooks" / f"{name}.ipynb"
        if not nb.exists():
            print("skip (not built):", name)
            continue
        subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                        "--ExecutePreprocessor.timeout=1800", "--output-dir", str(workdir / "executed"), str(nb)],
                       check=True, env=env, cwd=ROOT / "notebooks", capture_output=True)
        print("executed:", name)
    subprocess.run([sys.executable, "report/phase1/build_report.py"], check=True, cwd=ROOT, capture_output=True,
                   env={**env, "REPORT_PDF": str(workdir / "report_fixture.pdf")})
    print("built report (fixture):", workdir / "report_fixture.pdf")
    return workdir / "out"


if __name__ == "__main__":
    test_metrics(); test_diversity()
    try:
        test_recommenders()
    except ImportError:
        print("skip recommender unit tests (not implemented yet)")
    keep = sys.argv[sys.argv.index("--keep") + 1] if "--keep" in sys.argv else None
    if keep:
        out = run_notebooks(Path(keep))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            out = run_notebooks(Path(tmp))
            assert (out / "results" / "headline_metrics.json").exists()
            assert (Path(tmp) / "report_fixture.pdf").exists()
    print("all tests passed (synthetic fixture)")
