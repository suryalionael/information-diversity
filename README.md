# Information Diversity — Behind the Feed

**Does personalized information exposure come at the cost of diversity?**

Research repository for the TECHNATION Canada *Behind the Feed* challenge (Phase 1: a ≤5-page PDF visual
story, due 18 October 2026, 11:59 PM ET). We use the public Microsoft News Dataset (MIND) to study
patterns of online news exposure and recommendation.

> **Status:** Notebook 01 executed on MIND small; Notebooks 02–04 and the PDF builder are implemented and tested
> on a synthetic fixture, pending execution on the real data. Findings live in `outputs/results/` and the report
> (`report/phase1/behind-the-feed-phase1.pdf`); none are stated here.

## Research questions

1. **Amplification** — Which news categories receive disproportionately high exposure relative to their
   presence in the content catalogue?
2. **Narrowing** — Is greater user engagement/history *associated with* narrower information exposure?
3. **Relevance vs. diversity** — Can a recommender recover information diversity without sacrificing too
   much relevance?

## Dataset

[MIND](https://msnews.github.io/) (Wu et al., ACL 2020): news articles from Microsoft News with metadata, plus
anonymized user behaviour logs collected in the US. We start with **MIND small** (train + dev).

- `news.tsv` — article ID, category, subcategory, title, abstract, URL, entities.
- `behaviors.tsv` — impression ID, anonymized user ID, timestamp, click history, and the list of
  articles shown in the impression with click / no-click labels.

Because `behaviors.tsv` logs what was *shown*, we treat an **impression as a proxy for exposure** and a
**click as an engagement signal**. Impressions are not proof that content was read.

Download instructions: [`data/README.md`](data/README.md). The data is never committed.

## Methodology

| Notebook | Question | Approach |
|---|---|---|
| `01_big_picture` | Amplification | Data validation; catalogue vs. exposure vs. click shares; CTR; exposure amplification; article-level concentration (Lorenz, Gini, top-X%), long tail |
| `02_narrowing` | Narrowing | User-level Shannon entropy of exposure vs. clicks across behavioral activity groups (quantile-based) |
| `03_relevance_diversity` | Trade-off | *Simulated* recommenders (popularity, TF-IDF content, category preference) + MMR re-ranking; relevance–diversity frontier |
| `04_final_charts` | — | Publication charts from saved results |

### Key metrics

- **Exposure amplification** = exposure share / catalogue share (1 = proportional).
- **CTR** = clicks / impressions within a category (distinct from click share).
- **Gini coefficient, Lorenz curve, top-X% share** — article-level exposure concentration.
- **Shannon entropy** (normalized) — category diversity of what a user was shown or clicked.
- **AUC, nDCG@10** — ranking relevance of simulated recommenders on dev impressions.
- **Catalogue coverage, intra-list diversity, amplification ratio** — diversity of recommendations.

The full rules (observed vs. simulated, no causal claims, no demographic inference, leakage control) are in
[`CLAUDE.md`](CLAUDE.md).

## Repository structure

```text
information-diversity/
├── README.md, CLAUDE.md, requirements.txt, .gitignore
├── data/README.md          # how to obtain MIND (data itself is git-ignored)
├── notebooks/              # 01 → 04, run in order
├── run_pipeline.py         # one command: notebooks 01–04 + PDF
├── src/                    # data.py, metrics.py, diversity.py, recommenders.py, style.py
├── tests/                  # unit tests + synthetic fixture (never results)
├── assets/fonts/           # Source Sans 3 (SIL OFL) used by charts and the PDF
├── outputs/figures/        # exploratory (01_…, 02_…, 03_…) and final/ publication charts
├── outputs/results/        # generated metrics (CSV/JSON with provenance)
└── report/phase1/          # build_report.py → behind-the-feed-phase1.pdf
```

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place MIND small under data/raw/ (see data/README.md)
python run_pipeline.py              # Notebooks 01–04 (executed in place) + PDF report
python run_pipeline.py --from 4     # only final charts + PDF, from saved results
```

Tests (unit tests + the whole chain on a tiny *synthetic* fixture, written to a temp dir):
`python -m tests.test_pipeline`. Notebooks are generated from `notebooks/build_0X_*.py`.

## Limitations (apply to all results)

- MIND reflects Microsoft News in the US over a limited period; it does not represent the whole internet,
  Canadian audiences, or global news consumption.
- Users are anonymized; no demographic attributes exist and none are inferred.
- Logs are historical and observational; associations are not causal.
- Impressions are an exposure proxy, not evidence of reading.
- Observed impressions were produced by MSN's own recommendation ecosystem.
- Our recommenders are simulated strategies and do not reproduce MSN's proprietary algorithm.
