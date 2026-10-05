# CLAUDE.md — Working rules for this repository

Project: TECHNATION Canada "Behind the Feed" challenge, Phase 1 (deadline 2026-10-18, 11:59 PM ET).
Deliverable: a ≤5-page PDF visual story built on the Microsoft News Dataset (MIND, https://msnews.github.io/).

Central question: **Does personalized information exposure come at the cost of diversity?**
1. Amplification — which categories get more exposure than their catalogue presence?
2. Narrowing — is more engagement/history *associated with* narrower exposure?
3. Relevance vs. diversity — can a re-ranker recover diversity at small relevance cost?

Do not assume the answer to any question. The data decides.

## Mandatory methodological rules

1. **Never fabricate findings.** Every number, chart, or claim must come from code executed against
   the real MIND files. If an analysis isn't supported by the data, change the analysis. Synthetic
   fixtures under `tests/` exist only to test code paths — never report numbers from them, never write
   them to `outputs/`.
2. **Separate observed from simulated.**
   - Observed system = MIND `behaviors.tsv` impression logs (evidence about logged MSN exposure).
   - Simulated recommenders (Notebook 03) = our experiments. Always label them "simulated recommendation
     strategies". Never claim they reproduce MSN's proprietary algorithm.
3. **No causal claims** from observational data. Use "is associated with", "is consistent with",
   "our simulated recommender demonstrates". Causal language only with a valid experimental design.
4. **No demographic inference.** Users are anonymized. Never infer age, gender, race, income, politics,
   ethnicity, or user location. Segment only by observable behaviour and call them *behavioral groups*.
5. **Geography precision.** MIND is US Microsoft News data. Do not generalise to all internet users,
   Canadians, or global news. No user-geographic analysis. Article entities that are places are *article*
   geography, not user geography.

## Definitions (keep consistent everywhere)

- **Impression / exposure record** = one article shown within one logged impression. A proxy for
  recommendation exposure — *not* proof of attention or reading.
- **Click** = engagement signal (label 1 in the impression list).
- **Catalogue share** = articles in category / unique articles in the union of train+dev `news.tsv`.
  Caveat: `news.tsv` also contains articles that appear only in user click histories (older content),
  so a second baseline, **exposed-pool share** (unique articles shown ≥1 time), is always reported alongside.
- **Exposure share** = impressions in category / all impressions.
- **Click share** = clicks in category / all clicks. **CTR** = clicks in category / impressions in category.
  Never confuse the two.
- **Exposure amplification** = exposure share / catalogue share (1 = proportional). Neutral term — do not
  call it "bias".
- Impression IDs restart per split; the global key is `imp_key = "<split>-<impression_id>"`.
- The order of articles inside an impression list is not documented as display rank; don't treat it as one.

## Data leakage (Notebook 03)

- Build every recommender (popularity counts, TF-IDF profiles, category preferences) from **train** only.
- Evaluate on **dev** impressions; candidates = the articles shown in each dev impression, positives =
  clicked, negatives = shown-but-not-clicked. Dev clicks never feed a model.
- Document split logic in the notebook.

## Reporting standard

Every headline statistic is saved (via `src.metrics.HeadlineRegistry`) with: metric name, value,
denominator, population, filters, source notebook, calculation date → `outputs/results/headline_metrics.json`.
Tables → `outputs/results/*.csv`. Figures → `outputs/figures/`.

When reporting to the team, separate **FACT** / **INTERPRETATION** / **LIMITATION**.

## Engineering conventions

- Reusable logic lives in `src/` (`data.py`, `metrics.py`, `diversity.py`, `recommenders.py`); notebooks call it.
- Notebook sections: Purpose → Data loading → Validation → Analysis → Visualization → Interpretation → Limitations.
  Interpretation cells must only describe numbers printed by the notebook's own executed cells.
- Deterministic seeds (`SEED = 42`) wherever randomness appears.
- Never commit MIND data (`data/raw/`, `data/interim/` are git-ignored).
- Notebook 04 consumes saved results; it does not recompute analyses.
- Workflow gate: do not start a notebook until the previous one has run end-to-end on real data and its
  results have been reviewed.
