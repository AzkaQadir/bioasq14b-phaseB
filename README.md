# No Retrieval, No Fine-Tuning: A Prompt-Only Mistral-7B System for BioASQ 14b Phase B

A lightweight Phase B system for [BioASQ 14 Task B](http://bioasq.org/) (CLEF 2026). It answers
biomedical questions using **Mistral-7B-Instruct-v0.3 in 4-bit precision on a single T4 GPU**,
prompted with the gold snippets supplied by the challenge organisers. No retrieval pipeline,
no fine-tuning, no proprietary models.

Everything lives in one notebook: `bioasq14b_phaseB.ipynb`.

## What it does

Given a question, its type label, and gold PubMed snippets, the pipeline produces an **exact
answer** and an **ideal answer** for each of the four BioASQ question types:

| Type | Exact answer | Ideal answer |
|---|---|---|
| yesno | `"yes"` / `"no"` | ≤200-word paragraph |
| factoid | up to 5 ranked entities | ≤200-word paragraph |
| list | up to 100 entities | ≤200-word paragraph |
| summary | *(key omitted)* | ≤200-word paragraph |

Four stages: context construction → model loading → type-specific answer heads →
schema validation. Submissions that fail validation are never written to disk.

## Results

Evaluated on a **200-question held-out split of the training set** (52 yesno, 56 factoid,
48 list, 44 summary), using the official BioASQ D4.1 metrics.

| Type | Official metric | Primary | Few-shot ablation |
|---|---|---:|---:|
| yesno | macro-F1 | **0.9549** | 0.9063 |
| factoid | MRR | **0.4113** | 0.3765 |
| list | mean F1 | 0.3427 | **0.3750** |
| ideal | ROUGE-2 | **0.2565** | 0.2507 |
| ideal | ROUGE-SU4 | **0.2517** | 0.2451 |

> **These are holdout numbers, not official leaderboard scores.** They are computed on questions
> carved out of the BioASQ 14b *training* set using a local reimplementation of the official
> metrics. They are **not** comparable to scores reported on the official test batches, which
> are a different question set. Please do not cite them as leaderboard results.

**Few-shot helped list questions only** (+0.032 F1) and hurt everything else. The differences
are measured on ~50 questions per type, so the smaller deltas — particularly the ROUGE ones —
are within noise. Treat the ablation as suggestive, not conclusive.

## Primary vs. few-shot

Controlled by one flag in the `Config` cell:

```python
USE_FEW_SHOT = False   # primary system (default) — reproduces the main results
USE_FEW_SHOT = True    # TF-IDF few-shot ablation
```

The few-shot variant builds a per-type TF-IDF index over the training questions (excluding the
holdout) and prepends the top-N most cosine-similar solved examples to the prompt: 2 for
yesno/ideal, 3 for factoid/list.

*Note:* the four test batches submitted to the portal were produced with `USE_FEW_SHOT = True`,
since that was the configuration running when the test data arrived. The primary system is the
stronger one and is what the default reproduces.

## Getting the data

**The BioASQ dataset is not in this repository and cannot be redistributed.** Obtain it yourself:

1. Register at [participants-area.bioasq.org](https://participants-area.bioasq.org)
2. Add a system under *Edit Profile* (submissions without a registered system fail silently)
3. Download `training14b.json` (5,729 questions) and the Phase B test sets
4. Point `TRAIN_PATH` / `RUNS` in the `Config` cell at your local copies

Default paths assume a Kaggle dataset layout; change them for any other environment.

## Requirements

Runs on a free Kaggle T4 (16 GB VRAM). Installed by the first cell:

```
transformers  accelerate  bitsandbytes  nltk  scikit-learn  torch
```

Roughly 1–2 hours per 200 questions on a T4 at 4-bit.

## Notes on the implementation

- **ROUGE-SU4 from scratch.** Skip-bigrams with a max gap of 4, plus unigrams. The standard
  `rouge-score` package doesn't support SU4. Self-tested: identical → 1.0, disjoint → 0.0,
  gap-4 included, gap-5 excluded. Multi-reference gold ideal answers are scored by max F1.
- **Post-processing matters.** yesno falls back to `"yes"` when no valid token is parsed;
  factoid/list candidates are cleaned, deduplicated, and capped. These rules affect the
  reported metrics and are not yet instrumented — see below.
- **Known caveat:** the attention mask is not explicitly passed and pad token == eos token,
  which `transformers` warns may cause unexpected behaviour. Not yet investigated.

## Known limitations

- Post-processing fallback counts are unreported — the share of the yesno score attributable
  to the default-to-yes rule is currently unknown.
- Generation hyperparameters are partially unspecified (greedy on exact-answer heads;
  `temperature` is passed but ignored under greedy decoding).
- TF-IDF similarity is surface-level and does not guarantee retrieved examples share the
  answer format or entity type of the target question.

## Citation

The accompanying paper is a CLEF 2026 BioASQ working note (not accepted; preprint pending).
Until then, cite this repository directly.

## Licence

Code: MIT (see `LICENSE`). The BioASQ dataset is covered by the organisers' own terms and is
not included here.
