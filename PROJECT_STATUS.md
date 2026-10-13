# Project status (updated 2026-10-09; handoff 2026-10-07)

Synthetic Parallel Sentence and Named Entity Pair Generation for Indian Languages
(Hindi, Marathi; English as the controlled/reference language).

All research jobs were **stopped** at handoff. Nothing is running in the
background. Every number below comes from a result file produced by an actual
run; anything not run is marked as such. Validation scores logged during MT
training are **not** test results and are not reported as such.

## Status overview

| Component | Status |
|---|---|
| Dataset preparation + validation (NER, MT) | **COMPLETED** |
| NER training + evaluation (Hindi, Marathi, English) | **COMPLETED** |
| NER benchmarking (XLM-R, IndicNER, English BERT) | **COMPLETED** |
| NER error analysis | **COMPLETED** |
| English BERT token-level validation accuracy | **COMPLETED** — 99.26% (validation split) |
| MT model implementation (`mt/`) | **COMPLETED** |
| MT training English→Hindi | **PAUSED** — 10 of 10 planned epochs completed; evaluation pending |
| MT training English→Marathi | **PAUSED** — 5 of 10 planned epochs completed (resumable) |
| MT final test evaluation (BLEU / chrF / chrF++ / entity preservation) | **NOT RUN** — test results **NOT AVAILABLE** |
| COMET | **NOT RUN** (cannot be installed on this machine; Colab script prepared) |
| IndicTrans2 reference benchmark | **NOT RUN** (incompatible with transformers 5 here; Colab script prepared) |
| Entity-alignment gold sets | **COMPLETED** |
| Entity-alignment method (three-signal aligner) | **IMPLEMENTED**, **NOT EVALUATED** (needs the final MT models) |
| Anonymisation implementation | **COMPLETED** |
| Anonymisation evaluation, monolingual | **COMPLETED** |
| Anonymisation evaluation, cross-lingual | **NOT RUN** |
| End-to-end pipeline script | **IMPLEMENTED**, **NOT RUN** |

## NER (completed)

Model: BiLSTM-CRF **trained from scratch** (random word embeddings + character
CNN → 2-layer BiLSTM, 128/direction → linear → CRF; no pretrained weights).
Entity-level exact-match scores on held-out test sets, no aliases.

| Language | Dataset (test) | Precision | Recall | F1 | PER | ORG | LOC | Model |
|---|---|---|---|---|---|---|---|---|
| Hindi | Naamapadam hi test (864 sent., human-annotated) | 0.789 | 0.805 | **0.797** | 0.842 | 0.729 | 0.798 | `models/hindi_ner/` (5,396,278 params) |
| Marathi | Naamapadam mr test (1,060 sent., human-annotated) | 0.826 | 0.764 | **0.794** | 0.866 | 0.659 | 0.794 | `models/marathi_ner/` (7,124,914 params) |
| English | CoNLL-2003 test (3,453 sent.) | 0.865 | 0.856 | **0.861** | 0.893 | 0.794 | 0.898 | `models/english_ner_scratch/` (2,112,764 params) |
| English (reference) | CoNLL-2003 test | 0.917 | 0.929 | **0.923** | 0.957 | 0.883 | 0.931 | `models/english_ner/` — bert-base-cased fine-tuned, reference experiment only |

Benchmarks (same test sets and scorer):

| Language | Ours F1 | XLM-R F1 | IndicNER F1 | Note |
|---|---|---|---|---|
| Hindi | 0.797 | 0.799 | 0.828 | IndicNER gap −0.031, significant (paired-bootstrap 95% CI −0.047 … −0.016) |
| Marathi | 0.794 | 0.792 | 0.805 | IndicNER gap −0.011, not significant (CI −0.026 … +0.005) |
| English | 0.861 | 0.914 | — | English BERT (ours, fine-tuned): 0.923 |

English BERT on the **validation** split (`training/data/english/validation.jsonl`,
`models/english_ner`): token-level accuracy **99.26%** (50,979 / 51,360 tokens;
`results/ner/english_bert_validation_accuracy.json`). This is a validation
metric, not test accuracy, and is not comparable with the entity-level F1
scores above (85.7% of validation tokens are `O`).

Comparability: XLM-R (`Davlan/xlm-roberta-base-ner-hrl`) and IndicNER
(`ai4bharat/IndicNER` public checkpoint, mBERT-uncased fine-tuned — not the
model reported in its paper) are pretrained; IndicNER may have seen the
Naamapadam test sentences in training. Files: `training/results/evaluation.json`,
`results/ner/indicner_benchmark.json`.

Error analysis (`results/ner/error_analysis_summary.txt`, `results/ner/error_categories.txt`):
unseen words lower recall sharply (Marathi 80.5% → 59.2%); entities of 3+ words
are hardest (recall 54.7–76.6%); ORG is the weakest type in every language;
Hindi/Marathi errors are mainly boundaries and misses, English errors mainly
wrong type; Marathi attached case suffixes; some gold labels look wrong.

## Data (completed)

| Data | Source / licence | Splits | Validation |
|---|---|---|---|
| Hindi NER | Naamapadam (CC0) | 100,000 / 13,460 / 864 | 0 problems, 0 overlap (`training/validate_data.py`) |
| Marathi NER | Naamapadam (CC0) | 150,000 / 2,300 / 1,060 | 0 problems, 0 overlap |
| English NER | CoNLL-2003 (Reuters text, research use) | 13,492 / 3,250 / 3,453 | 0 problems, 0 overlap |
| en–hi parallel | Samanantar subset (CC-BY-NC-4.0), 65 sampled row groups | 50,000 / 1,000 / 1,000 | 0 problems (`mt/validate_parallel.py`); 841 pairs overlapping NER data removed |
| en–mr parallel | Samanantar subset, 68 sampled row groups | 50,000 / 1,000 / 1,000 | 0 problems; 2,659 pairs overlapping NER data removed |

## MT (implementation complete, training paused / evaluation pending)

Model (`mt/model.py`): Transformer encoder–decoder trained from scratch,
3+3 layers, d_model 256, 4 heads, FFN 1024, dropout 0.3, SentencePiece 8k
vocabularies per side (training split only), 9,626,624 parameters; Adam,
peak LR 7e-4, batch 64 sentences, label smoothing 0.1, seed 42.

| Pair | Epochs completed | Best checkpoint (by validation loss) | Final test results |
|---|---|---|---|
| en→hi | 10 of 10 | epoch 10, validation loss 4.224 | **NOT AVAILABLE** (evaluation not run) |
| en→mr | 5 of max 10 | epoch 5, validation loss 4.961 | **NOT AVAILABLE** |

BLEU / chrF / chrF++ / entity preservation: **not available**. COMET: **not run**.
IndicTrans2: **not run**. Checkpoints: `models/mt_en_hi/`, `models/mt_en_mr/`
(`model.pt` = best so far, `last_checkpoint.pt` = full resume state).

## Entity alignment (gold sets complete, evaluation not run)

Gold sets (`alignment/data/`): built from the held-out MT validation/test
splits; English NER mention → English Wikipedia article → its official
Wikidata Hindi/Marathi label must occur in the target sentence.

| Pair | Validation items | Test items |
|---|---|---|
| en–hi | 381 | **358** |
| en–mr | 189 | **223** |

Manual spot check: 25 of 28 sampled items clearly correct. Method
(`alignment/aligner.py`): ranks 1–4-word target spans by transliteration
similarity, similarity to the MT model's translation, and target-NER agreement
(weights tuned on validation). **Accuracy / P@1 / Recall@K / MRR: not
available** (evaluation requires the final MT models).

## Anonymisation (implementation + monolingual evaluation complete)

`anonymisation/anonymise.py`: NER entities (PER/LOC/ORG) + regex identifiers
(EMAIL, URL, PHONE, AADHAAR, PAN) replaced by typed placeholders (`[PER_1]`…);
Hindi/Marathi case suffixes kept outside the placeholder; one consistent
placeholder per entity; aligned cross-lingual entities share a placeholder.

| Lang | Residual identifier rate | Missed | False-positive rate | Wrong type | Non-entity preserved | Identity violations | Synthetic ID recall |
|---|---|---|---|---|---|---|---|
| English | 13.3% | 13.2% | 10.7% | 13.9% | 98.1% | 0 | 100% |
| Hindi | 9.7% | 6.8% | 13.2% | 4.5% | 97.2% | 0 | 100% |
| Marathi | 14.1% | 12.0% | 8.2% | 3.9% | 97.8% | 0 | 100% |

Negative controls (years, amounts) wrongly anonymised: 0/3 per language.
Cross-lingual consistency: **not run**. Results: `anonymisation/results/anonymisation_mono.{json,md}`.

## Pipeline

NER → alignment → anonymisation → MT → evaluation (`pipeline/run_pipeline.py`):

| Stage | Status |
|---|---|
| NER | complete, evaluated |
| Alignment | implemented, not evaluated |
| Anonymisation | implemented, monolingual evaluation complete |
| MT | implemented, training incomplete |
| Evaluation (MT metrics, entity preservation) | implemented, not run |
| End-to-end run | **not run** |

## Tests (last runs)

| Suite | Result |
|---|---|
| `tests/test_app.py` | 143 / 143 passed |
| `tests/test_demo.py` | 51 / 51 passed |
| `tests/test_models.py` | 61 / 63 — 2 genuine model errors (English "Infosys" → LOC; Marathi model on a mixed Hindi/English sentence) |
| `tests/test_mt.py`, `tests/test_anonymisation.py` | not run (need MT / pipeline results) |

## How to resume the remaining work

Keep the PC plugged in and prevent sleep (two epochs took ~13 h overnight
because the processes were throttled). From the project folder:

```
.venv\Scripts\python mt\train_mt.py --lang mr --resume
.venv\Scripts\python mt\evaluate_mt.py
.venv\Scripts\python alignment\evaluate_alignment.py
.venv\Scripts\python anonymisation\evaluate_anonymisation.py --part cross
.venv\Scripts\python pipeline\run_pipeline.py
.venv\Scripts\python tests\test_mt.py
.venv\Scripts\python tests\test_anonymisation.py
.venv\Scripts\python results\build_report.py
```

en→hi has finished its 10 planned epochs. `--resume` continues en→mr from
`last_checkpoint.pt` (epoch 5) with the same
optimiser state, schedule and data order. IndicTrans2 + COMET: after
`mt\evaluate_mt.py`, run `mt/colab_indictrans2_comet.py` in Google Colab
(instructions in its docstring) and place `indictrans2_comet_results.json` in
`mt/results/external/` before `results\build_report.py`.
