# Indic Entity Lab

A Streamlit app that finds **Person (PER)**, **Location (LOC)** and
**Organization (ORG)** names in English, Hindi and Marathi text, from typed
sentences or uploaded TXT / DOCX / PDF / CSV files, and compares the
entities of an English sentence with a Hindi/Marathi sentence you provide.
The app does not translate text.

All three languages use a **BiLSTM-CRF trained from scratch** in this
project (no pretrained embeddings or language models). A pretrained
multilingual model (XLM-R) and a fine-tuned BERT are kept only as
reference points for comparison.

The wider research project (parallel data, MT, entity alignment,
anonymisation) is described in **Research components and status** below.
The authoritative handoff record, including what is incomplete and how
to resume it, is **`PROJECT_STATUS.md`**.

## Run

From this folder (Windows):

```
.venv\Scripts\python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\streamlit run app.py
```

Then open http://localhost:8501. The first NER run downloads the model
(about 1 GB) if it is not cached yet.

## Test

```
.venv\Scripts\python tests\test_app.py
.venv\Scripts\python tests\test_demo.py
.venv\Scripts\python tests\test_models.py
.venv\Scripts\python tests\test_languages.py
.venv\Scripts\python tests\test_alignment.py
.venv\Scripts\python tests\test_regressions.py
.venv\Scripts\python tests\test_excel.py
.venv\Scripts\python tests\test_input_source.py
.venv\Scripts\python tests\test_navigation.py
```

- `test_input_source.py` checks that an uploaded document always takes
  priority over typed text (sections 01-04): no merging, no fallback for
  empty or unreadable documents, stale results cleared, manual mode back
  after removing the document.

- `test_excel.py` covers .xlsx and .xls uploads: multiple sheets, an
  empty sheet, blank and missing cells, cell locations, corrupt and empty
  workbooks, NER on spreadsheet sentences, and filling Entity Alignment
  from a bilingual row.

(Use `.venv312\Scripts\python` for the Python 3.12 environment.)

- `test_regressions.py` covers generic-label models, sub-word fragments,
  alias/full-name merging, model provenance, and checks that no
  translation UI remains; mocked model outputs for code behaviour, real
  inference reported separately.

- `test_alignment.py` checks entity alignment: Manya ↔ मान्या, NOIDA ↔
  नोएडा, type mismatches, unmatched and uncertain entities, offsets,
  one-to-one assignment, English–Marathi suffixes, and section 04 in the
  real app.

- `test_languages.py` runs English, Hindi and Marathi through sections
  01-04: language routing, multi-word names, alias review and the
  entity alignment section.

- `test_models.py` checks the trained models (loading, no pretraining,
  language routing, fallback, confidence, alias fallback, UI routing) and
  prints what each model predicts on its own, without aliases.

- `test_app.py` covers the NER functions, aliases, Marathi suffixes,
  confidence, sentence splitting and file reading. It also drives the real
  app with Streamlit's AppTest, including uploads of every file in
  `tests/samples/`.
- `test_demo.py` runs the presentation demo paragraph, the मुंबई suffix
  forms, titled full names and every user-facing error message through the
  real app.

## How it works

1. **Input**: typed text, or text extracted from TXT/DOCX/PDF/CSV
   (UTF-8 and UTF-16) or Excel (.xlsx via openpyxl, .xls via xlrd).
   For Excel, choose a sheet (or all sheets) and a text column (or all
   text columns); row 1 is the header row, and blank or number-only cells
   are skipped. Each sentence keeps its cell (e.g. `Sentences!B3
   (English)`), shown in the preview and as a Location column in the NER
   results.

   **Input source:** an uploaded document always takes priority. While a
   document is uploaded, only its content is processed; typed text is
   ignored (kept, not erased) and never used as a fallback, and an empty or
   unreadable document produces a message instead. Uploading, replacing or
   removing a document clears sentences and results prepared from the
   previous source.

   **Entity Alignment input** is separate from 01 · Input. Its "Input
   Method" selector offers *Enter Sentences Manually* (the two sentence
   fields) or *Upload a Document* (its own uploader: XLSX, XLS, CSV, DOCX,
   PDF, TXT). Only the selected method is processed; typed sentences are
   kept while hidden but never used in document mode, and an empty,
   unreadable or too-short document gives a message instead of a fallback.
   Excel/CSV: choose the sheet, the English and Hindi/Marathi columns and
   the row (one pair per row). DOCX/PDF/TXT: the text is split into
   numbered sentences with their script, and you choose the sentence for
   each language; nothing is assumed to be aligned. The selected pair then
   goes through the normal NER and alignment steps.
2. **Sentences**: split on `. ! ? । ॥` and newlines, but not after titles
   such as Dr./डॉ. or initials such as "A. P. J.".
3. **NER model** (fixed per workflow, shown as static text, no dropdown):
   Monolingual NER Extraction uses the *language-specific trained models*:
   the selected input language's BiLSTM-CRF trained from scratch.
   Bilingual NER Extraction uses the *baseline multilingual model*
   `Davlan/xlm-roberta-base-ner-hrl` for both sentences.
   If a trained model folder is missing, the app falls back to the baseline
   and says so.
4. **Merging**: word pieces ("Pri" + "ya") are joined, and spans cut in the
   middle of a Devanagari syllable are completed ("पुण्य" → "पुण्या").
   Attached Marathi suffixes (-त, -ला, -मध्ये, -च्या …) are never absorbed.
5. **Threshold**: the minimum confidence is applied after merging.
6. **Aliases**: `ENTITY_ALIASES` in `app.py` adds verified spellings the
   model misses (e.g. Geneva: जेनेवा / जिनेवा / जिनेव्हा / जिनिव्हा).
   When the model also finds that span, one entity is kept with the model's
   real confidence and the source "Multilingual NER model + configured
   alias". Alias-only matches show confidence 1.0 and the source
   "Configured alias — review".
7. **Overlaps**: the longer (then more confident) span wins.

## Models

### Final models (used by the app)
| Language | Model | Folder | Parameters |
|---|---|---|---|
| English | BiLSTM-CRF trained from scratch | `models/english_ner_scratch/` | 2,112,764 |
| Hindi | BiLSTM-CRF trained from scratch | `models/hindi_ner/` | 5,396,278 |
| Marathi | BiLSTM-CRF trained from scratch | `models/marathi_ner/` | 7,124,914 |

All three share one architecture and one implementation (`trained_ner.py`,
`training/train_bilstm_crf.py`):

- **Words:** a randomly initialised word embedding (128 dimensions;
  vocabulary built from the training split only).
- **Characters:** a randomly initialised character embedding (30
  dimensions) fed through a CNN (50 filters) and max-pooled.
- **Sequence model:** a 2-layer BiLSTM (128 units per direction,
  dropout 0.2), then a linear layer, then a CRF (Viterbi decoding).
- **Confidence:** the CRF's forward-backward probability of the predicted
  tag, averaged over the entity's words.
- **Training:** Adam (lr 0.001), batch size 32, gradient clipping 5, word
  dropout 0.05, CPU only. The best epoch is chosen on validation F1.

A character-CNN ablation on 20k sentences raised validation F1 by
3.2 points (Hindi) and 5.5 points (Marathi), so it is part of the
embedding layer. Each folder holds `model.pt`, `vocab.json` and
`config.json`, where `config.json` has the labels, settings and training
log.

| Model | Train sentences | Epochs (best) | Training time |
|---|---|---|---|
| Hindi | 100,000 | 8 (7) | 321 min |
| Marathi | 150,000 | 8 (7) | 293 min |
| English | 13,492 | 30 (26) | 73 min |

English has about 10x fewer training tokens, so it was allowed up to 30
epochs (early stopping, patience 5) instead of 8 (patience 2).

### Reference models (not used as final models)
- **English BERT** (`models/english_ner/`): `bert-base-cased` fine-tuned
  on the same English data (3 epochs, AdamW lr 5e-5). This is a
  transfer-learning comparison only.
- **XLM-R baseline**: `Davlan/xlm-roberta-base-ner-hrl`, pretrained and not
  trained here. It is used by the "Baseline multilingual model" option
  (the default) in sections 03 and 04.

### Results on held-out test sets
Pure model output (no aliases), entity-level exact match. Full table:
`training/results/evaluation.md`.

| Model | Language | Precision | Recall | F1 |
|---|---|---|---|---|
| **BiLSTM-CRF (scratch)** | Hindi | 0.789 | 0.805 | **0.797** |
| XLM-R baseline | Hindi | 0.821 | 0.778 | 0.799 |
| **BiLSTM-CRF (scratch)** | Marathi | 0.826 | 0.764 | **0.794** |
| XLM-R baseline | Marathi | 0.843 | 0.748 | 0.792 |
| **BiLSTM-CRF (scratch)** | English | 0.865 | 0.856 | **0.861** |
| BERT fine-tuned (reference) | English | 0.917 | 0.929 | 0.923 |
| XLM-R baseline | English | 0.903 | 0.926 | 0.914 |

The scratch models were trained on data from the same source as their test
sets; XLM-R was not, so the comparison is not perfectly like-for-like.

**English BERT, validation split:** token-level accuracy **99.26%**
(50,979 / 51,360 tokens of `training/data/english/validation.jsonl`, model
`models/english_ner`; `results/ner/english_bert_validation_accuracy.json`).
This is a validation metric, not test accuracy, and not comparable with the
entity-level F1 scores above: token accuracy counts BIO tags per token
(85.7% of validation tokens are `O`), entity F1 scores complete entities.

The **Research results** page of the app (`pages/1_Research_Results.py`,
linked from the sidebar) shows these results and the project status, read
from the same result files.

### Data
| Language | Dataset | Licence | Train / validation / test |
|---|---|---|---|
| Hindi | Naamapadam (`ai4bharat/naamapadam`) | CC0-1.0 | 100,000 (random subset of 985,787) / 13,460 / 864 |
| Marathi | Naamapadam | CC0-1.0 | 150,000 (random subset of 455,248) / 2,300 / 1,060 |
| English | CoNLL-2003 (`tner/conll2003`) | Reuters text: research use only | 13,492 / 3,250 / 3,453 |

- **Naamapadam:** the train and validation labels are machine-projected
  ("silver"); the test sets are human-annotated.
- **CoNLL-2003:** MISC is mapped to O.
- **All datasets:** training sentences that also occur in validation/test
  were removed; the test sets are unchanged.
- Details are in `training/data/<language>/dataset_info.json`.

### Retrain / re-evaluate
```
.venv\Scripts\python training\prepare_data.py
.venv\Scripts\python training\validate_data.py
.venv\Scripts\python training\train_hindi.py
.venv\Scripts\python training\train_marathi.py
.venv\Scripts\python training\train_english_scratch.py
.venv\Scripts\python training\evaluate.py
```
`training\train_english.py` reproduces the BERT reference experiment.

## Research components and status

Status at handoff (2026-10-07). All research jobs are stopped; details,
exact numbers and resume commands are in `PROJECT_STATUS.md`.

| Component | Folder | Status |
|---|---|---|
| NER (3 languages) + XLM-R / IndicNER / BERT comparison + error analysis | `training/`, `results/ner/`, `analysis/` | **Completed** |
| Parallel data: Samanantar subsets en–hi, en–mr (50k / 1k / 1k each, validated, NER overlap removed) | `mt/data/` | **Completed** |
| MT model: Transformer trained from scratch (9.6M parameters) | `mt/model.py`, `mt/train_mt.py` | Implementation **completed**; training **paused** (en→hi 10 / 10 epochs, en→mr 5 / 10 epochs, resumable); evaluation pending |
| MT evaluation: BLEU, chrF, chrF++, entity preservation | `mt/evaluate_mt.py` | Implemented, **not run** — no test results available |
| COMET + IndicTrans2 reference | `mt/colab_indictrans2_comet.py` | Colab script prepared, **not run** |
| Entity-alignment gold sets (Wikidata-verified; test: 358 en–hi, 223 en–mr) | `alignment/data/` | **Completed** |
| Three-signal aligner (transliteration + MT + NER) + evaluation (Accuracy, P@1, Recall@K, MRR) | `alignment/` | Implemented, **not evaluated** |
| Anonymisation (consistent typed placeholders) | `anonymisation/anonymise.py` | **Completed** |
| Anonymisation evaluation, monolingual | `anonymisation/results/` | **Completed** (residual identifier rate en 13.3%, hi 9.7%, mr 14.1%) |
| Anonymisation evaluation, cross-lingual | `anonymisation/evaluate_anonymisation.py --part cross` | **Not run** |
| End-to-end pipeline (NER → alignment → anonymisation → MT → evaluation) | `pipeline/run_pipeline.py` | Implemented, **not run** |
| Consolidated report (reads result files only) | `results/build_report.py` → `results/FINAL_REPORT.md` | Generated with completed results; missing results marked NOT AVAILABLE |

Validation scores logged during the unfinished MT training are not test
results and are not reported as such.

## What the app supports, and its limits

### A. Supported reliably (by the model, tested)
- PER / LOC / ORG in English, e.g. Priya Mehta, Satya Nadella,
  A. P. J. Abdul Kalam, Geneva, Delhi, Mumbai, Microsoft, Infosys.
- Common Hindi and Marathi names, cities and organizations, e.g.
  नरेंद्र मोदी, सचिन तेंडुलकर, दिल्ली, मुंबई, गूगल, भारतीय रिजर्व बैंक.
- Marathi suffixes stay outside the entity. Both मुंबईत and मुंबईमध्ये give
  the entity `मुंबई`, and the sentence is shown exactly as typed. Tested for
  -त, -मध्ये, येथे, -ला, -च्या, -ची, -चे, -वर, -पासून, -पर्यंत.
- Names after titles and initials: Dr./Mr./Prof./डॉ. and "A. P. J." do not
  split the sentence, and the title itself is never shown as an entity.

### B. Supported through configured aliases
- Spellings the model often misses, listed in `ENTITY_ALIASES` in `app.py`:
  - Geneva: जेनेवा, जिनेवा, जिनेव्हा, जिनिव्हा
  - Infosys: इन्फोसिस
  - Microsoft: माइक्रोसॉफ्ट, मायक्रोसॉफ्ट
  - Tata Consultancy Services: Hindi and Marathi spellings
  - one example person name, प्रिया
- Aliases only match whole words. For example, प्रिया is not matched inside
  सुप्रिया or प्रियांका.
- Source column:
  - `Configured alias — review`: found only through the alias list. Shown
    with confidence 1.0, so check it.
  - `Multilingual NER model + configured alias`: the model found it too. The
    model's real confidence is kept.
- The alias list is deliberately small. Add only verified variants.

### C. Model limitations

Trained models (observed in `tests/test_models.py`):
- The English model labels "Infosys" in "Aarav Sharma works at Infosys." as
  LOC instead of ORG (BERT and XLM-R get it right). English ORG is its
  weakest type (test F1 0.794).
- The Marathi model on a mixed Hindi + English sentence ("Geneva और जिनेव्हा
  के बीच बैठक हुई.") returns one long wrong ORG span. That span also hides
  the correct जिनेव्हा alias, because the longer span wins. Choose the
  language that matches the text.
- The Marathi model misses इन्फोसिस in "आरव शर्मा इन्फोसिसमध्ये काम
  करतात."; the alias list supplies it.

Baseline XLM-R (Auto Detect route):
- Some entities are missed in context, e.g. मुंबईत in
  "विराट कोहली मुंबईत राहतो." (found in most other sentences).
- Some full names are returned in two parts, e.g. रतन / टाटा,
  सुंदर / पिचाई.
- Rare Hindi/Marathi spellings of foreign names are often missed unless
  they are in the alias list.
- Inflected stems are shown as written, not converted to the base form.
  पुण्यात gives `पुण्या`, not `पुणे`.
- The model also predicts DATE, but the app deliberately does not show it
  (scope: PER/LOC/ORG). MONEY and EVENT are not supported by the model.
- The former Hindi fallback model `MichaelHuang/muril_base_cased_hindi_ner`
  is no longer configured: its labels are unnamed (LABEL_0 … LABEL_22) with
  no documented mapping to entity types. Any multilingual model ID entered
  in the sidebar is checked the same way and refused if its labels are
  unnamed.
- The baseline XLM-R model can tag only the first piece of a word (e.g.
  "Sentence Pairs" → `Sent`, `Pai` as PER). Spans that start inside a word,
  or end inside a Latin-script word, are discarded as sub-word fragments.
  Devanagari spans may still end inside a word (attached case suffixes).
- A configured alias such as `प्रिया` is extended to a full name only when a
  model span contains it. The baseline gives no such span for `प्रिया मेहता`,
  so it shows only `प्रिया` (for review); the Hindi BiLSTM-CRF
  ("Language-specific trained models") finds the full name.

### D. PDF limitations
- English text in PDFs is extracted correctly.
- Hindi/Marathi text in PDFs is usually damaged by extraction. Ligatures are
  lost and vowel signs are reordered (a PDF font limitation, not an app
  bug). The app shows a warning when this happens. Use DOCX or TXT for
  Hindi/Marathi documents.
- Scanned (image) PDFs contain no text. OCR is not included.

### E. Input language
- Sections 01-03 (Input, Sentence Preview, Entity Extraction) use an
  "Input Language" selector (English, Hindi, Marathi). 04 · Entity
  Alignment has a "Language Pair" selector (English → Hindi, English →
  Marathi, Hindi → English, Marathi → English) that sets the order and
  language of its two sentence fields. Each sentence goes to its own
  language's model; both sentences are entered by the user (nothing is
  translated). For alignment, the English side is always passed first, so
  either input order gives the same matches.
- Neither workflow has a model dropdown. Sections 01-03 show "Language-
  specific trained models" (the input language's BiLSTM-CRF); section 04
  shows "Baseline multilingual model" (XLM-R for both sentences).
- The sentence preview names the model that will be used, shows the
  detected script, and warns if the choice clearly contradicts the text
  (e.g. English selected for Devanagari text).

### F. Entity alignment (section 04)
- After NER on both sentences, `alignment/entity_pairs.py` pairs English
  entities with Hindi/Marathi entities. Score = transliteration similarity
  from `alignment/aligner.py` (Devanagari romanised, also tried with a case
  suffix such as मध्ये removed); entity types must agree.
- Pairs are assigned one-to-one, best score first. **Matched** = same type,
  score ≥ match threshold, no competing candidate above it. **Needs
  review** = score between the review and match thresholds, different
  types, or a competing candidate. Everything else is listed as
  unmatched; no entity is dropped.
- The thresholds (review 0.64, match 0.73) are calibrated by
  `alignment/calibrate_pairs.py` on the Wikidata-verified gold sets
  (validation split) and stored in `alignment/results/pair_calibration.json`.
  On the held-out test pairs, Matched has precision 99.6% (Hindi) / 97.0%
  (Marathi) and recall 76.8% / 86.1%.
- Not covered: names that are translated rather than transliterated
  (India / भारत, Supreme Court / सर्वोच्च न्यायालय) stay unmatched. The
  aligner's MT signal is not used, because translation was removed from the
  app and the MT models are unevaluated.

## Technical notes

- `.streamlit/config.toml` disables Streamlit's file watcher. Without this,
  harmless "No module named 'torchvision'" tracebacks flood the console.
  Restart the app after editing `app.py`.
- The console warning about `fix_mistral_regex` at model load is a false
  alarm for this model. The default tokenizer output matches the model's
  own SentencePiece tokenization, while the "fix" breaks it.
