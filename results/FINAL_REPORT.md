# Synthetic Parallel Sentence and Named Entity Pair Generation for Indian Languages — Final Research Package

Generated 2026-10-09 17:07 by `results/build_report.py` from the result files on disk. Every number below is read from a result file produced by an actual run; items that were not run or not finished are marked **NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md)**, items that could not be run are marked **NOT RUN** with the reason.

| Component | Status |
|---|---|
| NER | complete |
| MT / generation | training paused / evaluation pending — NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) |
| Entity alignment | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) |
| Anonymisation (monolingual) | complete |
| Anonymisation (cross-lingual) | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) |
| End-to-end pipeline | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) |
| IndicTrans2 reference | NOT RUN — Colab script prepared: mt/colab_indictrans2_comet.py (IndicTrans2's custom code is incompatible with transformers 5 used here) |
| COMET | NOT RUN — would be computed by mt/colab_indictrans2_comet.py in Colab (unbabel-comet cannot be installed here: no NumPy wheel for Python 3.14/ARM64, no C compiler) |

## 1. Named Entity Recognition

Entity-level exact match (type + span), micro-averaged, on held-out test sets; no aliases or post-processing.

| Language | Model | Test set | Sentences | P | R | F1 | PER F1 | ORG F1 | LOC F1 |
|---|---|---|---|---|---|---|---|---|---|
| Hindi | BiLSTM-CRF (scratch) | Naamapadam hi test (human-annotated) | 864 | 0.789 | 0.805 | **0.797** | 0.842 | 0.729 | 0.798 |
| Marathi | BiLSTM-CRF (scratch) | Naamapadam mr test (human-annotated) | 1060 | 0.826 | 0.764 | **0.794** | 0.866 | 0.659 | 0.794 |
| English | BiLSTM-CRF (scratch) | CoNLL-2003 test | 3453 | 0.865 | 0.856 | **0.861** | 0.893 | 0.794 | 0.898 |

### 1.1 Comparison with baselines

| Language | Ours F1 | Baseline | Baseline F1 | Difference | Comparability |
|---|---|---|---|---|---|
| Hindi | 0.797 | XLM-R (Davlan/xlm-roberta-base-ner-hrl, pretrained, not trained here) | 0.799 | -0.001 | same test set & scorer; baseline not trained on Naamapadam |
| Marathi | 0.794 | XLM-R (Davlan/xlm-roberta-base-ner-hrl, pretrained, not trained here) | 0.792 | +0.002 | same test set & scorer; baseline not trained on Naamapadam |
| English | 0.861 | XLM-R (Davlan/xlm-roberta-base-ner-hrl, pretrained, not trained here) | 0.914 | -0.054 | same test set & scorer; baseline not trained on Naamapadam |
| Hindi | 0.797 | IndicNER (ai4bharat/IndicNER public checkpoint, mBERT-uncased fine-tuned) | 0.828 | -0.031 (paired-bootstrap 95% CI -0.047 … -0.016) | same test set & scorer; pretrained; possible test exposure; not the paper's model |
| Marathi | 0.794 | IndicNER (ai4bharat/IndicNER public checkpoint, mBERT-uncased fine-tuned) | 0.805 | -0.011 (paired-bootstrap 95% CI -0.026 … +0.005) | same test set & scorer; pretrained; possible test exposure; not the paper's model |
| English | 0.861 | bert-base-cased fine-tuned in this project (reference experiment) | 0.923 | -0.062 | same data, split & scorer; pretrained vs scratch |

### 1.1a English BERT token-level accuracy (validation split)

Token-level accuracy = share of tokens whose predicted BIO tag matches the reference tag. It is a **validation** metric (not test accuracy) and is not comparable with the entity-level F1 scores above, which score complete entities.

| Model | Checkpoint | Dataset | Correct tokens | Total tokens | Token accuracy |
|---|---|---|---|---|---|
| English BERT (reference) | `models/english_ner` | `training/data/english/validation.jsonl` | 50,979 | 51,360 | **99.26%** |

Validation split, not the held-out test set. Token-level accuracy is not entity-level F1: 44,025 of the 51,360 validation tokens are O, so an all-O tagger would already score 85.7%.

### 1.2 NER error analysis (test sets)

From `results/ner/error_analysis_summary.txt` and `results/ner/error_categories.txt` (scripts `analysis/ner_error_analysis.py`, `analysis/ner_error_categories.py`).

```
######## HINDI  (recomputed F1 0.7972, gold entities 1929)
  correct                       1553  (80.5% of gold)
  right span, wrong type          34  (1.8% of gold)
  right type, wrong boundary     163  (8.4% of gold)
  wrong boundary and type         48  (2.5% of gold)
  missed                         131  (6.8% of gold)
  spurious (no gold overlap)     180
  type confusions (gold -> predicted): {('ORG', 'LOC'): 8, ('LOC', 'ORG'): 8, ('PER', 'ORG'): 6, ('ORG', 'PER'): 5, ('PER', 'LOC'): 4, ('LOC', 'PER'): 3}
  missed by type: {'LOC': 43, 'ORG': 54, 'PER': 34} | spurious by type: {'ORG': 70, 'LOC': 56, 'PER': 54}
  recall, entities all words seen in training: 1295/1572 = 82.4%
  recall, entities with an unseen word: 258/357 = 72.3%
  recall, 1-word entities: 728/899 = 81.0%
  recall, 2-word entities: 630/744 = 84.7%
  recall, 3+-word entities: 195/286 = 68.2%
  e.g. LOC -> ORG: ['राष्ट्रपति भवन', 'यूपी', 'सुमावली', 'मेहगांव', 'दिमनी']
  e.g. ORG -> LOC: ['सिंदरी', 'मुतंगी मस्जिद', 'होली एंजेल्स के आरसी चर्च', 'बायट्सफॉरऑल', 'इंडियन लेंग्वेज रिसोर्सेस सेंटर']
  e.g. ORG -> PER: ['वाजपेयी', 'Der Spiegel', 'परमेश्वर', 'विजिलेंस', 'अन्ना']
  e.g. PER -> ORG: ['जॉनसन', 'ओपीएस भदौरिया', 'साई', 'देवीलाल', 'बलिया']
  e.g. boundary ORG: ['रिलायंस  => predicted: रिलायंस रिलायंस', 'मोजर बीयर पावर  => predicted: बिजली मोजर बीयर पावर', 'कृष्णापत्तनम पावर कॉरपोरेशन लिमिटेड  => predicted: बिजली कृष्णापत्तनम पावर कॉरपोरेशन लिमिटेड', 'केएसके महानदी पावर  => predicted: बिजली केएसके महानदी पावर लिमिटेड', 'कांग्रेस  => predicted: कांग्रेस विधान परिषद']
  e.g. missed ORG: ['इनेलो', 'राष्ट्रीय स्वच्छ विद्यालय', 'केंद्रीय पर्यावरण', 'नेस्डैक', 'यूएससी']
  e.g. spurious LOC: ['भारत', 'निस', 'कुल्लू', 'मुक़ाबले', 'मंडी']
  e.g. spurious ORG: ['सी', 'मेक इन इंडिया', 'स्वच्छ भारत मिशन', 'बीए', 'एनआरसी']
  e.g. spurious PER: ['श्रीमती', 'ईश्वर', 'एंडर', 'Dinka language', 'मसीह']

######## MARATHI  (recomputed F1 0.7939, gold entities 1676)
  correct                       1281  (76.4% of gold)
  right span, wrong type          29  (1.7% of gold)
  right type, wrong boundary     137  (8.2% of gold)
  wrong boundary and type         28  (1.7% of gold)
  missed                         201  (12.0% of gold)
  spurious (no gold overlap)      97
  type confusions (gold -> predicted): {('LOC', 'PER'): 7, ('PER', 'LOC'): 6, ('ORG', 'LOC'): 6, ('ORG', 'PER'): 5, ('LOC', 'ORG'): 3, ('PER', 'ORG'): 2}
  missed by type: {'ORG': 77, 'LOC': 80, 'PER': 44} | spurious by type: {'ORG': 40, 'LOC': 31, 'PER': 26}
  recall, entities all words seen in training: 1091/1355 = 80.5%
  recall, entities with an unseen word: 190/321 = 59.2%
  recall, 1-word entities: 725/957 = 75.8%
  recall, 2-word entities: 480/580 = 82.8%
  recall, 3+-word entities: 76/139 = 54.7%
  e.g. LOC -> ORG: ['संयुक्त राज्यांच्या', 'वांद्रे पूर्वेतून', 'माता वैष्णोदेवी मंदीर']
  e.g. ORG -> LOC: ['तेलंगणात', 'फोर्टिस रुग्णालयात', 'मिलिटरी हॉस्पिटलमध्ये', 'ओशिवरा पोलीस ठाण्यात', 'राष्ट्रपती भवनात']
  e.g. ORG -> PER: ['शिवसेनेची', 'यहोवावर', 'परमेश्वर', 'इफिस', 'यहोवाने']
  e.g. PER -> ORG: ['व्हॉयेजर', 'माक्वी']
  e.g. boundary ORG: ['राष्ट्रवादी काँग्रेस  => predicted: राष्ट्रवादी', 'राष्ट्रवादी काँग्रेसचे  => predicted: राष्ट्रवादी', 'आयुष मंत्रालय  => predicted: आयुष मंत्रालय योग प्रशिक्षणासाठी आयुष मंत्रालय', 'श्रम मंत्रालय  => predicted: अन्न व पुरवठा आणि श्रम मंत्रालय', 'आयएनएस  => predicted: आयएनएस राणा']
  e.g. missed ORG: ['ट्विटर', 'केंद्रीय', 'हरित तंत्रज्ञान', 'कौशल्य विकास आणि उद्यमशीलता', 'दूरदर्शन']
  e.g. spurious LOC: ['भारताच्या', 'बिहान', 'ससून रूग्णालयामध्ये', 'सनत नगर', 'क्रिप्टोकरन्सी']
  e.g. spurious ORG: ['माहिम पोलीस', 'अर्थ आणि नियोजन', 'उर्जा', 'रेल्वे', 'कार्यकारी मंडळ']
  e.g. spurious PER: ['देव', 'यरहमेलची', 'हुरडा', 'मॅनजरसह', 'सुर्यवंशी']

######## ENGLISH  (recomputed F1 0.8605, gold entities 4946)
  correct                       4235  (85.6% of gold)
  right span, wrong type         313  (6.3% of gold)
  right type, wrong boundary      85  (1.7% of gold)
  wrong boundary and type         48  (1.0% of gold)
  missed                         265  (5.4% of gold)
  spurious (no gold overlap)     225
  type confusions (gold -> predicted): {('LOC', 'ORG'): 80, ('PER', 'ORG'): 78, ('ORG', 'LOC'): 57, ('ORG', 'PER'): 56, ('PER', 'LOC'): 28, ('LOC', 'PER'): 14}
  missed by type: {'LOC': 66, 'PER': 75, 'ORG': 124} | spurious by type: {'ORG': 156, 'PER': 29, 'LOC': 40}
  recall, entities all words seen in training: 2469/2723 = 90.7%
  recall, entities with an unseen word: 1766/2223 = 79.4%
  recall, 1-word entities: 2558/3049 = 83.9%
  recall, 2-word entities: 1471/1628 = 90.4%
  recall, 3+-word entities: 206/269 = 76.6%
  e.g. LOC -> ORG: ['West Indies', 'Uzbekistan', 'Wimbledon', 'Syria', 'Japan']
  e.g. ORG -> LOC: ['Santa Fe', 'Zywiec', 'Tasmania', 'Barbarians', 'Newmont']
  e.g. ORG -> PER: ['Santa Fe', 'Vicenza', 'Napoli', 'Norilsk', 'Kinder Bologna']
  e.g. PER -> ORG: ['Mills', 'Clarke', 'Van Boxmeer', 'Bitar', 'Albright']
  e.g. boundary ORG: ['Real Madrid  => predicted: Real Madrid Real Madrid', 'Barcelona  => predicted: Real Madrid Barcelona', 'Real Madrid  => predicted: Real Madrid Barcelona', 'Barcelona  => predicted: Barcelona Barcelona', 'Sporting  => predicted: Sporting Gijon']
  e.g. missed ORG: ['Bre-X', 'Newmont', 'NYMEX', 'League', 'Norilsk']
  e.g. spurious LOC: ['SHEFFIELD', 'Choice', 'PLAYOFF', 'Rose Bowl', 'PAPER']
  e.g. spurious ORG: ['Choice', 'Select', 'PF PA', 'EASTERN CONFERENCE', 'WESTERN CONFERENCE']
  e.g. spurious PER: ['UNCAPPED', 'CLOBBER', 'Lombardi Award', 'Starbuck', 'DEPRIVES']
```

## 2. Machine translation / generation (English → Hindi, English → Marathi)

**NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md)** — MT training paused (en→hi 10 of 10, en→mr 5 of 10 planned epochs; resumable from `models/mt_en_*/last_checkpoint.pt`). `mt/evaluate_mt.py` was not run, so no test BLEU/chrF/chrF++ or entity-preservation results exist.

### 2.2 IndicTrans2 reference (reference-only, NOT a like-for-like comparison)

**NOT RUN.** To run: `mt/colab_indictrans2_comet.py` in Colab on the exported test files (IndicTrans2's custom model and tokenizer code is incompatible with transformers 5 used in this project: it imports the removed `transformers.onnx` and relies on removed model/tokenizer internals). It is a *reference-only* baseline: IndicTrans2 is pretrained on far more data (BPCC, which includes Samanantar, the source of our test sentences). No published IndicTrans2 number is quoted, because none was verified on our test set.

## 3. Cross-lingual entity alignment

Gold sets (Wikidata-verified, built from the held-out MT validation/test splits; `alignment/build_gold.py`):

| Split | English mentions | Gold items | By type |
|---|---|---|---|
| hi/validation | 1211 | 381 | {'LOC': 173, 'ORG': 86, 'PER': 122} |
| hi/test | 1129 | 358 | {'LOC': 178, 'PER': 100, 'ORG': 80} |
| mr/validation | 754 | 189 | {'ORG': 39, 'LOC': 90, 'PER': 60} |
| mr/test | 777 | 223 | {'LOC': 103, 'PER': 86, 'ORG': 34} |

**NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md)** — `alignment/evaluate_alignment.py` was not run (it requires the final MT models).

## 4. Anonymisation

| Lang | Sentences / gold entities | Residual identifier rate | Missed | Partial | False-positive rate | Incorrect (wrong type) | Non-entity preserved | Consistency violations | Synthetic ID recall E/U/P/A/PAN | Neg. controls changed |
|---|---|---|---|---|---|---|---|---|---|---|
| en | 3453 / 4946 | 13.3% | 13.2% | 1.8% | 10.7% | 13.9% | 98.1% | 0 / 0 | 1.00/1.00/1.00/1.00/1.00 | 0/3 |
| hi | 864 / 1929 | 9.7% | 6.8% | 4.3% | 13.2% | 4.5% | 97.2% | 0 / 0 | 1.00/1.00/1.00/1.00/1.00 | 0/3 |
| mr | 1060 / 1676 | 14.1% | 12.0% | 3.8% | 8.2% | 3.9% | 97.8% | 0 / 0 | 1.00/1.00/1.00/1.00/1.00 | 0/3 |

Cross-lingual identity consistency: **NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md)**.

## 5. End-to-end pipeline (NER → alignment → anonymisation → MT → evaluation)

**NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md)** — `pipeline/run_pipeline.py` was not run (it requires the final MT models and alignment results).

## 6. Master table

| Component | Language | Model | Dataset | Main metric | Score | Baseline | Baseline score | Comparable? |
|---|---|---|---|---|---|---|---|---|
| NER | Hindi | BiLSTM-CRF scratch | Naamapadam hi test (human-annotated) | entity F1 | 0.797 | XLM-R | 0.799 | Yes (same data/scorer; different training data) |
| NER | Hindi | BiLSTM-CRF scratch | Naamapadam hi test (human-annotated) | entity F1 | 0.797 | IndicNER | 0.828 | Partly (same data/scorer; pretrained, possible test exposure) |
| NER | Marathi | BiLSTM-CRF scratch | Naamapadam mr test (human-annotated) | entity F1 | 0.794 | XLM-R | 0.792 | Yes (same data/scorer; different training data) |
| NER | Marathi | BiLSTM-CRF scratch | Naamapadam mr test (human-annotated) | entity F1 | 0.794 | IndicNER | 0.805 | Partly (same data/scorer; pretrained, possible test exposure) |
| NER | English | BiLSTM-CRF scratch | CoNLL-2003 test | entity F1 | 0.861 | BERT fine-tuned (ours) | 0.923 | Yes (same data/split/scorer) |
| NER | English | BiLSTM-CRF scratch | CoNLL-2003 test | entity F1 | 0.861 | XLM-R | 0.914 | Yes (same data/scorer) |
| MT | Hindi | Transformer scratch | Samanantar subset test (1000) | BLEU / chrF++ | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) | IndicTrans2 | NOT RUN | No (reference only) |
| Alignment | Hindi | translit+MT+NER aligner | Wikidata-verified gold | P@1 / MRR | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) | random ranking | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) | Yes |
| Anonymisation | Hindi | NER + regex placeholders | hindi NER test set | residual identifier rate (lower better) | 9.7% | — | — | n/a |
| MT | Marathi | Transformer scratch | Samanantar subset test (1000) | BLEU / chrF++ | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) | IndicTrans2 | NOT RUN | No (reference only) |
| Alignment | Marathi | translit+MT+NER aligner | Wikidata-verified gold | P@1 / MRR | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) | random ranking | NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md) | Yes |
| Anonymisation | Marathi | NER + regex placeholders | marathi NER test set | residual identifier rate (lower better) | 14.1% | — | — | n/a |

## 7. Experiment record (trained models)

| Model | Architecture | Parameters | Training samples | Epochs | LR | Batch | Seed | Best checkpoint | Evaluation |
|---|---|---|---|---|---|---|---|---|---|
| Hindi NER | BiLSTM-CRF (emb 128, char-CNN 50, BiLSTM 2×128/dir, CRF) | 5,396,278 | 100000 | 8 (max 8, patience 2) | 0.001 | 32 | 42 | epoch 7 (val F1 0.7569) | held-out test |
| Marathi NER | BiLSTM-CRF (emb 128, char-CNN 50, BiLSTM 2×128/dir, CRF) | 7,124,914 | 150000 | 8 (max 8, patience 2) | 0.001 | 32 | 42 | epoch 7 (val F1 0.7582) | held-out test |
| English NER | BiLSTM-CRF (emb 128, char-CNN 50, BiLSTM 2×128/dir, CRF) | 2,112,764 | 13492 | 30 (max 30, patience 5) | 0.001 | 32 | 42 | epoch 26 (val F1 0.9063) | held-out test |
| English BERT (reference) | bert-base-cased + token-classification head | 107,725,063 | 13492 | 3 (resumed after epoch 1) | 5e-05 | 16 | 42 | epoch 3 (val F1 0.9539) | held-out test |
| MT en→hi | Transformer 3+3 layers, d 256, ffn 1024, heads 4 | 9,626,624 | 50000 | 10 (max 10, patience 3) | 0.0007 | 64 | 42 | epoch 10 (val loss 4.2238) | not run (held-out test planned) |
| MT en→mr | Transformer 3+3 layers, d 256, ffn 1024, heads 4 | 9,626,624 | 50000 | 5 (max 10, patience 3) | 0.0007 | 64 | 42 | epoch 5 (val loss 4.9608) — training not finished | not run (held-out test planned) |

## 8. File audit

Protected files re-hashed now: 27 (NER checkpoints Hindi/Marathi/English, English BERT, NER datasets, trained_ner.py). Changed: **0** 

**Model checkpoints:**

- `models/hindi_ner` — present
- `models/marathi_ner` — present
- `models/english_ner_scratch` — present
- `models/english_ner` — present
- `models/mt_en_hi` — present
- `models/mt_en_mr` — present

**Evaluation outputs:**

- `training/results/evaluation.json` — present
- `results/ner/indicner_benchmark.json` — present
- `results/ner/error_analysis_summary.txt` — present
- `results/ner/error_categories.txt` — present
- `mt/results/mt_results.json` — not yet created
- `alignment/results/alignment_results.json` — not yet created
- `anonymisation/results/anonymisation_mono.json` — present
- `anonymisation/results/anonymisation_cross.json` — not yet created
- `pipeline/results/pipeline_summary.json` — not yet created

