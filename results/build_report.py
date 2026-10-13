"""
Assemble the final research package from the VERIFIED result files on disk.

    python results/build_report.py   ->   results/FINAL_REPORT.md

Reads only result/config files (it never runs a model). Anything whose
result file does not exist is shown as NOT AVAILABLE (not run at handoff)
or NOT RUN (with the reason) - no number is ever filled in by hand.
"""

import datetime
import hashlib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "FINAL_REPORT.md")
PENDING = "NOT AVAILABLE (not run — work stopped at handoff; see PROJECT_STATUS.md)"


def path(*p):
    return os.path.join(ROOT, *p)


def load(*p):
    f = path(*p)
    if not os.path.exists(f):
        return None
    with open(f, encoding="utf-8") as fh:
        return json.load(fh)


def fmt(x, digits=3):
    return "—" if x is None else (f"{x:.{digits}f}" if isinstance(x, float) else str(x))


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


# ----------------------------------------------------------------------------- data
ner_rows = {(r["model"], r["language"]): r for r in (load("training", "results", "evaluation.json") or [])}
indicner = load("results", "ner", "indicner_benchmark.json")
mt = load("mt", "results", "mt_results.json")
# IndicTrans2 + COMET, computed in Colab by mt/colab_indictrans2_comet.py
ext = load("mt", "results", "external", "indictrans2_comet_results.json")


def ext_check(lang):
    """Colab re-scoring of OUR hypotheses must reproduce our local scores and signatures."""
    if not (ext and mt and lang in ext):
        return None
    ours, local = ext[lang]["ours_rescored"], mt[lang]
    same_scores = all(abs(ours[k] - local[k]) < 0.01 for k in ("BLEU", "chrF", "chrF++"))
    same_sigs = all(ours["signatures"][k].split("|version")[0] == local["signatures"][k].split("|version")[0]
                    for k in ("BLEU", "chrF", "chrF++"))
    return same_scores and same_sigs
align = load("alignment", "results", "alignment_results.json")
gold_info = load("alignment", "data", "gold_info.json")
anon_mono = load("anonymisation", "results", "anonymisation_mono.json")
anon_cross = load("anonymisation", "results", "anonymisation_cross.json")
pipe = load("pipeline", "results", "pipeline_summary.json")
cfg = {k: load("models", k, "config.json") for k in ("hindi_ner", "marathi_ner", "english_ner_scratch", "mt_en_hi", "mt_en_mr")}
bert_info = load("models", "english_ner", "training_info.json")
bert_val_acc = load("results", "ner", "english_bert_validation_accuracy.json")
mt_data = {l: load("mt", "data", l, "dataset_info.json") for l in ("hi", "mr")}
ner_data = {l: load("training", "data", l, "dataset_info.json") for l in ("hindi", "marathi", "english")}

NER = {"Hindi": ("Hindi BiLSTM-CRF", "hindi"), "Marathi": ("Marathi BiLSTM-CRF", "marathi"),
       "English": ("English BiLSTM-CRF", "english")}


def ner(lang):
    return ner_rows.get(NER[lang])


def baseline(lang):
    return ner_rows.get(("Baseline XLM-R", NER[lang][1]))


L = []
w = L.append


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "---|" * len(header))
    for r in rows:
        w("| " + " | ".join(str(c) for c in r) + " |")
    w("")


# ----------------------------------------------------------------------------- header
w("# Synthetic Parallel Sentence and Named Entity Pair Generation for Indian Languages — Final Research Package")
w("")
w(f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} by `results/build_report.py` from the result files on disk. "
  "Every number below is read from a result file produced by an actual run; items that were not run or not finished are "
  f"marked **{PENDING}**, items that could not be run are marked **NOT RUN** with the reason.")
w("")
status = {"NER": "complete", "MT / generation": "complete" if mt else "training paused / evaluation pending — " + PENDING,
          "Entity alignment": "complete" if align else PENDING,
          "Anonymisation (monolingual)": "complete" if anon_mono else PENDING,
          "Anonymisation (cross-lingual)": "complete" if anon_cross else PENDING,
          "End-to-end pipeline": "complete" if pipe else PENDING,
          "IndicTrans2 reference": ("complete (Colab, reference-only)" if ext else
                                    "NOT RUN — Colab script prepared: mt/colab_indictrans2_comet.py (IndicTrans2's custom code is "
                                    "incompatible with transformers 5 used here)"),
          "COMET": ("complete (Colab, Unbabel/wmt22-comet-da)" if ext else
                    "NOT RUN — would be computed by mt/colab_indictrans2_comet.py in Colab (unbabel-comet cannot be installed "
                    "here: no NumPy wheel for Python 3.14/ARM64, no C compiler)")}
table(["Component", "Status"], status.items())

# ----------------------------------------------------------------------------- 1 NER
w("## 1. Named Entity Recognition")
w("")
w("Entity-level exact match (type + span), micro-averaged, on held-out test sets; no aliases or post-processing.")
w("")
rows = []
for lang in ("Hindi", "Marathi", "English"):
    r = ner(lang)
    rows.append([lang, "BiLSTM-CRF (scratch)", r["test_set"], r["test_sentences"], fmt(r["precision"]), fmt(r["recall"]),
                 f"**{fmt(r['f1'])}**", fmt(r["PER_f1"]), fmt(r["ORG_f1"]), fmt(r["LOC_f1"])])
table(["Language", "Model", "Test set", "Sentences", "P", "R", "F1", "PER F1", "ORG F1", "LOC F1"], rows)

w("### 1.1 Comparison with baselines")
w("")
rows = []
for lang in ("Hindi", "Marathi", "English"):
    r, b = ner(lang), baseline(lang)
    rows.append([lang, fmt(r["f1"]), "XLM-R (Davlan/xlm-roberta-base-ner-hrl, pretrained, not trained here)", fmt(b["f1"]),
                 f"{r['f1'] - b['f1']:+.3f}", "same test set & scorer; baseline not trained on Naamapadam"])
if indicner:
    for lang, key in (("Hindi", "hindi"), ("Marathi", "marathi")):
        x = indicner[key]
        rows.append([lang, fmt(x["ours"]["micro"]["f1"]), "IndicNER (ai4bharat/IndicNER public checkpoint, mBERT-uncased fine-tuned)",
                     fmt(x["indicner"]["micro"]["f1"]), f"{x['diff_ours_minus_indicner']:+.3f} "
                     f"(paired-bootstrap 95% CI {x['diff_95ci_paired_bootstrap'][0]:+.3f} … {x['diff_95ci_paired_bootstrap'][1]:+.3f})",
                     "same test set & scorer; pretrained; possible test exposure; not the paper's model"])
eb = ner_rows.get(("English BERT", "english"))
if eb:
    rows.append(["English", fmt(ner("English")["f1"]), "bert-base-cased fine-tuned in this project (reference experiment)",
                 fmt(eb["f1"]), f"{ner('English')['f1'] - eb['f1']:+.3f}", "same data, split & scorer; pretrained vs scratch"])
table(["Language", "Ours F1", "Baseline", "Baseline F1", "Difference", "Comparability"], rows)

if bert_val_acc:
    a = bert_val_acc
    w("### 1.1a English BERT token-level accuracy (validation split)")
    w("")
    w("Token-level accuracy = share of tokens whose predicted BIO tag matches the reference tag. It is a **validation** "
      "metric (not test accuracy) and is not comparable with the entity-level F1 scores above, which score complete entities.")
    w("")
    table(["Model", "Checkpoint", "Dataset", "Correct tokens", "Total tokens", "Token accuracy"],
          [["English BERT (reference)", f"`{a['model_dir']}`", f"`{a['dataset']}`", f"{a['correct_tokens']:,}",
            f"{a['total_tokens']:,}", f"**{100 * a['correct_tokens'] / a['total_tokens']:.2f}%**"]])
    w(a["note"])
    w("")

w("### 1.2 NER error analysis (test sets)")
w("")
w("From `results/ner/error_analysis_summary.txt` and `results/ner/error_categories.txt` "
  "(scripts `analysis/ner_error_analysis.py`, `analysis/ner_error_categories.py`).")
w("")
summary_txt = open(path("results", "ner", "error_analysis_summary.txt"), encoding="utf-8").read()
w("```")
w(summary_txt.strip())
w("```")
w("")

# ----------------------------------------------------------------------------- 2 MT
w("## 2. Machine translation / generation (English → Hindi, English → Marathi)")
w("")
if mt:
    rows = []
    for lang in ("hi", "mr"):
        r = mt[lang]
        comet = r["COMET"] if r["COMET"] is not None else (ext[lang]["ours_rescored"]["COMET"] if ext_check(lang) else None)
        rows.append([f"en→{lang}", r["test_pairs"], fmt(r["BLEU"], 2), fmt(r["chrF"], 2), fmt(r["chrF++"], 2),
                     fmt(comet, 4) if comet is not None else "NOT RUN"])
    table(["Pair", "Test pairs", "BLEU", "chrF", "chrF++", "COMET"], rows)
    w(f"Decoding: {mt['hi']['decoding']}. sacrebleu signatures: BLEU `{mt['hi']['signatures']['BLEU']}`, "
      f"chrF `{mt['hi']['signatures']['chrF']}`.")
    w("")
    w("### 2.1 Named-entity preservation in translation (reference-anchored, NER on both sides, no aliases)")
    w("")
    rows = []
    for lang in ("hi", "mr"):
        e = mt[lang]["entity_preservation"]
        c, r = e["counts"], e["rates"]
        rows.append([f"en→{lang}", c["reference_entities"], f"{c['preserved']} ({pct(r['preserved_rate'])})",
                     f"{c['translated_incorrectly']} ({pct(r['translated_incorrectly_rate'])})",
                     f"{c['dropped']} ({pct(r['dropped_rate'])})",
                     f"{c['hallucinated_added']}/{c['hypothesis_entities']} ({pct(r['hallucinated_added_rate'])})"])
    table(["Pair", "Reference entities", "Preserved", "Translated incorrectly", "Dropped", "Hallucinated/added"], rows)
    for lang in ("hi", "mr"):
        ex = load("mt", "results", lang, "entity_examples.json") or {}
        w(f"Examples en→{lang} (from `mt/results/{lang}/entity_examples.json`):")
        for cat in ("translated_incorrectly", "dropped", "hallucinated_added"):
            for item in ex.get(cat, [])[:2]:
                w(f"- {cat}: `{json.dumps(item, ensure_ascii=False)[:220]}`")
        w("")
else:
    epochs = ", ".join(f"en→{l} {cfg[k]['training'].get('epochs_run', '?')} of {cfg[k]['training']['epochs_max']}"
                       for l, k in (("hi", "mt_en_hi"), ("mr", "mt_en_mr")) if cfg[k])
    w(f"**{PENDING}** — MT training paused ({epochs} planned epochs; resumable from `models/mt_en_*/last_checkpoint.pt`). `mt/evaluate_mt.py` was not run, so no test BLEU/chrF/chrF++ or entity-preservation results exist.")
    w("")
w("### 2.2 IndicTrans2 reference (reference-only, NOT a like-for-like comparison)")
w("")
if ext and mt:
    rows = []
    for lang in ("hi", "mr"):
        i, o = ext[lang]["indictrans2"], ext[lang]["ours_rescored"]
        ok = ext_check(lang)
        rows.append([f"en→{lang}", ext[lang]["test_pairs"],
                     f"{mt[lang]['BLEU']:.2f} / {mt[lang]['chrF++']:.2f} / {fmt(o['COMET'], 4)}",
                     f"{i['BLEU']:.2f} / {i['chrF++']:.2f} / {fmt(i['COMET'], 4)}",
                     f"{mt[lang]['BLEU'] - i['BLEU']:+.2f} / {mt[lang]['chrF++'] - i['chrF++']:+.2f} / "
                     f"{(o['COMET'] - i['COMET']):+.4f}",
                     "yes" if ok else "**NO — scores/signatures differ; do not compare**"])
    table(["Pair", "Test pairs", "Ours BLEU / chrF++ / COMET", "IndicTrans2 BLEU / chrF++ / COMET",
           "Difference (ours − IT2)", "Same sentences, references & scorer verified"], rows)
    w(f"IndicTrans2 run in Colab: {json.dumps(ext['environment'])}. Same test sentences and references, same "
      "sacrebleu configuration (verified by re-scoring our hypotheses there). Not like-for-like: IndicTrans2 is a "
      "large pretrained model trained on BPCC, which includes Samanantar — the source of our test sentences — so "
      "test-set exposure is likely. No published IndicTrans2 number is quoted.")
else:
    w("**NOT RUN.** To run: `mt/colab_indictrans2_comet.py` in Colab on the exported test files (IndicTrans2's custom model "
      "and tokenizer code is incompatible with transformers 5 used in this project: it imports the removed "
      "`transformers.onnx` and relies on removed model/tokenizer internals). It is a *reference-only* baseline: "
      "IndicTrans2 is pretrained on far more data (BPCC, which includes Samanantar, the source of our test "
      "sentences). No published IndicTrans2 number is quoted, because none was verified on our test set.")
w("")

# ----------------------------------------------------------------------------- 3 alignment
w("## 3. Cross-lingual entity alignment")
w("")
if gold_info:
    rows = [[k, v["english_mentions"], v["gold_items"], v.get("by_type")] for k, v in gold_info["splits"].items()]
    w("Gold sets (Wikidata-verified, built from the held-out MT validation/test splits; `alignment/build_gold.py`):")
    w("")
    table(["Split", "English mentions", "Gold items", "By type"], rows)
if align:
    rows = []
    for lang in ("hi", "mr"):
        r = align[lang]
        for name in ("random", "translit", "mt", "ner", "combined"):
            m = r["test_metrics"][name]
            rows.append([f"en→{lang}", name, m["items"], fmt(m["accuracy"]), fmt(m["precision_at_1"]),
                         fmt(m["recall_at_3"]), fmt(m["recall_at_5"]), fmt(m["recall_at_10"]), fmt(m["mrr"])])
    table(["Pair", "Aligner", "Test items", "Accuracy", "Precision@1", "Recall@3", "Recall@5", "Recall@10", "MRR"], rows)
    for lang in ("hi", "mr"):
        e = align[lang]["test_error_analysis"]
        w(f"Error analysis en→{lang} (combined aligner, weights {align[lang]['tuned_weights_on_validation']} tuned on validation):")
        w("")
        table(["Category", "Count", "Share"],
              [[k, v["count"], pct(v["share"])] for k, v in e["categories"].items()]
              + [["missing (gold not among candidates)", e["missing_gold_not_in_candidates"]["count"],
                  pct(e["missing_gold_not_in_candidates"]["share"])],
                 ["P@1 with ≥2 same-type candidate entities", e["multiple_same_type_candidates"]["items"],
                  fmt(e["multiple_same_type_candidates"]["precision_at_1"])],
                 ["P@1 with ≤1 same-type candidate entity", e["single_or_no_same_type_candidate"]["items"],
                  fmt(e["single_or_no_same_type_candidate"]["precision_at_1"])]]
              + [[f"P@1 {t}", v["items"], fmt(v["precision_at_1"])] for t, v in e["by_type"].items()])
else:
    w(f"**{PENDING}** — `alignment/evaluate_alignment.py` was not run (it requires the final MT models).")
    w("")

# ----------------------------------------------------------------------------- 4 anonymisation
w("## 4. Anonymisation")
w("")
if anon_mono:
    rows = []
    for lang in ("en", "hi", "mr"):
        r = anon_mono[lang]
        m, s, c = r["metrics"], r["synthetic_identifiers"], r["counts"]
        rows.append([lang, f"{c['sentences']} / {c['gold_entities']}", pct(m["residual_identifier_rate"]),
                     pct(m["missed_anonymisation_rate"]), pct(m["partial_anonymisation_rate"]),
                     pct(m["false_positive_rate"]), pct(m["incorrect_anonymisation_rate"]),
                     pct(m["non_entity_preservation"]),
                     f"{m['identity_consistency']['entity_texts_with_multiple_placeholders']} / "
                     f"{m['identity_consistency']['placeholders_shared_by_different_texts']}",
                     "/".join(f"{v:.2f}" for v in s["recall_by_type"].values()),
                     f"{s['negative_controls_wrongly_anonymised']}/{s['negative_controls']}"])
    table(["Lang", "Sentences / gold entities", "Residual identifier rate", "Missed", "Partial", "False-positive rate",
           "Incorrect (wrong type)", "Non-entity preserved", "Consistency violations", "Synthetic ID recall E/U/P/A/PAN",
           "Neg. controls changed"], rows)
if anon_cross:
    table(["Pair", "Gold pairs", "Same placeholder both sides", "Gold target left visible"],
          [[f"en-{l}", anon_cross[l]["gold_pairs"], pct(anon_cross[l]["cross_lingual_consistency"]),
            pct(anon_cross[l]["target_gold_left_visible"])] for l in ("hi", "mr")])
else:
    w(f"Cross-lingual identity consistency: **{PENDING}**.")
    w("")

# ----------------------------------------------------------------------------- pipeline
w("## 5. End-to-end pipeline (NER → alignment → anonymisation → MT → evaluation)")
w("")
if pipe:
    rows = [[f"en-{l}", p["examples"], p["examples_ok"], p["crashes"], p["examples_with_empty_output"],
             p["examples_with_corrupted_text"], p["examples_with_leakage"],
             f"{p['placeholders_preserved_by_mt']['preserved']}/{p['placeholders_preserved_by_mt']['total']} "
             f"({pct(p['placeholders_preserved_by_mt']['rate'])})", p["protected_ner_files_unchanged"]]
            for l, p in pipe.items()]
    table(["Pair", "Examples", "All checks OK", "Crashes", "Empty", "Corrupted", "Leakage",
           "Placeholders kept by MT", "NER checkpoints unchanged"], rows)
else:
    w(f"**{PENDING}** — `pipeline/run_pipeline.py` was not run (it requires the final MT models and alignment results).")
    w("")

# ----------------------------------------------------------------------------- master table
w("## 6. Master table")
w("")
rows = []
for lang in ("Hindi", "Marathi"):
    r, b = ner(lang), baseline(lang)
    rows.append(["NER", lang, "BiLSTM-CRF scratch", r["test_set"], "entity F1", fmt(r["f1"]), "XLM-R", fmt(b["f1"]),
                 "Yes (same data/scorer; different training data)"])
    if indicner:
        x = indicner[NER[lang][1]]
        rows.append(["NER", lang, "BiLSTM-CRF scratch", r["test_set"], "entity F1", fmt(r["f1"]), "IndicNER",
                     fmt(x["indicner"]["micro"]["f1"]), "Partly (same data/scorer; pretrained, possible test exposure)"])
r, eb = ner("English"), ner_rows.get(("English BERT", "english"))
rows.append(["NER", "English", "BiLSTM-CRF scratch", r["test_set"], "entity F1", fmt(r["f1"]), "BERT fine-tuned (ours)",
             fmt(eb["f1"]) if eb else "—", "Yes (same data/split/scorer)"])
rows.append(["NER", "English", "BiLSTM-CRF scratch", r["test_set"], "entity F1", fmt(r["f1"]), "XLM-R",
             fmt(baseline("English")["f1"]), "Yes (same data/scorer)"])
for lang, code in (("Hindi", "hi"), ("Marathi", "mr")):
    it2 = (f"{ext[code]['indictrans2']['BLEU']:.2f} / {ext[code]['indictrans2']['chrF++']:.2f}"
           if ext_check(code) else "NOT RUN")
    if mt:
        rows.append(["MT", lang, "Transformer scratch", f"Samanantar subset test ({mt[code]['test_pairs']})", "BLEU / chrF++",
                     f"{mt[code]['BLEU']:.2f} / {mt[code]['chrF++']:.2f}", "IndicTrans2", it2,
                     "No (reference only: pretrained, likely test exposure)"])
    else:
        rows.append(["MT", lang, "Transformer scratch", "Samanantar subset test (1000)", "BLEU / chrF++", PENDING,
                     "IndicTrans2", "NOT RUN", "No (reference only)"])
    if align:
        m = align[code]["test_metrics"]
        rows.append(["Alignment", lang, "translit+MT+NER aligner", f"Wikidata-verified gold ({m['combined']['items']})",
                     "P@1 / MRR", f"{m['combined']['precision_at_1']:.3f} / {m['combined']['mrr']:.3f}",
                     "random ranking", f"{m['random']['precision_at_1']:.3f} / {m['random']['mrr']:.3f}", "Yes (same items/candidates)"])
    else:
        rows.append(["Alignment", lang, "translit+MT+NER aligner", "Wikidata-verified gold", "P@1 / MRR", PENDING,
                     "random ranking", PENDING, "Yes"])
    if anon_mono:
        a = anon_mono[code]["metrics"]
        rows.append(["Anonymisation", lang, "NER + regex placeholders", NER[lang][1] + " NER test set",
                     "residual identifier rate (lower better)", pct(a["residual_identifier_rate"]), "—", "—", "n/a"])
table(["Component", "Language", "Model", "Dataset", "Main metric", "Score", "Baseline", "Baseline score", "Comparable?"], rows)

# ----------------------------------------------------------------------------- experiment record
w("## 7. Experiment record (trained models)")
w("")
rows = []
for key, name in (("hindi_ner", "Hindi NER"), ("marathi_ner", "Marathi NER"), ("english_ner_scratch", "English NER")):
    c = cfg[key]
    t = c["training"]
    m = c["model"]
    rows.append([name, f"BiLSTM-CRF (emb {m['embedding_dim']}, char-CNN {m['char_filters']}, BiLSTM {m['num_layers']}×{m['hidden_dim']}/dir, CRF)",
                 f"{c['parameters']:,}", t["train_sentences"], f"{t['epochs_run']} (max {t['epochs_max']}, patience {t['patience']})",
                 t["learning_rate"], t["batch_size"], t["seed"], f"epoch {c['best_epoch']} (val F1 {c['best_dev_f1']})",
                 "held-out test"])
if bert_info:
    t = bert_info["training"]
    rows.append(["English BERT (reference)", "bert-base-cased + token-classification head", f"{bert_info['parameters']:,}",
                 t["train_sentences"], f"{len(bert_info['log'])} (resumed after epoch 1)", t["learning_rate"], t["batch_size"],
                 t["seed"], f"epoch {bert_info['best_epoch']} (val F1 {bert_info['best_dev_f1']})", "held-out test"])
for key, name in (("mt_en_hi", "MT en→hi"), ("mt_en_mr", "MT en→mr")):
    c = cfg[key]
    if c:
        t, m = c["training"], c["model"]
        rows.append([name, f"Transformer {m['encoder_layers']}+{m['decoder_layers']} layers, d {m['d_model']}, ffn {m['ffn_dim']}, heads {m['heads']}",
                     f"{c['parameters']:,}", t["train_pairs"], f"{t.get('epochs_run', '?')} (max {t['epochs_max']}, patience {t['patience']})",
                     t["peak_learning_rate"], t["batch_size_sentences"], t["seed"],
                     f"epoch {c.get('best_epoch')} (val loss {c.get('best_val_loss')})"
                     + ("" if mt or t.get("epochs_run") == t["epochs_max"] else " — training not finished"),
                     "held-out test" if mt else "not run (held-out test planned)"])
table(["Model", "Architecture", "Parameters", "Training samples", "Epochs", "LR", "Batch", "Seed", "Best checkpoint", "Evaluation"], rows)

# ----------------------------------------------------------------------------- file audit
w("## 8. File audit")
w("")
expected = {}
with open(path("training", "logs", "checksums_before_option_a.txt"), encoding="utf-8") as f:
    for line in f:
        digest, p = line.split(maxsplit=1)
        expected[p.strip().lstrip("*")] = digest
changed = [p for p, d in expected.items()
           if hashlib.sha256(open(path(p), "rb").read()).hexdigest() != d]
w(f"Protected files re-hashed now: {len(expected)} (NER checkpoints Hindi/Marathi/English, English BERT, NER datasets, "
  f"trained_ner.py). Changed: **{len(changed)}** {changed if changed else ''}")
w("")
groups = {
    "Model checkpoints": ["models/hindi_ner", "models/marathi_ner", "models/english_ner_scratch", "models/english_ner",
                          "models/mt_en_hi", "models/mt_en_mr"],
    "Evaluation outputs": ["training/results/evaluation.json", "results/ner/indicner_benchmark.json",
                           "results/ner/error_analysis_summary.txt", "results/ner/error_categories.txt",
                           "mt/results/mt_results.json", "alignment/results/alignment_results.json",
                           "anonymisation/results/anonymisation_mono.json", "anonymisation/results/anonymisation_cross.json",
                           "pipeline/results/pipeline_summary.json"],
}
for g, items in groups.items():
    w(f"**{g}:**")
    w("")
    for it in items:
        w(f"- `{it}` — {'present' if os.path.exists(path(it)) else 'not yet created'}")
    w("")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"written {OUT} ({len(L)} lines); protected files changed: {len(changed)}")
