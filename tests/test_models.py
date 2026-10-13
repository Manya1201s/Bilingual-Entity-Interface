"""
Tests for the trained NER models and their integration in app.py.

Run from the project folder (after training):

    .venv\\Scripts\\python tests\\test_models.py

Two kinds of output:
* CHECK lines are asserted (loading, routing, fallback, confidence, alias
  fallback, full app pipeline, UI routing). Exit code 0 only if all pass.
* MODEL-ONLY lines report what each trained model predicts WITHOUT aliases.
  They are observations of model quality and are not counted as test
  passes or failures, so aliases can never hide a model miss.
"""

import json
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
APP_PATH = os.path.join(PROJECT, "app.py")
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import app  # noqa: E402
from trained_ner import LABELS, BertNERPredictor, BiLSTMCRFPredictor  # noqa: E402

failures, passes = [], 0
observations = []


def check(name, condition, detail=""):
    global passes
    if condition:
        passes += 1
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}\n        -> {detail}")


def observe(language, sentence, expected, predicted):
    """Record model-only output: which expected entities the model found."""
    found = {(e["word"], e["entity_group"]) for e in predicted}
    hits = [f"{t}->{label}" for t, label in expected if (t, label) in found]
    misses = [f"{t}->{label}" for t, label in expected if (t, label) not in found]
    observations.append((language, sentence, hits, misses, predicted))
    status = "all found" if not misses else f"missed {misses}"
    print(f"  MODEL-ONLY [{language}] {sentence}\n"
          f"      predicted: {[(e['word'], e['entity_group'], round(e['score'], 3)) for e in predicted]}"
          f"\n      expected : {status}")


def section(title):
    print(f"\n=== {title}")


def found(entities, text, label):
    return any(e["Entity"] == text and e["Entity Type"] == label for e in entities)


def well_formed(entities, sentence):
    spans = sorted((e["Start"], e["End"]) for e in entities)
    return (all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
            and all(sentence[e["Start"]:e["End"]] == e["Entity"] for e in entities)
            and all(0.0 <= e["Confidence"] <= 1.0 for e in entities))


def ner(text, language):
    return app.extract_ner(text, 0.30, app.MODEL_NAME, language)


# ----------------------------------------------------------------------
section("Model loading")

predictors = {}
for language in ("English", "Hindi", "Marathi"):
    predictor, problem = app.load_trained_model(language)
    predictors[language] = predictor
    check(f"{language} model loads from models/{app.TRAINED_MODELS[language][0]}",
          predictor is not None, problem)

for language, predictor in predictors.items():
    if predictor is None:
        continue
    info = predictor.info
    model = predictor.model
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    check(f"{language}: BiLSTM-CRF predictor, labels = common label set",
          isinstance(predictor, BiLSTMCRFPredictor) and predictor.labels == LABELS)
    check(f"{language}: config says trained from scratch, no pretrained embeddings",
          "from scratch" in info["model_type"] and "no pretrained" in info["model_type"])
    check(f"{language}: embedding table size == vocabulary built from training data",
          model.word_embedding.num_embeddings == len(predictor.vocab.words) == info["vocab_size"])
    check(f"{language}: architecture = Embedding -> BiLSTM -> Linear -> CRF "
          f"({n_params:,} parameters)",
          model.lstm.bidirectional and hasattr(model, "crf") and n_params == info["parameters"])
    check(f"{language}: a best checkpoint was selected on validation data",
          info.get("best_epoch", 0) >= 1 and info.get("best_dev_f1", 0) > 0, info.get("best_epoch"))

check("English route uses models/english_ner_scratch (not the BERT experiment)",
      app.TRAINED_MODELS["English"][0] == "english_ner_scratch"
      and not isinstance(predictors["English"], BertNERPredictor))
bert_dir = os.path.join(PROJECT, "models", "english_ner")
check("English BERT experiment preserved as a reference (models/english_ner)",
      os.path.exists(os.path.join(bert_dir, "model.safetensors"))
      and all(folder != "english_ner" for folder, _ in app.TRAINED_MODELS.values()))

# ----------------------------------------------------------------------
section("Language routing in extract_ner")

ROUTES = [
    ("English", "Aarav Sharma works at Infosys.", "English BiLSTM-CRF (scratch)"),
    ("Hindi", "प्रिया मेहता दिल्ली में रहती हैं।", "Hindi BiLSTM-CRF (scratch)"),
    ("Marathi", "प्रिया मेहता जिनेव्हा येथे गेल्या.", "Marathi BiLSTM-CRF (scratch)"),
    ("Auto Detect", "Aarav Sharma works at Infosys.", app.BASELINE_SOURCE),
]
for language, sentence, source in ROUTES:
    entities = ner(sentence, language)
    sources = {e["Source"] for e in entities}
    check(f"{language:11} -> {source}",
          any(s.startswith(source) for s in sources), sources)

saved = dict(app.TRAINED_MODELS)
app.TRAINED_MODELS["Hindi"] = ("does_not_exist_ner", saved["Hindi"][1])
app.load_trained_model.clear()
entities = ner("प्रिया मेहता दिल्ली में रहती हैं।", "Hindi")
check("missing trained model -> falls back to baseline, no crash",
      entities and all(e["Source"].startswith(app.BASELINE_SOURCE) or "alias" in e["Source"]
                       for e in entities), entities)
app.TRAINED_MODELS.update(saved)
app.load_trained_model.clear()

# ----------------------------------------------------------------------
section("STEP 16 sentences: model-only (observed) and full pipeline (asserted)")

STEP16 = [
    ("Hindi", "आरव शर्मा इंफोसिस में काम करते हैं।", [("आरव शर्मा", "PER"), ("इंफोसिस", "ORG")]),
    ("Hindi", "प्रिया मेहता दिल्ली में रहती हैं।", [("प्रिया मेहता", "PER"), ("दिल्ली", "LOC")]),
    ("Marathi", "आरव शर्मा इन्फोसिसमध्ये काम करतात.", [("आरव शर्मा", "PER"), ("इन्फोसिस", "ORG")]),
    ("Marathi", "प्रिया मेहता जिनेव्हा येथे गेल्या.", [("प्रिया मेहता", "PER"), ("जिनेव्हा", "LOC")]),
    ("English", "Aarav Sharma works at Infosys.", [("Aarav Sharma", "PER"), ("Infosys", "ORG")]),
    ("English", "Priya Mehta visited Geneva.", [("Priya Mehta", "PER"), ("Geneva", "LOC")]),
]
for language, sentence, expected in STEP16:
    observe(language, sentence, expected, predictors[language].predict(sentence))
    entities = ner(sentence, language)
    for text, label in expected:
        row = [e for e in entities if e["Entity"] == text and e["Entity Type"] == label]
        check(f"pipeline [{language}] {text} -> {label}"
              + (f" (source: {row[0]['Source']})" if row else ""),
              bool(row), [(e["Entity"], e["Entity Type"], e["Source"]) for e in entities])
    check(f"pipeline [{language}] well-formed: no duplicates/overlaps, exact spans",
          well_formed(entities, sentence), entities)

# ----------------------------------------------------------------------
section("Geneva: model-only first, then alias fallback")

GENEVA = [
    ("English", "Geneva is an international city.", "Geneva"),
    ("Hindi", "जेनेवा में एक सम्मेलन हुआ।", "जेनेवा"),
    ("Hindi", "जिनेवा में एक सम्मेलन हुआ।", "जिनेवा"),
    ("Marathi", "जिनेव्हा येथे परिषद झाली.", "जिनेव्हा"),
]
for language, sentence, geneva in GENEVA:
    observe(language, sentence, [(geneva, "LOC")], predictors[language].predict(sentence))
for language, sentence, geneva in GENEVA:
    entities = ner(sentence, language)
    row = [e for e in entities if e["Entity"] == geneva]
    check(f"pipeline [{language}] {geneva} -> LOC exactly once"
          + (f" (source: {row[0]['Source']}, conf {row[0]['Confidence']})" if row else ""),
          len(row) == 1 and row[0]["Entity Type"] == "LOC", entities)

# ----------------------------------------------------------------------
section("Confidence and source")

SCORE_SENTENCES = {"English": "Aarav Sharma works at Infosys.",
                   "Hindi": "प्रिया मेहता दिल्ली में रहती हैं।",
                   "Marathi": "आरव शर्मा इन्फोसिसमध्ये काम करतात."}
for language, sentence in SCORE_SENTENCES.items():
    entities = predictors[language].predict(sentence)
    check(f"{language} model scores are real probabilities in (0, 1]",
          entities and all(0.0 < e["score"] <= 1.0 for e in entities), entities)

entities = ner("जिनेव्हा येथे परिषद झाली.", "Marathi")
alias_rows = [e for e in entities if e["Source"] == "Configured alias — review"]
model_rows = [e for e in entities if "alias" not in e["Source"]]
check("alias-only entities keep confidence 1.0 and source 'Configured alias — review'",
      all(e["Confidence"] == 1.0 for e in alias_rows))
check("model entities never get the alias confidence label",
      all(e["Source"] in [v[1] for v in app.TRAINED_MODELS.values()] for e in model_rows), model_rows)
for language, sentence in (("English", "Priya Mehta visited Geneva."),
                           ("Hindi", "जिनेवा में एक सम्मेलन हुआ।")):
    both = [e for e in ner(sentence, language) if e["Source"].endswith("+ configured alias")]
    check(f"[{language}] model + alias rows keep the model's own confidence (< 1.0)",
          all(e["Confidence"] < 1.0 for e in both), both)

# ----------------------------------------------------------------------
section("Mixed-language input")

for language in ("Hindi", "Marathi", "English"):
    sentence = "Geneva और जिनेव्हा के बीच बैठक हुई."
    entities = ner(sentence, language)
    check(f"[{language}] mixed sentence: no crash, well-formed, जिनेव्हा LOC (model or alias)",
          well_formed(entities, sentence) and found(entities, "जिनेव्हा", "LOC"),
          [(e["Entity"], e["Entity Type"], e["Source"]) for e in entities])

# ----------------------------------------------------------------------
section("Evaluation results")

path = os.path.join(PROJECT, "training", "results", "evaluation.json")
rows = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else []
names = {(r["model"], r["language"]) for r in rows}
check("evaluation.json has the 3 final models, the BERT reference and the baseline",
      {("Hindi BiLSTM-CRF", "hindi"), ("Marathi BiLSTM-CRF", "marathi"),
       ("English BiLSTM-CRF", "english"), ("English BERT", "english"),
       ("Baseline XLM-R", "hindi"),
       ("Baseline XLM-R", "marathi"), ("Baseline XLM-R", "english")} <= names, names)
check("every result was computed on a held-out test set",
      rows and all("test" in r["test_set"] and r["test_sentences"] > 0 for r in rows))

# ----------------------------------------------------------------------
section("Streamlit UI routing (real app via AppTest)")

from streamlit.testing.v1 import AppTest  # noqa: E402

for language, sentence, source in ROUTES[:3]:
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.selectbox(key="ner_language").set_value(language)
    at.selectbox(key="ner_engine").set_value(app.TRAINED_ENGINE)
    at.text_area(key="manual_text").set_value(sentence)
    at.button(key="process_input").click().run()
    caption_ok = any(f"NER model for this input: {source}" in c.value for c in at.caption)
    at.button(key="run_ner").click().run()
    tables = [d.value for d in at.dataframe if "Source" in d.value]
    sources = set(tables[0]["Source"]) if tables else set()
    check(f"UI: {language} selected -> caption names {source}, table rows from it",
          not at.exception and caption_ok and any(s.startswith(source) for s in sources),
          {"exceptions": [e.value for e in at.exception], "sources": sources})

at = AppTest.from_file(APP_PATH, default_timeout=900)
at.run()
at.segmented_control(key="bilingual_pair").set_value("English → Marathi").run()
at.text_area(key="bilingual_english").set_value("Priya Mehta visited Geneva.")
at.text_area(key="bilingual_second").set_value("प्रिया मेहता जिनेव्हा येथे गेल्या.")
at.selectbox(key="bilingual_engine").set_value("Language-specific trained models").run()
at.button(key="run_bilingual").click().run()
tables = [d.value for d in at.dataframe if "Source" in d.value]
sources = set().union(*[set(t["Source"]) for t in tables]) if tables else set()
check("UI bilingual: trained models -> English and Marathi BiLSTM-CRF used",
      not at.exception and any(s.startswith("English BiLSTM-CRF") for s in sources)
      and any(s.startswith("Marathi BiLSTM-CRF") for s in sources), sources)

# ----------------------------------------------------------------------
print(f"\nMODEL-ONLY observations (not asserted):")
total_expected = sum(len(h) + len(m) for _, _, h, m, _ in observations)
total_hits = sum(len(h) for _, _, h, _, _ in observations)
for language, sentence, hits, misses, _ in observations:
    print(f"  [{language:7}] {len(hits)}/{len(hits) + len(misses)} found by the model alone"
          + (f"  missed: {misses}" if misses else "") + f"  | {sentence}")
print(f"  total: {total_hits}/{total_expected} expected entities found by the models alone")

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
