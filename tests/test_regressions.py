"""
Regression tests for the NER / entity-boundary / alias / provenance bugs.

Run from the project folder with the Python 3.12 environment:

    .venv312\\Scripts\\python tests\\test_regressions.py

Part 1 (deterministic): the multilingual model is replaced by a fake that
returns fixed token-classification output, so these checks test the
app's post-processing code, not model quality.
Part 2 (real inference): the project's NER models. Assertions are about
code behaviour (right model, well-formed output); model-quality
observations are printed, not asserted.
Part 3: the real Streamlit app (AppTest), including a check that no
translation UI remains (the feature was removed).

Exit code is 0 only if every check passes.
"""

import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
APP_PATH = os.path.join(PROJECT, "app.py")

sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

import app  # noqa: E402  (importing runs the Streamlit script in bare mode)

failures = []
passes = 0


def check(name, condition, detail=""):
    global passes
    if condition:
        passes += 1
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}\n        -> {detail}")


def section(title):
    print(f"\n=== {title}")


def offsets_exact(text, entities):
    return all(text[e["Start"]:e["End"]] == e["Entity"] for e in entities)


def summary(entities):
    return [(e["Entity"], e["Entity Type"], e["Start"], e["End"], e["Source"]) for e in entities]


# ------------------------------------------------------------------
# PART 1 - deterministic, fake multilingual model
# ------------------------------------------------------------------

FAKE_OUTPUT = {}


def fake_load_model(model_name):
    return lambda text: [dict(p) for p in FAKE_OUTPUT.get(text, [])]


def pred(text, piece, label, score, start=None):
    start = text.index(piece) if start is None else start
    return {"entity_group": label, "score": score, "word": piece,
            "start": start, "end": start + len(piece)}


def fake_ner(text, outputs, threshold=0.30):
    FAKE_OUTPUT[text] = outputs
    return app.extract_ner(text, threshold, "fake/model", "Auto Detect")


real_load_model = app.load_model
app.load_model = fake_load_model

section("Bug C: sub-word fragments are not entities (fake model)")

text = "Sentence Pairs"
# Exactly what Davlan/xlm-roberta-base-ner-hrl returns ("simple" aggregation).
outputs = [pred(text, "Sent", "PER", 0.778), pred(text, "Pai", "PER", 0.705)]
for threshold in (0.30, 0.0):
    ents = fake_ner(text, outputs, threshold)
    check(f"'Sentence Pairs' -> no 'Sent'/'Pai' PER (threshold {threshold})",
          ents == [], summary(ents))

text = "Priya lives here."
ents = fake_ner(text, [pred(text, "Pri", "PER", 0.9), pred(text, "ya", "PER", 0.8, 3)])
check("sub-word pieces of one name are joined: 'Pri'+'ya' -> 'Priya' PER 0-5",
      summary(ents) == [("Priya", "PER", 0, 5, app.BASELINE_SOURCE)], summary(ents))

text = "Li met Bo in Xi."
ents = fake_ner(text, [pred(text, "Li", "PER", 0.9), pred(text, "Bo", "PER", 0.9),
                       pred(text, "Xi", "LOC", 0.9)])
check("short whole-word entities kept ('Li', 'Bo', 'Xi')",
      [e["Entity"] for e in ents] == ["Li", "Bo", "Xi"], summary(ents))

text = "Supriya came home."
ents = fake_ner(text, [pred(text, "priya", "PER", 0.95)])
check("span starting inside a word ('priya' in 'Supriya') rejected", ents == [], summary(ents))

text = "Aarav works in Pune."
ents = fake_ner(text, [pred(text, "Aarav", "PER", 0.95), pred(text, "Pun", "LOC", 0.9)])
check("only the fragment is dropped; whole-word 'Aarav' kept",
      [e["Entity"] for e in ents] == ["Aarav"], summary(ents))

text = "आरव शर्मा पुण्यात काम करतात।"
ents = fake_ner(text, [pred(text, "आरव शर्मा", "PER", 0.95), pred(text, "पुण्य", "LOC", 0.9)])
check("Devanagari span before an attached suffix kept ('पुण्य' -> 'पुण्या' in 'पुण्यात')",
      [(e["Entity"], e["Entity Type"]) for e in ents] == [("आरव शर्मा", "PER"), ("पुण्या", "LOC")],
      summary(ents))
check("offsets exact after syllable completion", offsets_exact(text, ents), summary(ents))

section("Bug D: alias + model span merging (fake model)")

text = "प्रिया मेहता एक अंतरराष्ट्रीय सम्मेलन के लिए जिनेवा गई।"
ents = fake_ner(text, [pred(text, "प्रिया मेहता", "PER", 0.22), pred(text, "जिनेवा", "LOC", 0.62)])
per = [e for e in ents if e["Entity Type"] == "PER"]
check("weak model span 'प्रिया मेहता' confirmed by alias 'प्रिया' -> full name kept",
      [(e["Entity"], e["Start"], e["End"]) for e in per] == [("प्रिया मेहता", 0, 12)], summary(ents))
check("…with the model's own confidence and both sources named",
      per and per[0]["Confidence"] == 0.22 and "configured alias" in per[0]["Source"]
      and per[0]["Source"].startswith(app.BASELINE_SOURCE), summary(per))
check("offsets exact", offsets_exact(text, ents), summary(ents))

ents = fake_ner(text, [pred(text, "जिनेवा", "LOC", 0.62)])
check("no model evidence -> alias 'प्रिया' only, flagged for review (no fabricated surname)",
      [(e["Entity"], e["Source"]) for e in ents if e["Entity Type"] == "PER"]
      == [("प्रिया", "Configured alias — review")], summary(ents))

ents = fake_ner(text, [pred(text, "मेहता एक", "PER", 0.2)])
check("weak span that does not contain the alias is not used to extend it",
      [e["Entity"] for e in ents if e["Entity Type"] == "PER"] == ["प्रिया"], summary(ents))

text = "विक्रम पटेल पुणे में प्रिया मेहता से मिले।"
ents = fake_ner(text, [pred(text, "विक्रम पटेल", "PER", 1.0), pred(text, "पुणे", "LOC", 0.999),
                       pred(text, "प्रिया मेहता", "PER", 0.25)])
check("three separate entities: विक्रम पटेल PER, पुणे LOC, प्रिया मेहता PER",
      [(e["Entity"], e["Entity Type"], e["Start"], e["End"]) for e in ents]
      == [("विक्रम पटेल", "PER", 0, 11), ("पुणे", "LOC", 12, 16), ("प्रिया मेहता", "PER", 21, 33)],
      summary(ents))

section("Bug B: generic label names (fake configs)")

check("LABEL_0 … LABEL_22 -> no entity labels",
      not app.model_has_entity_labels({i: f"LABEL_{i}" for i in range(23)}))
check("named BIO labels -> entity labels",
      app.model_has_entity_labels({0: "O", 1: "B-PER", 2: "I-PER", 3: "B-LOC"}))

app.load_model = real_load_model


class GenericConfig:
    id2label = {i: f"LABEL_{i}" for i in range(23)}


real_auto_config = app.AutoConfig


class FakeAutoConfig:
    @staticmethod
    def from_pretrained(name):
        return GenericConfig()


app.AutoConfig = FakeAutoConfig
try:
    app.load_model("fake/generic-label-model")
    raised = None
except ValueError as exc:
    raised = str(exc)
app.AutoConfig = real_auto_config
check("load_model refuses a generic-label model with an explanation",
      raised is not None and "LABEL_0" in raised and "PER/LOC/ORG" in raised, raised)
check("MuRIL generic-label model no longer configured in the app",
      not hasattr(app, "HINDI_MODEL_NAME") and not hasattr(app, "load_hindi_model"))

try:
    from transformers import AutoConfig
    muril = AutoConfig.from_pretrained("MichaelHuang/muril_base_cased_hindi_ner",
                                       local_files_only=True)
    check("real MuRIL config (cached) has generic labels -> rejected",
          not app.model_has_entity_labels(muril.id2label), muril.id2label)
except Exception as exc:
    print(f"  SKIP  real MuRIL config not cached ({type(exc).__name__})")

# ------------------------------------------------------------------
# PART 2 - real inference
# ------------------------------------------------------------------

section("Bugs C/D with the real NER models")

text = "Sentence Pairs"
ents = app.extract_ner(text, 0.30, app.MODEL_NAME, "Auto Detect")
check("baseline XLM-R: no partial-word entities from 'Sentence Pairs'",
      all(e["Entity"] in ("Sentence", "Pairs", "Sentence Pairs") for e in ents), summary(ents))
print(f"  OBSERVED baseline: {summary(ents)}")
print(f"  OBSERVED English BiLSTM-CRF (model quality): "
      f"{summary(app.extract_ner(text, 0.30, app.MODEL_NAME, 'English'))}")

text = "Manya lives in NOIDA"
ents = app.extract_ner(text, 0.30, app.MODEL_NAME, "Auto Detect")
check("English NER still finds Manya PER and NOIDA LOC (baseline)",
      [(e["Entity"], e["Entity Type"]) for e in ents] == [("Manya", "PER"), ("NOIDA", "LOC")], summary(ents))

HINDI_CASES = [
    ("प्रिया मेहता एक अंतरराष्ट्रीय सम्मेलन के लिए जिनेवा गई।",
     [("प्रिया मेहता", "PER")]),
    ("विक्रम पटेल पुणे में प्रिया मेहता से मिले।",
     [("विक्रम पटेल", "PER"), ("पुणे", "LOC"), ("प्रिया मेहता", "PER")]),
]
for text, expected in HINDI_CASES:
    ents = app.extract_ner(text, 0.30, app.MODEL_NAME, "Hindi")
    found = [(e["Entity"], e["Entity Type"]) for e in ents]
    check(f"Hindi BiLSTM-CRF: {expected} in {text}",
          all(x in found for x in expected), summary(ents))
    check("  offsets exact", offsets_exact(text, ents), summary(ents))
    base = app.extract_ner(text, 0.30, app.MODEL_NAME, "Auto Detect")
    check("  baseline: no fabricated full name without model evidence",
          all(e["Source"] != "Configured alias — review" or e["Entity"] == "प्रिया" for e in base)
          and offsets_exact(text, base), summary(base))
    print(f"  OBSERVED baseline XLM-R (model quality): {summary(base)}")

# ------------------------------------------------------------------
# PART 3 - real Streamlit app
# ------------------------------------------------------------------

from streamlit.testing.v1 import AppTest  # noqa: E402

section("UI: model provenance")

at = AppTest.from_file(APP_PATH, default_timeout=900)
at.run()
check("sidebar no longer advertises the unusable Hindi fallback",
      not any("Hindi fallback" in c.value for c in at.sidebar.caption))

for engine, caption_model, row_source in [
    (app.TRAINED_ENGINE, "Hindi BiLSTM-CRF (scratch)", "Hindi BiLSTM-CRF (scratch)"),
]:
    # Monolingual workflow: fixed to the language-specific trained models.
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.selectbox(key="ner_language").set_value("Hindi").run()
    at.text_area(key="manual_text").set_value(HINDI_CASES[1][0])
    at.button(key="process_input").click().run()
    caption_ok = any(f"NER model for this input: {caption_model}" in c.value for c in at.caption)
    at.button(key="run_ner").click().run()
    tables = [d.value for d in at.dataframe if "Source" in d.value]
    sources = set(tables[0]["Source"]) if tables else set()
    check(f"{engine}: caption names {caption_model} and rows come from {row_source}",
          not at.exception and caption_ok and sources
          and all(s.startswith(row_source) or s.startswith("Configured alias") for s in sources),
          {"caption_ok": caption_ok, "sources": sources, "exc": [e.value for e in at.exception]})
    check(f"{engine}: no MuRIL / Hindi-model warning",
          not any("MuRIL" in m.value or "muril" in m.value
                  for m in list(at.info) + list(at.warning)))

section("UI: 'Sentence Pairs' through every NER route (no partial-word spans)")


def whole_words(text, rows):
    """Every span starts and ends at a word boundary of `text`."""
    def boundary(i):
        return i in (0, len(text)) or not (
            app.is_joining_char(text[i - 1]) and app.is_joining_char(text[i]))
    return all(boundary(s) and boundary(e) and text[s:e] == t for s, e, t in rows)


SP = "Sentence Pairs"
for language in app.NER_LANGUAGES:
    for engine in [app.TRAINED_ENGINE]:  # monolingual: fixed trained models
        at = AppTest.from_file(APP_PATH, default_timeout=900)
        at.run()
        at.selectbox(key="ner_language").set_value(language).run()
        at.text_area(key="manual_text").set_value(SP)
        at.button(key="process_input").click().run()
        at.button(key="run_ner").click().run()
        tables = [d.value for d in at.dataframe if "Entity Type" in d.value]
        rows = list(zip(tables[0]["Start"], tables[0]["End"], tables[0]["Entity"])) if tables else []
        check(f"section 03 {language} / {engine}: no 'Sent'/'Pai' or other partial-word span",
              not at.exception and whole_words(SP, rows), rows)
        if rows:
            print(f"  OBSERVED model-level whole-phrase prediction (not a span bug): "
                  f"{tables[0][['Entity', 'Entity Type', 'Confidence', 'Source']].values.tolist()}")

for engine in app.NER_ENGINES[:1]:  # bilingual: fixed baseline model
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.segmented_control(key="bilingual_pair").set_value("English → Marathi").run()
    at.text_area(key="bilingual_english").set_value(SP)
    at.text_area(key="bilingual_second").set_value("वाक्य जोड्या")
    at.button(key="run_bilingual").click().run()
    english = [d.value for d in at.dataframe if "Entity Type" in d.value
               and set(d.value["Entity"]) <= {"Sentence", "Pairs", SP, "Sent", "Pai"}]
    rows = [r for t in english for r in zip(t["Start"], t["End"], t["Entity"])]
    check(f"section 04 English + Marathi / {engine}: no partial-word span on the English side",
          not at.exception and whole_words(SP, rows), rows)

section("UI: translation feature removed")

at = AppTest.from_file(APP_PATH, default_timeout=900)
at.run()
at.text_area(key="manual_text").set_value("Manya lives in NOIDA")
at.button(key="process_input").click().run()
at.button(key="run_ner").click().run()

ui_text = " ".join(
    [w.label for w in at.selectbox] + [w.label for w in at.text_area]
    + [b.label for b in at.button]
    + [m.value for m in at.markdown if "<style>" not in m.value]
    + [c.value for c in at.caption] + [h.value for h in at.header]
    + [x.value for x in list(at.info) + list(at.warning) + list(at.success)]
).lower()
FORBIDDEN = ("translat", "direction", "bleu", "draft")
check("no translation / direction wording anywhere in the rendered UI",
      not any(w in ui_text for w in FORBIDDEN), [w for w in FORBIDDEN if w in ui_text])
check("no translation buttons or selectors",
      not any(b.key in ("run_translation", "bilingual_draft") for b in at.button)
      and not any(sb.key in ("ner_direction", "bilingual_direction") for sb in at.selectbox))
check("input language selector offers English / Hindi / Marathi",
      list(at.selectbox(key="ner_language").options) == ["English", "Hindi", "Marathi"])
check("app no longer exposes translation helpers",
      not any(hasattr(app, n) for n in (
          "DIRECTIONS", "load_translator", "translate_sentences",
          "render_translation_step", "entity_loss")))
ents = [d.value for d in at.dataframe if "Entity Type" in d.value]
check("English NER still runs in section 03",
      not at.exception and bool(ents) and set(ents[0]["Entity"]) == {"Manya", "NOIDA"},
      ents[0].to_dict("records") if ents else [e.value for e in at.exception])

at = AppTest.from_file(os.path.join(PROJECT, "pages", "1_Research_Results.py"),
                       default_timeout=300).run()
page_text = " ".join([m.value for m in at.markdown if "<style>" not in m.value]
                     + [c.value for c in at.caption]
                     + [h.value for h in at.header]).lower()
check("Research results page renders without translation content",
      not at.exception and not any(w in page_text for w in ("translat", "bleu", "epochs"))
      and "99.26%" in page_text,
      {"exc": [e.value for e in at.exception],
       "found": [w for w in ("translat", "bleu", "epochs") if w in page_text]})

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
