"""
Input-language selector tests for Indic Entity Lab (app.py).

Run from the project folder with the Python 3.12 environment:

    .venv312\\Scripts\\python tests\\test_languages.py

Checks, for English, Hindi and Marathi:
- sections 01-03: the selected input language routes the text to that
  language's BiLSTM-CRF ("Language-specific trained models"), the caption
  names it, labels are PER/LOC/ORG and offsets are exact;
- multi-word names come back as one entity and aliases stay reviewable;
- section 04 (bilingual comparison): English + Hindi/Marathi sentences go
  to the right model on each side.

Uses the trained NER models in models/; nothing is downloaded or trained.
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
from streamlit.testing.v1 import AppTest  # noqa: E402

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


# language -> (sentence, model, entities that must be found as one span)
CASES = {
    "English": ("Aarav Sharma met Priya Mehta in Pune.",
                "English BiLSTM-CRF (scratch)",
                [("Aarav Sharma", "PER"), ("Priya Mehta", "PER")]),
    "Hindi": ("विक्रम पटेल पुणे में प्रिया मेहता से मिले।",
              "Hindi BiLSTM-CRF (scratch)",
              [("विक्रम पटेल", "PER"), ("पुणे", "LOC"), ("प्रिया मेहता", "PER")]),
    "Marathi": ("प्रिया मेहता जिनेव्हा येथे गेल्या.",
                "Marathi BiLSTM-CRF (scratch)",
                [("प्रिया मेहता", "PER"), ("जिनेव्हा", "LOC")]),
}

section("Sections 01-03: every input language through the UI")

for language, (sentence, model, expected) in CASES.items():
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.selectbox(key="ner_language").set_value(language)
    at.selectbox(key="ner_engine").set_value(app.TRAINED_ENGINE).run()
    at.text_area(key="manual_text").set_value(sentence)
    at.button(key="process_input").click().run()

    check(f"{language}: text area asks for {language}",
          at.text_area(key="manual_text").label == f"Enter one or more {language} sentences",
          at.text_area(key="manual_text").label)
    check(f"{language}: caption names {model}",
          any(f"NER model for this input: {model}" in c.value for c in at.caption),
          [c.value for c in at.caption])
    check(f"{language}: no language-mismatch warning",
          not any("Selected language" in w.value for w in at.warning),
          [w.value for w in at.warning])

    at.button(key="run_ner").click().run()
    tables = [d.value for d in at.dataframe if "Entity Type" in d.value]
    df = tables[0] if tables else None
    check(f"{language}: NER ran with {model}",
          not at.exception and df is not None
          and all(s.startswith(model) for s in df["Source"]),
          {"exceptions": [e.value for e in at.exception],
           "sources": None if df is None else set(df["Source"])})
    if df is None:
        continue
    print(df[["Entity", "Entity Type", "Start", "End", "Source"]].to_string())
    found = list(zip(df["Entity"], df["Entity Type"]))
    check(f"{language}: multi-word / expected entities found as single spans {expected}",
          all(x in found for x in expected), found)
    check(f"{language}: labels are PER/LOC/ORG only", set(df["Entity Type"]) <= app.VALID_LABELS)
    check(f"{language}: offsets match the sentence",
          all(sentence[s:e] == t for s, e, t in zip(df["Start"], df["End"], df["Entity"])))

section("Alias review (baseline model)")

text = "प्रिया एक अंतरराष्ट्रीय सम्मेलन के लिए जिनेवा गई।"
ents = app.extract_ner(text, 0.30, app.MODEL_NAME, "Auto Detect")
print("  ", [(e["Entity"], e["Entity Type"], e["Source"]) for e in ents])
alias_hits = {e["Entity"]: e for e in ents if e["Entity"] in ("प्रिया", "जिनेवा")}
check("both configured alias spellings are reported",
      set(alias_hits) == {"प्रिया", "जिनेवा"}, list(alias_hits))
check("each alias hit is either alias-only (review, 1.0) or model + alias (model confidence)",
      all((e["Source"] == "Configured alias — review" and e["Confidence"] == 1.0)
          or (e["Source"].endswith("+ configured alias") and e["Confidence"] < 1.0)
          for e in alias_hits.values()),
      [(e["Entity"], e["Source"], e["Confidence"]) for e in alias_hits.values()])

section("Section 04: bilingual comparison")

for second, sentence in [("Hindi", CASES["Hindi"][0]), ("Marathi", CASES["Marathi"][0])]:
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.segmented_control(key="bilingual_pair").set_value(f"English → {second}")
    at.selectbox(key="bilingual_engine").set_value(app.TRAINED_ENGINE).run()
    at.text_area(key="bilingual_english").set_value(CASES["English"][0])
    at.text_area(key="bilingual_second").set_value(sentence)

    check(f"{second}: boxes labelled English / {second}",
          at.text_area(key="bilingual_english").label == "English Sentence"
          and at.text_area(key="bilingual_second").label == f"{second} Sentence")

    at.button(key="run_bilingual").click().run()
    tables = [d.value for d in at.dataframe if "Source" in d.value]
    sources = set().union(*[set(t["Source"]) for t in tables]) if tables else set()
    check(f"{second}: English BiLSTM-CRF on the English side, {second} BiLSTM-CRF on the other",
          not at.exception
          and any(s.startswith("English BiLSTM-CRF") for s in sources)
          and any(s.startswith(f"{second} BiLSTM-CRF") for s in sources),
          {"exceptions": [e.value for e in at.exception], "sources": sources})
    check(f"{second}: CSV export offered",
          any(b.label == "Download Entities (CSV)" for b in at.get("download_button")))

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
