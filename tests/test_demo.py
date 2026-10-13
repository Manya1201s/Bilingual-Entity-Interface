"""
Final demo test for Indic Entity Lab, run through the real Streamlit app
(streamlit.testing AppTest executes app.py exactly as `streamlit run`
does, including buttons, file uploads, session state and messages).

Run from the project folder:

    .venv\\Scripts\\python tests\\test_demo.py

Covers: the English + Hindi + Marathi demo paragraph (one language at a
time in the monolingual UI, all at once in the multilingual model), Marathi
suffixes on मुंबई, full names with titles/initials, and user-facing
error messages. Exit code is 0 only if every check passes.
"""

import os
import sys
import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
sys.stdout.reconfigure(encoding="utf-8")

from streamlit.testing.v1 import AppTest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
APP_PATH = os.path.join(os.path.dirname(HERE), "app.py")
sys.path.insert(0, os.path.dirname(HERE))
SAMPLES = os.path.join(HERE, "samples")

DEMO_TEXT = (
    "Dr. Priya Mehta visited Geneva. She met Satya Nadella of Microsoft "
    "and engineers from Infosys.\n"
    "नरेंद्र मोदी दिल्ली में एक बैठक में शामिल हुए। भारतीय रिजर्व बैंक के "
    "अधिकारी भी वहाँ मौजूद थे।\n"
    "सचिन तेंडुलकर मुंबईत गेले आणि जिनेव्हा येथे परिषद झाली.\n"
    "जेनेवा और जिनेवा एक ही शहर के दो हिंदी नाम हैं।"
)

MODEL = "Multilingual NER model"
BOTH = "Multilingual NER model + configured alias"
ALIAS = "Configured alias — review"

# (entity, label, allowed sources)
EXPECTED = [
    ("Priya Mehta", "PER", {MODEL}),
    ("Geneva", "LOC", {BOTH}),
    ("Satya Nadella", "PER", {MODEL}),
    ("Microsoft", "ORG", {MODEL}),
    ("Infosys", "ORG", {MODEL}),
    ("नरेंद्र मोदी", "PER", {MODEL}),
    ("दिल्ली", "LOC", {MODEL}),
    ("भारतीय रिजर्व बैंक", "ORG", {MODEL}),
    ("सचिन तेंडुलकर", "PER", {MODEL}),
    ("मुंबई", "LOC", {MODEL}),
    ("जिनेव्हा", "LOC", {ALIAS, BOTH}),
    ("जेनेवा", "LOC", {ALIAS, BOTH}),
    ("जिनेवा", "LOC", {ALIAS, BOTH}),
]

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


def new_app():
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    return at


def run_typed(text, threshold=None, language=None):
    """Type text, click Prepare Sentences, then Extract Entities (like a user).
    Streamlit messages only live for one run, so the Process-step
    messages are captured before NER runs."""
    at = new_app()
    if language is not None:
        at.selectbox(key="ner_language").set_value(language).run()
    at.text_area(key="manual_text").set_value(text)
    at.button(key="process_input").click().run()
    at.process_messages = [s.value for s in at.success] + [w.value for w in at.warning]
    if threshold is not None:
        at.slider(key="ner_threshold").set_value(threshold)
    at.button(key="run_ner").click().run()
    return at


def results_table(at):
    tables = [d.value for d in at.dataframe if "Entity Type" in d.value]
    return tables[0] if tables else None


# ------------------------------------------------------------------
# The Monolingual NER Extraction workflow uses the language-specific
# BiLSTM-CRF of the selected input language, so the mixed demo paragraph
# is entered one language at a time, as a user would. The multilingual
# model (still used by the bilingual workflow) is checked on the whole
# mixed paragraph directly.

import app  # noqa: E402  (bare mode; extract_ner and constants only)

DEMO_PARTS = {
    "English": ("Dr. Priya Mehta visited Geneva. She met Satya Nadella of Microsoft "
                "and engineers from Infosys.", 2),
    "Hindi": ("नरेंद्र मोदी दिल्ली में एक बैठक में शामिल हुए। भारतीय रिजर्व बैंक के "
              "अधिकारी भी वहाँ मौजूद थे।\n"
              "जेनेवा और जिनेवा एक ही शहर के दो हिंदी नाम हैं।", 3),
    "Marathi": ("सचिन तेंडुलकर मुंबईत गेले आणि जिनेव्हा येथे परिषद झाली.", 1),
}
LANGUAGE_OF = {
    "Priya Mehta": "English", "Geneva": "English", "Satya Nadella": "English",
    "Microsoft": "English", "Infosys": "English", "नरेंद्र मोदी": "Hindi",
    "दिल्ली": "Hindi", "भारतीय रिजर्व बैंक": "Hindi", "जेनेवा": "Hindi",
    "जिनेवा": "Hindi", "सचिन तेंडुलकर": "Marathi", "मुंबई": "Marathi",
    "जिनेव्हा": "Marathi",
}
MODELS = {"English": "English BiLSTM-CRF (scratch)", "Hindi": "Hindi BiLSTM-CRF (scratch)",
          "Marathi": "Marathi BiLSTM-CRF (scratch)"}

print("\n=== Demo paragraph through the Streamlit UI (one language at a time)")

found_ui = {}
for language, (text, sentence_count) in DEMO_PARTS.items():
    at = run_typed(text, language=language)
    check(f"{language}: no exception / error box", not at.exception and not at.error,
          [e.value for e in at.exception] + [e.value for e in at.error])
    check(f"{language}: Prepare Sentences -> {sentence_count} sentences",
          any(f"Extracted {sentence_count} sentence" in m for m in at.process_messages),
          at.process_messages)
    check(f"{language}: 'Entity extraction complete'",
          any("Entity extraction complete" in s.value for s in at.success))
    df = results_table(at)
    print(df[["Sentence No.", "Entity", "Entity Type", "Confidence",
              "Start", "End", "Source"]].to_string())
    check(f"{language}: every entity comes from the {language} model (alias may confirm)",
          all(s.startswith(MODELS[language]) or s == ALIAS for s in df["Source"]),
          set(df["Source"]))
    check(f"{language}: spans are exact, no overlaps, no duplicates",
          all(s[a:b] == e for s, a, b, e in zip(df["Sentence"], df["Start"], df["End"], df["Entity"]))
          and not df.duplicated(["Sentence No.", "Start", "End"]).any())
    for entity, entity_type in zip(df["Entity"], df["Entity Type"]):
        found_ui.setdefault(entity, entity_type)
    if language == "Marathi":
        check("sentences displayed unchanged (original Devanagari kept)",
              "सचिन तेंडुलकर मुंबईत गेले आणि जिनेव्हा येथे परिषद झाली." in list(df["Sentence"]))
    legend = [e for e in at.expander if e.label == "How to read the results"]
    check(f"{language}: legend explaining labels / confidence / source is shown",
          legend and any("Confidence" in m.value and "Source" in m.value
                         for m in legend[0].markdown),
          [e.label for e in at.expander])

for entity, label, _ in EXPECTED:
    language = LANGUAGE_OF[entity]
    # The Marathi model keeps attached case suffixes (मुंबईत), see below.
    hits = [e for e, t in found_ui.items() if t == label and e.startswith(entity)]
    check(f"{entity} -> {label} found by the {language} model", bool(hits), found_ui)

print("\n=== Demo paragraph through the multilingual model (function level)")

rows = [dict(e, Sentence=s) for s in app.split_sentences(DEMO_TEXT)
        for e in app.extract_ner(s, 0.30, app.MODEL_NAME, "Auto Detect")]
check("multilingual model: exactly the expected 13 entities, nothing extra",
      sorted(r["Entity"] for r in rows) == sorted(e for e, _, _ in EXPECTED),
      [r["Entity"] for r in rows])
for entity, label, sources in EXPECTED:
    hits = [r for r in rows if r["Entity"] == entity]
    check(f"multilingual: {entity} -> {label} from {sorted(sources)}",
          len(hits) == 1 and hits[0]["Entity Type"] == label
          and hits[0]["Sentence"][hits[0]["Start"]:hits[0]["End"]] == entity
          and 0.5 <= hits[0]["Confidence"] <= 1.0 and hits[0]["Source"] in sources,
          hits)

# ------------------------------------------------------------------
print("\n=== Marathi suffixes on मुंबई")

FORMS = ["मुंबई", "मुंबईत", "मुंबईमध्ये", "मुंबईला", "मुंबईच्या", "मुंबईची",
         "मुंबईचे", "मुंबईवर", "मुंबईपासून", "मुंबईपर्यंत"]
for form in FORMS:
    sentence = f"आम्ही {form} गेलो."
    ents = app.extract_ner(sentence, 0.30, app.MODEL_NAME, "Auto Detect")
    suffix = form[len("मुंबई"):] or "(none)"
    check(f"multilingual: {form:12} -> 'मुंबई' LOC, suffix '{suffix}' excluded",
          [(e["Entity"], e["Entity Type"]) for e in ents] == [("मुंबई", "LOC")], ents)

at = run_typed("\n".join(f"आम्ही {form} गेलो." for form in FORMS), language="Marathi")
df = results_table(at)
for number, form in enumerate(FORMS, start=1):
    rows = df[df["Sentence No."] == number]
    ok = (len(rows) == 1 and rows.iloc[0]["Entity"].startswith("मुंबई")
          and rows.iloc[0]["Entity Type"] == "LOC"
          and rows.iloc[0]["Sentence"] == f"आम्ही {form} गेलो.")
    check(f"monolingual UI: {form:12} -> one LOC entity for मुंबई, sentence unchanged",
          ok, rows.to_dict("records"))
print("  OBSERVED monolingual (Marathi BiLSTM-CRF) entities, not asserted: "
      f"{list(df['Entity'])} - the trained model keeps attached case suffixes")

# ------------------------------------------------------------------
print("\n=== Full names with titles / initials")

NAMES = [
    ("Dr. A. P. J. Abdul Kalam visited Delhi.", "A. P. J. Abdul Kalam"),
    ("Dr. Priya Mehta visited Geneva.", "Priya Mehta"),
    ("Mr. Narendra Modi met the press.", "Narendra Modi"),
    ("Prof. Sachin Tendulkar spoke in Mumbai.", "Sachin Tendulkar"),
]
at = run_typed("\n".join(s for s, _ in NAMES), language="English")
check("4 lines stay 4 sentences (titles/initials not split)",
      any("Extracted 4 sentences" in m for m in at.process_messages),
      at.process_messages)
df = results_table(at)
for number, (sentence, name) in enumerate(NAMES, start=1):
    rows = df[(df["Sentence No."] == number) & (df["Entity Type"] == "PER")]
    check(f"monolingual UI: {name} -> one PER from the English model, title excluded",
          len(rows) == 1 and name.endswith(rows.iloc[0]["Entity"])
          and not rows.iloc[0]["Entity"].startswith(("Dr", "Mr", "Prof"))
          and rows.iloc[0]["Source"] == MODELS["English"], rows.to_dict("records"))
    ents = [e for e in app.extract_ner(sentence, 0.30, app.MODEL_NAME, "Auto Detect")
            if e["Entity Type"] == "PER"]
    check(f"multilingual: {name} -> PER, full name, found by the model (not an alias)",
          [e["Entity"] for e in ents] == [name] and [e["Source"] for e in ents] == [MODEL], ents)
print("  OBSERVED monolingual PER entities, not asserted: "
      f"{list(df[df['Entity Type'] == 'PER']['Entity'])}")

# ------------------------------------------------------------------
print("\n=== Error / empty-input messages (no tracebacks)")

MIME = {"txt": "text/plain", "csv": "text/csv", "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}


def upload(name, content=None):
    at = new_app()
    if content is None:
        with open(os.path.join(SAMPLES, name), "rb") as f:
            content = f.read()
    mime = MIME.get(name.rsplit(".", 1)[-1], "application/octet-stream")
    at.file_uploader[0].upload(name, content, mime)
    at.run()
    at.button(key="process_input").click().run()
    return at


def messages(at):
    return ([e.value for e in at.error] + [w.value for w in at.warning]
            + [s.value for s in at.success])


def no_traceback(at):
    text = " ".join(messages(at))
    return not at.exception and "Traceback" not in text


at = new_app()
at.button(key="process_input").click().run()
check("empty input -> 'No readable text was found' warning",
      no_traceback(at) and any("No readable text was found" in m for m in messages(at)),
      messages(at))

for name, expected in [
    ("empty.txt", "No readable text was found"),
    ("empty.csv", "No readable text was found"),
    ("empty.docx", "No readable text was found"),
    ("corrupt.docx", "The DOCX file could not be opened"),
    ("corrupt.pdf", "The PDF file could not be read"),
    ("empty.pdf", "The PDF file could not be read"),
    ("other_column.csv", "The text column could not be detected automatically"),
    ("malformed.csv", "The CSV file could not be read"),
    ("bad_encoding.txt", "Extracted 2 sentences"),
    ("cp1252_english.txt", "Extracted 1 sentence from"),
]:
    at = upload(name)
    found = [m for m in messages(at) if expected in m]
    check(f"{name:20} -> '{expected}'", no_traceback(at) and found, messages(at))
    if name in ("corrupt.docx", "corrupt.pdf"):
        check(f"{name:20} -> message has no Python class name prefix",
              not any("ValueError:" in m for m in found), found)

at = run_typed("आज हवामान छान आहे. It is a sunny day.")
check("no entities -> 'No entities were detected' warning",
      no_traceback(at) and any("No entities were detected" in w.value for w in at.warning),
      messages(at))

# ------------------------------------------------------------------
print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
