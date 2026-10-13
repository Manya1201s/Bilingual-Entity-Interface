"""
Final demo test for Indic Entity Lab, run through the real Streamlit app
(streamlit.testing AppTest executes app.py exactly as `streamlit run`
does, including buttons, file uploads, session state and messages).

Run from the project folder:

    .venv\\Scripts\\python tests\\test_demo.py

Covers: the mixed English + Hindi + Marathi demo paragraph, Marathi
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


def run_typed(text, threshold=None):
    """Type text, click Prepare Sentences, then Extract Entities (like a user).
    Streamlit messages only live for one run, so the Process-step
    messages are captured before NER runs."""
    at = new_app()
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
print("\n=== Demo paragraph through the Streamlit UI")

at = run_typed(DEMO_TEXT)
check("no exception / error box", not at.exception and not at.error,
      [e.value for e in at.exception] + [e.value for e in at.error])
check("Process input: 6 sentences extracted",
      any("Extracted 6 sentences" in m for m in at.process_messages),
      at.process_messages)
check("Extract Entities: 'Entity extraction complete'",
      any("Entity extraction complete" in s.value for s in at.success))

df = results_table(at)
print(df[["Sentence No.", "Entity", "Entity Type", "Confidence",
          "Start", "End", "Source"]].to_string())

check("exactly the expected 13 entities, nothing extra",
      sorted(df["Entity"]) == sorted(e for e, _, _ in EXPECTED),
      list(df["Entity"]))

for entity, label, sources in EXPECTED:
    rows = df[df["Entity"] == entity]
    if len(rows) != 1:
        check(f"{entity}: appears exactly once", False, len(rows))
        continue
    row = rows.iloc[0]
    span_ok = row["Sentence"][row["Start"]:row["End"]] == entity
    check(f"{entity} -> {label} | span [{row['Start']},{row['End']}] | "
          f"conf {row['Confidence']} | {row['Source']}",
          row["Entity Type"] == label and span_ok
          and 0.5 <= row["Confidence"] <= 1.0 and row["Source"] in sources,
          row.to_dict())

overlap = False
for _, group in df.groupby("Sentence No."):
    spans = sorted(zip(group["Start"], group["End"]))
    overlap |= any(a[1] > b[0] for a, b in zip(spans, spans[1:]))
check("no overlapping spans within any sentence", not overlap)
check("no duplicate rows", not df.duplicated(["Sentence No.", "Start", "End"]).any())
check("sentences displayed unchanged (original Devanagari kept)",
      "सचिन तेंडुलकर मुंबईत गेले आणि जिनेव्हा येथे परिषद झाली." in list(df["Sentence"]))
check("metrics show PER 4, LOC 6, ORG 3, Total 13",
      [m.value for m in at.metric if m.label in ("PER", "LOC", "ORG", "Total Entities")]
      == ["4", "6", "3", "13"],
      [(m.label, m.value) for m in at.metric])
legend = [e for e in at.expander if e.label == "How to read the results"]
check("legend explaining labels / confidence / source is shown",
      legend and any("Confidence" in m.value and "Source" in m.value
                     for m in legend[0].markdown),
      [e.label for e in at.expander])

# ------------------------------------------------------------------
print("\n=== Marathi suffixes on मुंबई (one sentence per form, via the UI)")

FORMS = ["मुंबई", "मुंबईत", "मुंबईमध्ये", "मुंबईला", "मुंबईच्या", "मुंबईची",
         "मुंबईचे", "मुंबईवर", "मुंबईपासून", "मुंबईपर्यंत"]
at = run_typed("\n".join(f"आम्ही {form} गेलो." for form in FORMS))
df = results_table(at)
for number, form in enumerate(FORMS, start=1):
    rows = df[df["Sentence No."] == number]
    ok = (len(rows) == 1 and rows.iloc[0]["Entity"] == "मुंबई"
          and rows.iloc[0]["Entity Type"] == "LOC"
          and rows.iloc[0]["Sentence"] == f"आम्ही {form} गेलो.")
    suffix = form[len("मुंबई"):] or "(none)"
    check(f"{form:12} -> entity 'मुंबई' LOC, suffix '{suffix}' excluded, "
          "sentence unchanged", ok, rows.to_dict("records"))

# ------------------------------------------------------------------
print("\n=== Full names with titles / initials")

NAMES = [
    ("Dr. A. P. J. Abdul Kalam visited Delhi.", "A. P. J. Abdul Kalam"),
    ("Dr. Priya Mehta visited Geneva.", "Priya Mehta"),
    ("Mr. Narendra Modi met the press.", "Narendra Modi"),
    ("Prof. Sachin Tendulkar spoke in Mumbai.", "Sachin Tendulkar"),
]
at = run_typed("\n".join(s for s, _ in NAMES))
check("4 lines stay 4 sentences (titles/initials not split)",
      any("Extracted 4 sentences" in m for m in at.process_messages),
      at.process_messages)
df = results_table(at)
for number, (sentence, name) in enumerate(NAMES, start=1):
    rows = df[(df["Sentence No."] == number) & (df["Entity Type"] == "PER")]
    check(f"{name} -> PER, found by the model (not an alias)",
          list(rows["Entity"]) == [name]
          and list(rows["Source"]) == [MODEL], rows.to_dict("records"))

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
