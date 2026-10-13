"""
Input-source tests: an uploaded document always wins over typed text.

Run from the project folder with the Python 3.12 environment:

    .venv312\\Scripts\\python tests\\test_input_source.py

Covers app sections 01-03 (Prepare Sentences / Extract Entities) and 04
(Extract and Align Entities). Document content is passed in memory (TXT
and CSV bytes) or taken from tests/samples (Excel, DOCX).
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
SAMPLES = os.path.join(HERE, "samples")
APP_PATH = os.path.join(PROJECT, "app.py")

sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd  # noqa: E402
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


MANUAL = "Ravi Kumar lives in Chennai."
DOC = "Priya Mehta visited Geneva. Aarav Sharma works at Infosys in Pune."
DOC_SENTENCES = ["Priya Mehta visited Geneva.", "Aarav Sharma works at Infosys in Pune."]
NO_DOC_TEXT = "No readable text was found in the uploaded document. Please upload a valid document."
MIME = {"txt": "text/plain", "csv": "text/csv", "docx":
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"}


def sample(name):
    with open(os.path.join(SAMPLES, name), "rb") as f:
        return f.read()


def new_app(manual=None, upload=None):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    if manual is not None:
        at.text_area(key="manual_text").set_value(manual)
    if upload is not None:
        name, content = upload
        at.file_uploader[0].upload(name, content, MIME[name.rsplit(".", 1)[1]])
    at.run()
    return at


def prepared(at):
    """Sentences shown in 02 · Sentence Preview ([] if the section is absent)."""
    if "02 · Sentence Preview" not in [h.value for h in at.header]:
        return []
    texts = [m.value for m in at.markdown]
    count = int(next(m.value for m in at.metric if m.label == "Sentences"))
    starts = [i for i, t in enumerate(texts) if t.startswith("**Sentence ")]
    return [texts[i + 1].replace("\\", "") for i in starts[:count]]


def captions(at):
    return [c.value for c in at.caption]


# ------------------------------------------------------------------
section("The input-source rule (deterministic)")

check("no document -> manual", app.input_source(None) == "manual")
check("any uploaded document -> document", app.input_source(object()) == "document")

sheets = {"S": pd.DataFrame({"English": ["Manya lives in NOIDA", "Ravi"],
                             "Hindi": ["मान्या नोएडा में रहती है।", float("nan")]})}
sides, problem = app.spreadsheet_pair(sheets, "S", "English", "Hindi", 2, ("Hindi", "English"))
check("spreadsheet_pair keeps the selected pair order (Hindi → English)",
      sides == [("Hindi", "मान्या नोएडा में रहती है।"), ("English", "Manya lives in NOIDA")]
      and problem is None, (sides, problem))
sides, problem = app.spreadsheet_pair(sheets, "S", "English", "Hindi", 3, ("English", "Hindi"))
check("spreadsheet_pair: a missing cell (NaN) is reported, nothing is filled in",
      sides is None and problem.startswith("Row 3 has an empty English or Hindi cell"), problem)

# ------------------------------------------------------------------
section("Sections 01-03")

at = new_app(manual=MANUAL)
check("no document: source shown as manual sentences",
      "Input source: Manual sentences" in captions(at))
at.button(key="process_input").click().run()
check("no document: the typed text is processed",
      prepared(at) == [MANUAL] and any("from the text box" in s.value for s in at.success),
      prepared(at))

at = new_app(manual=MANUAL, upload=("doc.txt", DOC.encode()))
check("document + typed text: source shown as the uploaded document",
      "Input source: Uploaded document (doc.txt)" in captions(at), captions(at))
check("document + typed text: notice that the text box is ignored",
      any("text in the text box is ignored" in i.value for i in at.info))
at.button(key="process_input").click().run()
check("document + typed text: only the document's sentences are prepared",
      prepared(at) == DOC_SENTENCES
      and any(s.value == "Extracted 2 sentences from doc.txt." for s in at.success),
      prepared(at))
check("typed text is kept in the text box (not erased)",
      at.text_area(key="manual_text").value == MANUAL)
at.button(key="run_ner").click().run()
ner = next((d.value for d in at.dataframe if "Entity Type" in d.value.columns), None)
check("entity extraction runs on document sentences only; nothing from the typed text",
      ner is not None and set(ner["Sentence"]) <= set(DOC_SENTENCES)
      and not {"Ravi Kumar", "Chennai"} & set(ner["Entity"]),
      None if ner is None else ner[["Sentence", "Entity"]].to_dict("records"))

for name, content in (("empty.txt", b""), ("blank.txt", b"   \n\n  ")):
    at = new_app(manual=MANUAL, upload=(name, content))
    at.button(key="process_input").click().run()
    check(f"{name} + typed text: clear message, no fallback to the typed text",
          any(w.value == NO_DOC_TEXT for w in at.warning) and prepared(at) == [],
          ([w.value for w in at.warning], prepared(at)))

at = new_app(manual=MANUAL, upload=("corrupt.docx", sample("corrupt.docx")))
at.button(key="process_input").click().run()
check("unreadable document + typed text: error, no fallback, no sentences",
      any(e.value.startswith("The input could not be processed") for e in at.error)
      and prepared(at) == [], [e.value for e in at.error])

at = new_app(manual=MANUAL)
at.button(key="process_input").click().run()
at.button(key="run_ner").click().run()
at.file_uploader[0].upload("empty.txt", b"", "text/plain")
at.run()
check("uploading a document clears sentences and results prepared from the typed text",
      prepared(at) == [] and not any("Entity Type" in d.value.columns for d in at.dataframe))
at.button(key="process_input").click().run()
check("…and the empty document still does not fall back to them",
      prepared(at) == [] and any(w.value == NO_DOC_TEXT for w in at.warning))

at = new_app(manual=MANUAL, upload=("doc.txt", DOC.encode()))
at.button(key="process_input").click().run()
at.file_uploader[0].clear()
at.run()
check("removing the document returns to manual input and drops the document's sentences",
      "Input source: Manual sentences" in captions(at) and prepared(at) == [], captions(at))
at.button(key="process_input").click().run()
check("…and Prepare Sentences then uses the typed text", prepared(at) == [MANUAL], prepared(at))

at = new_app(manual=MANUAL, upload=("doc.txt", DOC.encode()))
at.button(key="process_input").click().run()
at.file_uploader[0].upload("other.txt", "Vikram Patel met Priya in Delhi.".encode(), "text/plain")
at.run()
check("replacing the document drops sentences prepared from the previous document",
      prepared(at) == [] and "Input source: Uploaded document (other.txt)" in captions(at))

at = new_app(manual=MANUAL, upload=("bilingual.xlsx", sample("bilingual.xlsx")))
at.button(key="process_input").click().run()
check("Excel + typed text: only spreadsheet content is prepared",
      len(prepared(at)) == 11 and MANUAL not in prepared(at)
      and "Manya lives in NOIDA" in prepared(at), prepared(at))

# ------------------------------------------------------------------
section("Section 04 (Entity Alignment): input method")

import io  # noqa: E402
from docx import Document  # noqa: E402

DOC_METHOD, MANUAL_METHOD = "Upload a Document", "Enter Sentences Manually"
PAIR_TEXT = "Manya lives in NOIDA. मान्या नोएडा में रहती है।"


def docx_bytes(paragraphs):
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def fill_manual_pair(at, english="Ravi Kumar lives in Chennai.",
                     hindi="रवि कुमार चेन्नई में रहते हैं।"):
    at.text_area(key="bilingual_english").set_value(english)
    at.text_area(key="bilingual_second").set_value(hindi)
    at.run()


def document_mode(at, upload=None, pair=None):
    if pair:
        at.segmented_control(key="bilingual_pair").set_value(pair)
    at.segmented_control(key="align_input_method").set_value(DOC_METHOD).run()
    if upload:
        name, content = upload
        at.file_uploader(key="align_document").upload(name, content, MIME.get(
            name.rsplit(".", 1)[1], "application/octet-stream"))
        at.run()
    return at


def alignment(at, indic="Hindi"):
    table = next((d.value for d in at.dataframe if "Status" in d.value.columns), None)
    return [] if table is None else list(zip(table["English Entity"], table[f"{indic} Entity"]))


def warnings_of(at):
    return [w.value for w in at.warning] + [e.value for e in at.error] + [i.value for i in at.info]


at = new_app()
check("Input Method offers manual and document input; manual is the default",
      at.segmented_control(key="align_input_method").options == [MANUAL_METHOD, DOC_METHOD]
      and at.segmented_control(key="align_input_method").value == MANUAL_METHOD)
fill_manual_pair(at, "Manya lives in NOIDA", "मान्या नोएडा में रहती है।")
at.button(key="run_bilingual").click().run()
check("manual mode: the typed sentence pair is aligned",
      ("Manya", "मान्या") in alignment(at)
      and "Source: Manually entered sentences" in captions(at), alignment(at))

at = new_app(upload=("bilingual.xlsx", sample("bilingual.xlsx")))
fill_manual_pair(at, "Manya lives in NOIDA", "मान्या नोएडा में रहती है।")
at.button(key="run_bilingual").click().run()
check("a document uploaded in 01 · Input does not change the alignment input",
      ("Manya", "मान्या") in alignment(at)
      and "Source: Manually entered sentences" in captions(at), captions(at))

at = new_app()
fill_manual_pair(at)
document_mode(at, ("bilingual.xlsx", sample("bilingual.xlsx")))
check("document mode hides the manual sentence fields",
      not any(t.key in ("bilingual_english", "bilingual_second") for t in at.text_area))
at.button(key="run_bilingual").click().run()
pairs = alignment(at)
check("document mode: alignment uses the document only, never the typed pair",
      ("Manya", "मान्या") in pairs and not any("Ravi" in (e or "") for e, _ in pairs)
      and "Source: bilingual.xlsx · Sentences, row 2" in captions(at), (pairs, captions(at)))

at = new_app()
fill_manual_pair(at)
document_mode(at)
at.button(key="run_bilingual").click().run()
check("document mode without a file: message, no fallback to the typed pair",
      any("Upload a document to align its sentences." in m for m in warnings_of(at))
      and alignment(at) == [], warnings_of(at))

for name, content, expected in (
        ("empty.txt", b"", "No readable text was found in the uploaded document"),
        ("one.txt", "Manya lives in NOIDA.".encode(), "does not contain enough sentences"),
        ("empty.xlsx", sample("empty.xlsx"), "No data rows were found in the uploaded document"),
        ("corrupt.xlsx", sample("corrupt.xlsx"), "The document could not be read"),
        ("corrupt.docx", sample("corrupt.docx"), "The document could not be read")):
    at = new_app()
    fill_manual_pair(at)
    document_mode(at, (name, content))
    at.button(key="run_bilingual").click().run()
    check(f"{name}: clear message, nothing aligned, no fallback to the typed pair",
          not at.exception and any(expected in m for m in warnings_of(at))
          and alignment(at) == [], warnings_of(at))

for name, content in (("pair.txt", PAIR_TEXT.encode()),
                      ("pair.docx", docx_bytes(["Manya lives in NOIDA.", "मान्या नोएडा में रहती है।"])),
                      ("pair.csv", "English,Hindi\nManya lives in NOIDA,मान्या नोएडा में रहती है।\n".encode())):
    at = new_app()
    document_mode(at, (name, content))
    at.button(key="run_bilingual").click().run()
    check(f"{name}: sentence pair extracted and aligned (Manya ↔ मान्या)",
          not at.exception and ("Manya", "मान्या") in alignment(at), (alignment(at), warnings_of(at)))

at = new_app()
document_mode(at, ("pair.txt", PAIR_TEXT.encode()), pair="Hindi → English")
check("TXT: sentences listed with their script; choices follow the pair (Hindi first)",
      at.selectbox(key="align_first_Hindi").value == 1
      and at.selectbox(key="align_second_English").value == 0
      and any("Script" in d.value.columns for d in at.dataframe))
at.button(key="run_bilingual").click().run()
check("TXT, Hindi → English: entity tables in pair order, alignment matched",
      [h.value for h in at.subheader][:2] == ["Hindi Entities", "English Entities"]
      and ("Manya", "मान्या") in alignment(at))
at.selectbox(key="align_first_Hindi").set_value(0).run()
check("TXT: the same sentence chosen for both languages is rejected",
      any("Choose two different sentences." in w for w in warnings_of(at)), warnings_of(at))
at.selectbox(key="align_second_English").set_value(1).run()
check("TXT: a Latin-script sentence chosen as Hindi is rejected",
      any("is not written in Devanagari script" in w for w in warnings_of(at)), warnings_of(at))
at.button(key="run_bilingual").click().run()
check("…and the invalid choice is not aligned", alignment(at) == [])

at = new_app()
document_mode(at, ("mixed.pdf", sample("mixed.pdf")))
check("PDF: text extracted and offered as numbered sentences (no exception)",
      not at.exception and any("Script" in d.value.columns for d in at.dataframe),
      [e.value for e in at.exception] + warnings_of(at))

at = new_app()
fill_manual_pair(at, "Manya lives in NOIDA", "मान्या नोएडा में रहती है।")
at.button(key="run_bilingual").click().run()
document_mode(at)
check("switching to document mode clears results of the manual pair", alignment(at) == [])
at.segmented_control(key="align_input_method").set_value(MANUAL_METHOD).run()
check("switching back to manual mode keeps the typed sentences",
      at.text_area(key="bilingual_english").value == "Manya lives in NOIDA"
      and at.text_area(key="bilingual_second").value == "मान्या नोएडा में रहती है।")
at.button(key="run_bilingual").click().run()
check("…and aligns them again", ("Manya", "मान्या") in alignment(at))

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
