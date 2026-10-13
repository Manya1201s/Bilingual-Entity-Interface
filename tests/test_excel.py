"""
Excel upload tests (.xlsx via openpyxl, .xls via xlrd) for app.py.

Run from the project folder with the Python 3.12 environment:

    .venv312\\Scripts\\python tests\\test_excel.py

Fixtures in tests/samples: bilingual.xlsx / bilingual.xls (sheets
"Sentences": ID, English, Hindi, Marathi with a missing cell and a blank
row; "Notes": Text, Count with blank and number-only cells; "Empty"),
empty.xlsx, corrupt.xlsx, corrupt.xls.
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


def raw(name):
    with open(os.path.join(SAMPLES, name), "rb") as f:
        return f.read()


MIME = {"xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel"}

# ------------------------------------------------------------------
section("Reading workbooks")

for name in ("bilingual.xlsx", "bilingual.xls"):
    sheets = app.read_excel_bytes(raw(name), name)
    check(f"{name}: all three sheets read, headers from row 1",
          list(sheets) == ["Sentences", "Notes", "Empty"]
          and list(sheets["Sentences"].columns) == ["ID", "English", "Hindi", "Marathi"],
          {n: list(df.columns) for n, df in sheets.items()})

    cells, stats = app.excel_cells(sheets)
    texts = [t for t, _ in cells]
    check(f"{name}: all sheets + all text columns -> 10 text cells from 5 rows in 2 sheets",
          len(cells) == 10 and stats == {"sheets": 2, "rows": 5}, (len(cells), stats))
    check(f"{name}: blank cells, the blank row and number-only cells (ID, Count) are skipped",
          all(t.strip() and any(ch.isalpha() for ch in t) for t in texts)
          and "1" not in texts and "7" not in texts, texts)
    check(f"{name}: locations keep sheet, Excel cell and header",
          ("Manya lives in NOIDA", "Sentences!B2 (English)") in cells
          and ("मान्या नोएडा में रहती है।", "Sentences!C2 (Hindi)") in cells
          and ("आरव शर्मा पुण्यात इन्फोसिसमध्ये काम करतात।", "Sentences!D5 (Marathi)") in cells
          and ("Vikram Patel met Priya in Delhi.", "Notes!A2 (Text)") in cells, cells[:4])

    cells, stats = app.excel_cells(sheets, "Sentences", "Marathi")
    check(f"{name}: one sheet + one column; the missing Marathi cell is skipped",
          [loc for _, loc in cells] == ["Sentences!D2 (Marathi)", "Sentences!D5 (Marathi)"]
          and stats == {"sheets": 1, "rows": 2}, (cells, stats))

    check(f"{name}: read_file returns the same text joined (shared upload path)",
          "Manya lives in NOIDA" in app.read_file(type("U", (), {
              "name": name, "getvalue": lambda self, b=raw(name): b})()))

check("default text column: a 'Text' column is chosen, otherwise all text columns",
      app.default_text_column(["Text", "Count"]) == "Text"
      and app.default_text_column(["English", "Hindi"]) == app.ALL_COLUMNS)
check("column letters: A, Z, AA, AZ",
      [app.column_letter(i) for i in (0, 25, 26, 51)] == ["A", "Z", "AA", "AZ"])

check("empty.xlsx: sheet read, no cells",
      app.excel_cells(app.read_excel_bytes(raw("empty.xlsx"), "empty.xlsx"))[0] == [])
for name in ("corrupt.xlsx", "corrupt.xls"):
    try:
        app.read_excel_bytes(raw(name), name)
        message = None
    except ValueError as exc:
        message = str(exc)
    check(f"{name}: clear error, no traceback",
          message == "The Excel file could not be read. It may be damaged, "
                     "password-protected, or not a real Excel workbook.", message)

# ------------------------------------------------------------------
section("Upload Document tab (real app)")


def upload(name, sheet=None, column=None):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.file_uploader[0].upload(name, raw(name), MIME[name.rsplit(".", 1)[1]])
    at.run()
    if sheet is not None:
        at.selectbox(key="excel_sheet").set_value(sheet).run()
    if column is not None:
        at.selectbox(key=f"excel_text_column_{sheet or app.ALL_SHEETS}").set_value(column).run()
    return at


for name in ("bilingual.xlsx", "bilingual.xls"):
    at = upload(name)
    check(f"{name}: Sheet and Text Column selectors shown (All sheets / All text columns)",
          not at.exception
          and at.selectbox(key="excel_sheet").options == [app.ALL_SHEETS, "Sentences", "Notes"]
          and at.selectbox(key=f"excel_text_column_{app.ALL_SHEETS}").value == app.ALL_COLUMNS,
          [e.value for e in at.exception])
    check(f"{name}: empty sheet reported as skipped",
          any("1 empty sheet skipped" in c.value for c in at.caption))

    at.button(key="process_input").click().run()
    check(f"{name}: success message counts sentences, rows and sheets",
          any(s.value == f"Extracted 11 sentences from 5 rows in 2 sheets of {name}."
              for s in at.success), [s.value for s in at.success] + [e.value for e in at.error])

    at = upload(name, sheet="Sentences", column="English")
    at.selectbox(key="ner_language").set_value("English").run()
    at.button(key="process_input").click().run()
    check(f"{name}: one sheet + English column -> 3 sentences from 3 rows in 1 sheet",
          any(s.value == f"Extracted 3 sentences from 3 rows in 1 sheet of {name}."
              for s in at.success), [s.value for s in at.success])
    preview = next(d.value for d in at.dataframe if "Sentence" in d.value.columns) \
        if at.dataframe else None
    at.button(key="run_ner").click().run()
    tables = [d.value for d in at.dataframe if "Entity Type" in d.value.columns]
    ner = tables[0] if tables else None
    check(f"{name}: NER runs on the spreadsheet sentences and keeps their cell location",
          not at.exception and ner is not None and "Location" in ner.columns
          and set(ner["Location"]) <= {"Sentences!B2 (English)", "Sentences!B3 (English)",
                                       "Sentences!B5 (English)"}
          and {"Manya", "NOIDA"} <= set(ner["Entity"]),
          None if ner is None else ner[["Location", "Entity", "Entity Type"]].to_dict("records"))
    if ner is not None:
        check(f"{name}: entity offsets still refer to the sentence",
              all(s[a:b] == e for s, a, b, e in zip(ner["Sentence"], ner["Start"], ner["End"],
                                                     ner["Entity"])))

    at = upload(name, sheet="Notes")
    check(f"{name}: 'Text' column auto-selected on the Notes sheet",
          at.selectbox(key="excel_text_column_Notes").value == "Text")
    at.button(key="process_input").click().run()
    check(f"{name}: multi-sentence cell split into sentences, blank and number cells skipped",
          any(s.value == f"Extracted 3 sentences from 2 rows in 1 sheet of {name}."
              for s in at.success), [s.value for s in at.success])

for name, expected in (("corrupt.xlsx", "The Excel file could not be previewed."),
                       ("corrupt.xls", "The Excel file could not be previewed.")):
    at = upload(name)
    check(f"{name}: friendly error, no exception",
          not at.exception and any(e.value.startswith(expected) for e in at.error),
          [e.value for e in at.error] + [e.value for e in at.exception])

at = upload("empty.xlsx")
check("empty.xlsx: 'no data rows' notice", any("no data rows" in i.value for i in at.info))
at.button(key="process_input").click().run()
check("empty.xlsx: processing reports no readable text",
      not at.exception and any("No readable text was found" in w.value for w in at.warning))

# ------------------------------------------------------------------
section("Bilingual spreadsheet -> Entity Alignment (document input method)")

EXPECTED_INDIC = {"Hindi": "मान्या नोएडा में रहती है।", "Marathi": "मान्या नोएडामध्ये राहते."}


def align_upload(name, pair):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.segmented_control(key="bilingual_pair").set_value(pair)
    at.segmented_control(key="align_input_method").set_value("Upload a Document").run()
    at.file_uploader(key="align_document").upload(name, raw(name), MIME[name.rsplit(".", 1)[1]])
    at.run()
    return at


for name in ("bilingual.xlsx", "bilingual.xls"):
    for pair, indic in (("English → Hindi", "Hindi"), ("Marathi → English", "Marathi")):
        at = align_upload(name, pair)
        check(f"{name} / {pair}: sheet and language columns selectable (guessed from headers)",
              at.selectbox(key="align_sheet").options == ["Sentences", "Notes"]
              and at.selectbox(key="align_english_Sentences").value == "English"
              and at.selectbox(key=f"align_indic_{indic}_Sentences").value == indic)
        at.button(key="run_bilingual").click().run()
        shown = [h.value for h in at.subheader]
        table = next((d.value for d in at.dataframe if "Status" in d.value.columns), None)
        got = [] if table is None else list(zip(table["English Entity"], table[f"{indic} Entity"],
                                                table["Status"]))
        sentences = [m.value for m in at.markdown]
        check(f"{name} / {pair}: alignment runs on row 2 of the spreadsheet (pair order kept)",
              not at.exception and ("Manya", "मान्या", "Matched") in got
              and shown[:2] == [f"{l} Entities" for l in app.LANGUAGE_PAIRS[pair]]
              and "Manya lives in NOIDA" in sentences
              and app.escape_markdown(EXPECTED_INDIC[indic]) in sentences, (got, shown))

at = align_upload("bilingual.xlsx", "English → Marathi")
at.number_input(key="align_row_Sentences").set_value(3).run()
check("row with a missing Marathi cell -> warning before running",
      any("Row 3 has an empty English or Marathi cell." in w.value for w in at.warning))
at.button(key="run_bilingual").click().run()
check("...and Extract and Align does not run",
      not any("Status" in d.value.columns for d in at.dataframe)
      and not any("Entities extracted" in s.value for s in at.success))

at = align_upload("bilingual.xlsx", "English → Hindi")
at.selectbox(key="align_sheet").set_value("Notes").run()
at.selectbox(key="align_english_Notes").set_value("Text")
at.selectbox(key="align_indic_Hindi_Notes").set_value("Text").run()
check("the same column for both languages is rejected",
      any("Choose two different columns" in w.value for w in at.warning))

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
