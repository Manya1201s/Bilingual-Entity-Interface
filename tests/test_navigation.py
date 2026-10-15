"""
NER Extraction navigation tests (app.py): the "Monolingual NER Extraction"
and "Bilingual NER Extraction" cards open the matching existing workflow.

Run from the project folder with the Python 3.12 environment:

    .venv312\\Scripts\\python tests\\test_navigation.py

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


MONO, BI = "Monolingual NER Extraction", "Bilingual NER Extraction"


def nav_buttons(at):
    return {b.key: b for b in at.button if b.key in ("open_monolingual", "open_bilingual")}


def tab_headers(at, label):
    tab = next(t for t in at.tabs if t.label == label)
    return [h.value for h in tab.header]


print("\n=== Layout")
at = AppTest.from_file(APP_PATH, default_timeout=900)
at.run()
check("app starts without exceptions", not at.exception, [e.value for e in at.exception])
check("'NER Extraction' section comes first, before the workflows",
      [h.value for h in at.header][0] == "NER Extraction", [h.value for h in at.header])
buttons = nav_buttons(at)
check("two buttons: 'Open Monolingual NER Extraction' and 'Open Bilingual NER Extraction'",
      buttons.get("open_monolingual") is not None and buttons.get("open_bilingual") is not None
      and buttons["open_monolingual"].label == f"Open {MONO}"
      and buttons["open_bilingual"].label == f"Open {BI}", list(buttons))
labels = [t.label for t in at.tabs]
check("workflow tabs carry the same names (monolingual first)",
      MONO in labels and BI in labels and labels.index(MONO) < labels.index(BI), labels)
check("monolingual tab holds the existing sections 01 · Input (+ 02/03 after processing)",
      tab_headers(at, MONO) == ["01 · Input"], tab_headers(at, MONO))
check("bilingual tab holds the existing section 04 · Entity Alignment",
      tab_headers(at, BI) == ["04 · Entity Alignment"], tab_headers(at, BI))

print("\n=== Switching")
check("monolingual workflow is open by default (highlighted button)",
      at.session_state["ner_workflow"] == MONO
      and buttons["open_monolingual"].proto.type == "primary"
      and buttons["open_bilingual"].proto.type == "secondary")

at.text_area(key="manual_text").set_value("Manya lives in NOIDA.")
at.button(key="open_bilingual").click().run()
buttons = nav_buttons(at)
check("'Bilingual NER Extraction' opens the bilingual tab and highlights its button",
      not at.exception and at.session_state["ner_workflow"] == BI
      and buttons["open_bilingual"].proto.type == "primary"
      and buttons["open_monolingual"].proto.type == "secondary")

at.text_area(key="bilingual_english").set_value("Manya lives in NOIDA")
at.text_area(key="bilingual_second").set_value("मान्या नोएडा में रहती है।")
at.button(key="run_bilingual").click().run()
table = next((d.value for d in at.dataframe if "Status" in d.value.columns), None)
check("the existing bilingual workflow runs (Manya ↔ मान्या matched)",
      table is not None and ("Manya", "मान्या", "Matched") in list(
          zip(table["English Entity"], table["Hindi Entity"], table["Status"])))

at.button(key="open_monolingual").click().run()
check("'Monolingual NER Extraction' opens the monolingual tab again",
      at.session_state["ner_workflow"] == MONO
      and nav_buttons(at)["open_monolingual"].proto.type == "primary")
check("text typed before switching is kept",
      at.text_area(key="manual_text").value == "Manya lives in NOIDA.")
check("bilingual results are kept while the other tab is open",
      any("Status" in d.value.columns for d in at.dataframe))

at.button(key="process_input").click().run()
at.button(key="run_ner").click().run()
ner = next((d.value for d in at.dataframe if "Entity Type" in d.value.columns), None)
check("the existing monolingual workflow runs (Manya PER, NOIDA LOC)",
      ner is not None and {("Manya", "PER"), ("NOIDA", "LOC")} <= set(
          zip(ner["Entity"], ner["Entity Type"])),
      None if ner is None else ner[["Entity", "Entity Type"]].values.tolist())
check("monolingual tab now also shows 02 · Sentence Preview, 03 · Entity Extraction "
      "and Generate New Sentences",
      tab_headers(at, MONO) == ["01 · Input", "02 · Sentence Preview", "03 · Entity Extraction",
                                "Generate New Sentences"],
      tab_headers(at, MONO))

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
