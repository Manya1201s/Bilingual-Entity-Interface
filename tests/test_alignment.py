"""
Cross-lingual entity alignment tests (alignment/entity_pairs.py and app
section 04).

Run from the project folder with the Python 3.12 environment:

    .venv312\\Scripts\\python tests\\test_alignment.py

Part 1 (deterministic): entity tables are written by hand, as NER would
return them, so these checks test the alignment logic itself.
Part 2: the real Streamlit app with the real NER models.
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

from alignment.entity_pairs import (  # noqa: E402
    MATCHED, REVIEW, align_entities, load_calibration, pair_score,
)

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


def entities(sentence, spans):
    """Entity dicts in app format for (text, type) spans found in `sentence`."""
    out, start = [], 0
    for text, etype in spans:
        s = sentence.index(text, start)
        out.append({"Entity": text, "Entity Type": etype, "Confidence": 0.9,
                    "Start": s, "End": s + len(text), "Source": "test"})
        start = s + len(text)
    return out


def pairs(rows, status=None):
    return [((r["source"] or {}).get("Entity"), (r["target"] or {}).get("Entity"), r["status"])
            for r in rows if status is None or r["status"] == status]


calibration = load_calibration()
check("calibration file present (alignment/calibrate_pairs.py was run)", calibration is not None)
MATCH = calibration["match_threshold"]
REVIEW_T = calibration["review_threshold"]
check(f"thresholds sane: review {REVIEW_T} <= match {MATCH} <= 1",
      0 < REVIEW_T <= MATCH <= 1)
check("missing calibration file -> None (alignment reported unavailable, not guessed)",
      load_calibration(os.path.join(HERE, "no_such_file.json")) is None)


def align(en, other, lang="Hindi"):
    return align_entities(en, other, MATCH, REVIEW_T, "English", lang)


# ------------------------------------------------------------------
section("Part 1: alignment logic (hand-written entity tables)")

EN = "Manya lives in NOIDA"
HI = "मान्या नोएडा में रहती है।"
en = entities(EN, [("Manya", "PER"), ("NOIDA", "LOC")])
hi = entities(HI, [("मान्या", "PER"), ("नोएडा", "LOC")])
rows = align(en, hi)
check("Manya ↔ मान्या matched (PER)", ("Manya", "मान्या", MATCHED) in pairs(rows), pairs(rows))
check("NOIDA ↔ नोएडा matched (LOC)", ("NOIDA", "नोएडा", MATCHED) in pairs(rows), pairs(rows))
check("matched rows carry the original entity dicts (text, type, offsets, confidence, provenance)",
      all(r["source"] in en and r["target"] in hi for r in rows)
      and all(EN[r["source"]["Start"]:r["source"]["End"]] == r["source"]["Entity"]
              and HI[r["target"]["Start"]:r["target"]["End"]] == r["target"]["Entity"] for r in rows))
check("evidence names the signal and the type check",
      all("Name similarity" in r["evidence"] and "same type" in r["evidence"] for r in rows),
      [r["evidence"] for r in rows])

rows = align(entities("Infosys is big.", [("Infosys", "ORG")]),
             entities("इन्फोसिस बड़ा है।", [("इन्फोसिस", "LOC")]))
check("incompatible types (ORG vs LOC) are not Matched despite a perfect spelling match",
      pairs(rows) == [("Infosys", "इन्फोसिस", REVIEW)] and "types differ" in rows[0]["evidence"],
      [(p, r["evidence"]) for p, r in zip(pairs(rows), rows)])

rows = align(entities("Manya met Ravi.", [("Manya", "PER"), ("Ravi", "PER")]),
             entities("मान्या आई।", [("मान्या", "PER")]))
check("same type alone is not evidence: Ravi is not paired with मान्या",
      ("Ravi", None, "Unmatched English Entity") in pairs(rows)
      and ("Manya", "मान्या", MATCHED) in pairs(rows), pairs(rows))

rows = align(entities("India won.", [("India", "LOC")]),
             entities("भारत जीता।", [("भारत", "LOC")]))
check("translated name (India / भारत) stays unmatched on both sides, nothing dropped",
      sorted(pairs(rows), key=str) == sorted([("India", None, "Unmatched English Entity"),
                                               (None, "भारत", "Unmatched Hindi Entity")], key=str),
      pairs(rows))

score = pair_score("Priya Mehta", "प्रिया")[0]
rows = align(entities("Priya Mehta came.", [("Priya Mehta", "PER")]),
             entities("प्रिया आई।", [("प्रिया", "PER")]))
expected = MATCHED if score >= MATCH else REVIEW if score >= REVIEW_T else "Unmatched English Entity"
check(f"partial name 'Priya Mehta' ↔ 'प्रिया' (score {score:.3f}) follows the thresholds -> {expected}",
      rows[0]["status"] == expected and expected != MATCHED, pairs(rows))

EN3 = "Priya Mehta met Vikram Patel in Pune."
HI3 = "विक्रम पटेल पुणे में प्रिया मेहता से मिले।"
rows = align(entities(EN3, [("Priya Mehta", "PER"), ("Vikram Patel", "PER"), ("Pune", "LOC")]),
             entities(HI3, [("विक्रम पटेल", "PER"), ("पुणे", "LOC"), ("प्रिया मेहता", "PER")]))
check("three entities in different word order aligned correctly",
      sorted(pairs(rows)) == sorted([("Priya Mehta", "प्रिया मेहता", MATCHED),
                                     ("Vikram Patel", "विक्रम पटेल", MATCHED),
                                     ("Pune", "पुणे", MATCHED)]), pairs(rows))

rows = align(entities("Priya and Priya Mehta.", [("Priya", "PER"), ("Priya Mehta", "PER")]),
             entities("प्रिया मेहता आईं।", [("प्रिया मेहता", "PER")]))
targets = [r["target"]["Entity"] for r in rows if r["target"] is not None and r["source"] is not None]
check("no many-to-one: one Hindi entity is assigned to at most one English entity",
      len(targets) == len(set(targets)) and any(r["source"] and r["target"] is None for r in rows),
      pairs(rows))

rows = align(entities("Priya called.", [("Priya", "PER")]),
             entities("प्रिया ने प्रिया को बुलाया।", [("प्रिया", "PER"), ("प्रिया", "PER")]))
check("two equally good candidates -> assigned pair marked uncertain, the other listed unmatched",
      pairs(rows, REVIEW) == [("Priya", "प्रिया", REVIEW)]
      and "competing" in rows[0]["evidence"]
      and (None, "प्रिया", "Unmatched Hindi Entity") in pairs(rows), pairs(rows))

rows = align([], entities(HI, [("मान्या", "PER")]))
check("no English entities -> Hindi entity still listed as unmatched",
      pairs(rows) == [(None, "मान्या", "Unmatched Hindi Entity")], pairs(rows))

MR = "मान्या नोएडामध्ये राहते आणि पुण्यात काम करते."
rows = align(entities("Manya lives in NOIDA and works in Pune.",
                      [("Manya", "PER"), ("NOIDA", "LOC"), ("Pune", "LOC")]),
             entities(MR, [("मान्या", "PER"), ("नोएडामध्ये", "LOC"), ("पुण्या", "LOC")]), "Marathi")
check("English–Marathi: Manya, NOIDA (with case suffix मध्ये), Pune (पुण्या) matched",
      sorted(pairs(rows)) == sorted([("Manya", "मान्या", MATCHED),
                                     ("NOIDA", "नोएडामध्ये", MATCHED),
                                     ("Pune", "पुण्या", MATCHED)]), pairs(rows))
noida = next(r for r in rows if r["source"]["Entity"] == "NOIDA")
check("suffix handling is shown as evidence; offsets still cover the full Marathi entity",
      "मध्ये" in noida["evidence"] and MR[noida["target"]["Start"]:noida["target"]["End"]] == "नोएडामध्ये",
      noida["evidence"])

# ------------------------------------------------------------------
section("Part 2: the real app (section 04)")

from streamlit.testing.v1 import AppTest  # noqa: E402


def run_app(second, english, other, engine):
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.segmented_control(key="bilingual_pair").set_value(f"English → {second}")
    at.selectbox(key="bilingual_engine").set_value(engine).run()
    at.text_area(key="bilingual_english").set_value(english)
    at.text_area(key="bilingual_second").set_value(other)
    at.button(key="run_bilingual").click().run()
    table = next((d.value for d in at.dataframe if "Status" in d.value.columns), None)
    return at, table


BASELINE, TRAINED = "Baseline multilingual model", "Language-specific trained models"

at, table = run_app("Hindi", EN, HI, BASELINE)
print(table.to_string() if table is not None else "no alignment table")
check("alignment table has the required columns",
      table is not None and {"English Entity", "Hindi Entity", "Type", "Alignment Score",
                             "Evidence", "Status"} <= set(table.columns),
      None if table is None else list(table.columns))
got = [] if table is None else list(zip(table["English Entity"], table["Hindi Entity"], table["Status"]))
check("UI: Manya ↔ मान्या and NOIDA ↔ नोएडा matched (baseline NER)",
      ("Manya", "मान्या", MATCHED) in got and ("NOIDA", "नोएडा", MATCHED) in got, got)
check("UI: offsets shown and match the sentences",
      table is not None and list(table["English Offsets"]) == ["0–5", "15–20"]
      and list(table["Hindi Offsets"]) == ["0–6", "7–12"], None if table is None else table.to_dict("list"))
metrics = {m.label: m.value for m in at.metric}
check("UI: summary counts",
      metrics.get("English Entities") == "2" and metrics.get("Hindi Entities") == "2"
      and metrics.get("Matched Pairs") == "2" and metrics.get("Needs Review") == "0"
      and metrics.get("Unmatched (English / Hindi)") == "0 / 0", metrics)
check("UI: old generic 'not aligned' warning replaced by a truthful status",
      not any("confirmed entity alignment nahi" in w.value for w in at.warning)
      and any("Alignment complete" in s.value and "not been verified by a reviewer" in s.value
              for s in at.success))

at, table = run_app("Hindi", EN, HI, TRAINED)
got = [] if table is None else list(zip(table["English Entity"], table["Hindi Entity"], table["Status"]))
print("  OBSERVED trained-model alignment:", got)
entity_tables = [d.value for d in at.dataframe if "Source" in d.value.columns]
english_found = sorted(entity_tables[0]["Entity"]) if len(entity_tables) > 0 else []
hindi_found = sorted(entity_tables[1]["Entity"]) if len(entity_tables) > 1 else []
check("UI: every entity from both NER tables appears in the alignment table (nothing discarded)",
      table is not None and not at.exception
      and sorted(e for e in table["English Entity"] if e) == english_found
      and sorted(h for h in table["Hindi Entity"] if h) == hindi_found,
      {"english": english_found, "hindi": hindi_found, "alignment": got})
check("UI: entities without evidence are shown as unmatched, not forced into a pair",
      all(s != MATCHED or pair_score(e, h)[0] >= MATCH for e, h, s in got), got)

at, table = run_app("Marathi", EN, "मान्या नोएडामध्ये राहते.", BASELINE)
got = [] if table is None else list(zip(table["English Entity"], table["Marathi Entity"], table["Status"]))
print("  OBSERVED English–Marathi:", got)
check("UI English–Marathi: alignment table present, no exceptions", table is not None and not at.exception)
check("UI English–Marathi: NOIDA aligned to the Marathi NOIDA entity",
      any(e == "NOIDA" and h.startswith("नोएडा") and s == MATCHED for e, h, s in got), got)

# ------------------------------------------------------------------
section("Part 3: the four language pairs")

import app  # noqa: E402  (bare mode; only constants and pure helpers are used)

PAIRS = {
    "English → Hindi": ("English", "Hindi"),
    "English → Marathi": ("English", "Marathi"),
    "Hindi → English": ("Hindi", "English"),
    "Marathi → English": ("Marathi", "English"),
}
check("exactly the four language pairs, in order", app.LANGUAGE_PAIRS == PAIRS, app.LANGUAGE_PAIRS)

# Deterministic: input order never changes the alignment.
hi_side = ("Hindi", HI, hi)
en_side = ("English", EN, en)
check("english_first puts English first and keeps each side unchanged",
      app.english_first([hi_side, en_side]) == [en_side, hi_side]
      and app.english_first([en_side, hi_side]) == [en_side, hi_side]
      and app.english_first([hi_side, en_side])[1] is hi_side)
forward = align(*[side[2] for side in app.english_first([en_side, hi_side])])
reverse = align(*[side[2] for side in app.english_first([hi_side, en_side])])
check("Hindi-first input gives the same alignment as English-first input",
      pairs(forward) == pairs(reverse), (pairs(forward), pairs(reverse)))

SENTENCES = {
    "English": EN,
    "Hindi": HI,
    "Marathi": "मान्या नोएडामध्ये राहते.",
}
MODELS = {
    "English": "English BiLSTM-CRF (scratch)",
    "Hindi": "Hindi BiLSTM-CRF (scratch)",
    "Marathi": "Marathi BiLSTM-CRF (scratch)",
}


def run_pair(pair, engine, texts=None, click=True):
    first, second = PAIRS[pair]
    texts = texts or {first: SENTENCES[first], second: SENTENCES[second]}
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    at.segmented_control(key="bilingual_pair").set_value(pair)
    at.selectbox(key="bilingual_engine").set_value(engine).run()
    indic = second if first == "English" else first
    at.text_area(key="bilingual_english").set_value(texts.get("English", ""))
    at.text_area(key="bilingual_second").set_value(texts.get(indic, ""))
    if click:
        at.button(key="run_bilingual").click().run()
    return at


for pair, (first, second) in PAIRS.items():
    indic = second if first == "English" else first

    # Field order, labels and placeholders, before running anything.
    at = run_pair(pair, BASELINE, click=False)
    boxes = [t for t in at.text_area if t.key in ("bilingual_english", "bilingual_second")]
    check(f"{pair}: fields '{first} Sentence' then '{second} Sentence' with matching placeholders",
          [b.label for b in boxes] == [f"{first} Sentence", f"{second} Sentence"]
          and [b.placeholder for b in boxes] == [app.EXAMPLE_SENTENCES[first],
                                                 app.EXAMPLE_SENTENCES[second]],
          [(b.label, b.placeholder) for b in boxes])

    # Model routing: trained models show which model handled each side.
    at = run_pair(pair, TRAINED)
    tables = [d.value for d in at.dataframe if "Source" in d.value.columns]
    check(f"{pair}: no exceptions, two entity tables", not at.exception and len(tables) == 2,
          [e.value for e in at.exception])
    if len(tables) == 2:
        for language, table in zip((first, second), tables):
            sources = set(table["Source"])
            check(f"{pair}: {language} sentence handled only by {MODELS[language]}",
                  all(src.startswith(MODELS[language]) or src.startswith("Configured alias")
                      for src in sources) and any(src.startswith(MODELS[language]) for src in sources),
                  sources)
            sentence = SENTENCES[language]
            check(f"{pair}: {language} offsets refer to the {language} sentence",
                  all(sentence[s:e] == t for s, e, t in
                      zip(table["Start"], table["End"], table["Entity"])),
                  table[["Entity", "Start", "End"]].to_dict("records"))
    subheaders = [h.value for h in at.subheader]
    check(f"{pair}: entity tables shown in input order ({first}, then {second})",
          subheaders[:2] == [f"{first} Entities", f"{second} Entities"], subheaders)
    check(f"{pair}: results labelled with the pair",
          any(c.value == f"Results for {pair}" for c in at.caption))

    # Alignment with the baseline model, which finds both names in all three languages.
    at = run_pair(pair, BASELINE)
    table = next((d.value for d in at.dataframe if "Status" in d.value.columns), None)
    got = [] if table is None else list(zip(table["English Entity"], table[f"{indic} Entity"],
                                            table["Status"]))
    check(f"{pair}: alignment table has English and {indic} columns",
          table is not None and {"English Entity", f"{indic} Entity"} <= set(table.columns),
          None if table is None else list(table.columns))
    check(f"{pair}: Manya and NOIDA matched to their {indic} names",
          ("Manya", "मान्या", MATCHED) in got
          and any(e == "NOIDA" and h.startswith("नोएडा") and s == MATCHED for e, h, s in got), got)
    check(f"{pair}: selected pair kept after the run",
          at.segmented_control(key="bilingual_pair").value == pair)

# Validation: language-specific messages, nothing processed.
at = run_pair("Hindi → English", BASELINE, texts={"Hindi": HI})
check("Hindi → English with an empty English field -> 'Enter the English sentence.'",
      any(w.value == "Enter the English sentence." for w in at.warning)
      and not any("Status" in d.value.columns for d in at.dataframe),
      [w.value for w in at.warning])
at = run_pair("Marathi → English", BASELINE, texts={"English": "   "})
check("whitespace-only inputs -> asks for both sentences in pair order",
      any(w.value == "Enter both the Marathi and the English sentence." for w in at.warning),
      [w.value for w in at.warning])

# Changing the pair moves the fields, never the text.
at = run_pair("English → Hindi", BASELINE, click=False)
at.segmented_control(key="bilingual_pair").set_value("Hindi → English").run()
boxes = {t.key: t for t in at.text_area}
check("switching English → Hindi to Hindi → English keeps each text in its own language field",
      boxes["bilingual_english"].value == EN and boxes["bilingual_english"].label == "English Sentence"
      and boxes["bilingual_second"].value == HI and boxes["bilingual_second"].label == "Hindi Sentence")

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
