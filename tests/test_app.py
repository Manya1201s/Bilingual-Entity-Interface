"""
Final test suite for Indic Entity Lab (app.py).

Run from the project folder with the project's virtual environment:

    .venv\\Scripts\\python tests\\test_app.py

Part 1 tests the NER / alias / merging / file-reading functions directly.
Part 2 drives the real Streamlit app (streamlit.testing AppTest):
typed input, file uploads of every sample in tests/samples, NER and the
bilingual comparison.

The NER model must already be downloaded (it is cached after the first
run of the app). Exit code is 0 only if every check passes.
"""

import io
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

THRESHOLD = 0.30
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


def ner(text, threshold=THRESHOLD):
    return app.extract_ner(text, threshold, app.MODEL_NAME)


def found(entities, text, entity_type):
    return any(
        e["Entity"] == text and e["Entity Type"] == entity_type
        for e in entities
    )


def well_formed(entities, sentence):
    """No overlapping spans and every Entity equals sentence[Start:End]."""
    spans = sorted((e["Start"], e["End"]) for e in entities)
    no_overlap = all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))
    offsets_ok = all(
        sentence[e["Start"]:e["End"]] == e["Entity"] for e in entities
    )
    labels_ok = all(e["Entity Type"] in app.VALID_LABELS for e in entities)
    conf_ok = all(0.0 <= e["Confidence"] <= 1.0 for e in entities)
    return no_overlap and offsets_ok and labels_ok and conf_ok


def section(title):
    print(f"\n=== {title}")


def short(entities):
    return [(e["Entity"], e["Entity Type"], e["Confidence"]) for e in entities]


# ------------------------------------------------------------------
# PART 1 - functions
# ------------------------------------------------------------------

section("PER / LOC / ORG in English, Hindi, Marathi and mixed text")

CASES = [
    # English
    ("Aarav Sharma works at Infosys in Pune.",
     [("Aarav Sharma", "PER"), ("Infosys", "ORG"), ("Pune", "LOC")]),
    ("Satya Nadella of Microsoft visited Geneva.",
     [("Satya Nadella", "PER"), ("Microsoft", "ORG"), ("Geneva", "LOC")]),
    # Hindi
    ("नरेंद्र मोदी ने दिल्ली में भाषण दिया।",
     [("नरेंद्र मोदी", "PER"), ("दिल्ली", "LOC")]),
    ("भारतीय रिज़र्व बैंक का मुख्यालय मुंबई में है।",
     [("भारतीय रिज़र्व बैंक", "ORG"), ("मुंबई", "LOC")]),
    ("आरव शर्मा पुणे में इन्फोसिस में काम करते हैं।",
     [("आरव शर्मा", "PER"), ("पुणे", "LOC"), ("इन्फोसिस", "ORG")]),
    # Marathi
    ("नरेंद्र मोदी यांनी दिल्लीत भाषण दिले.",
     [("नरेंद्र मोदी", "PER"), ("दिल्ली", "LOC")]),
    ("गूगलने पुण्यात नवीन कार्यालय सुरू केले.",
     [("गूगल", "ORG"), ("पुण्या", "LOC")]),
    ("सचिन तेंडुलकर मुंबईचा आहे.",
     [("सचिन तेंडुलकर", "PER"), ("मुंबई", "LOC")]),
    ("आरव शर्मा पुण्यात इन्फोसिसमध्ये काम करतात।",
     [("आरव शर्मा", "PER"), ("पुण्या", "LOC"), ("इन्फोसिस", "ORG")]),
    # Mixed
    ("Geneva और जिनेव्हा के बीच बैठक हुई.",
     [("Geneva", "LOC"), ("जिनेव्हा", "LOC")]),
    ("Sundar Pichai आणि प्रिया मायक्रोसॉफ्टमध्ये जिनिव्हा येथे भेटले.",
     [("Sundar Pichai", "PER"), ("प्रिया", "PER"),
      ("मायक्रोसॉफ्ट", "ORG"), ("जिनिव्हा", "LOC")]),
]

for sentence, expected in CASES:
    entities = ner(sentence)
    print(f"  {sentence}\n        {short(entities)}")
    for text, entity_type in expected:
        check(f"{text} -> {entity_type}", found(entities, text, entity_type),
              short(entities))
    check("well-formed (no overlap, offsets, labels, confidence)",
          well_formed(entities, sentence), entities)

section("Geneva spellings")

GENEVA = [
    ("Geneva is an international city.", "Geneva"),
    ("जेनेवा येथे परिषद झाली.", "जेनेवा"),
    ("जिनेवा येथे परिषद झाली.", "जिनेवा"),
    ("जिनेव्हा येथे परिषद झाली.", "जिनेव्हा"),
    ("जिनेव्हा येथे एक आंतरराष्ट्रीय परिषद आयोजित करण्यात आली.", "जिनेव्हा"),
    ("परिषद जिनेव्हामध्ये झाली.", "जिनेव्हा"),
    ("जिनेव्हाच्या जवळ एक तलाव आहे.", "जिनेव्हा"),
    ("जिनिव्हा शहरात संयुक्त राष्ट्रांचे कार्यालय आहे.", "जिनिव्हा"),
]

for sentence, geneva in GENEVA:
    entities = ner(sentence)
    locs = [e for e in entities if e["Entity Type"] == "LOC"]
    check(f"{sentence!r} -> {geneva} LOC (exactly once)",
          found(entities, geneva, "LOC") and len(locs) == 1
          and well_formed(entities, sentence), short(entities))

section("Marathi suffixes / postpositions are not absorbed")

SUFFIXES = ["मध्ये", " येथे", "त", "ला", "च्या", "ची", "चे", "वर",
            "पासून", "पर्यंत"]

for base in ["मुंबई", "दिल्ली", "नाशिक", "पुणे", "जिनेव्हा"]:
    results = []
    for suffix in SUFFIXES:
        sentence = f"आम्ही {base}{suffix} गेलो."
        entities = ner(sentence)
        results.append(
            found(entities, base, "LOC") and well_formed(entities, sentence)
        )
    check(f"{base} + {len(SUFFIXES)} suffixes -> '{base}' LOC each time",
          all(results), [s for s, ok in zip(SUFFIXES, results) if not ok])

entities = ner("आम्ही पुण्यात गेलो.")
check("'पुण्यात' -> 'पुण्या' (complete syllable, suffix 'त' excluded)",
      found(entities, "पुण्या", "LOC"), short(entities))

section("Alias system")

entities = ner("सुप्रिया आणि प्रियांका पुण्यात राहतात.")
check("alias 'प्रिया' not matched inside सुप्रिया / प्रियांका",
      not any("alias" in e["Source"] and e["Entity"] == "प्रिया"
              for e in entities), short(entities))

entities = ner("The Genevan delegation arrived.")
check("alias 'Geneva' not matched inside 'Genevan'",
      not any("alias" in e["Source"] for e in entities), entities)

entities = ner("जिनेव्हा येथे परिषद झाली.")
check("alias-only entity: confidence 1.0, source 'Configured alias — review'",
      len(entities) == 1 and entities[0]["Confidence"] == 1.0
      and entities[0]["Source"] == "Configured alias — review", entities)

entities = ner("डॉ. प्रिया मेहता ने जिनेवा में माइक्रोसॉफ्ट के सत्या नडेला से मुलाकात की।")
geneva = [e for e in entities if e["Entity"] == "जिनेवा"]
check("model + alias: one entity, real model confidence kept (< 1.0)",
      len(geneva) == 1 and geneva[0]["Confidence"] < 1.0
      and geneva[0]["Source"] == "Multilingual NER model + configured alias",
      geneva)
check("longer model span 'प्रिया मेहता' wins over alias 'प्रिया'",
      found(entities, "प्रिया मेहता", "PER")
      and not found(entities, "प्रिया", "PER"), short(entities))

entities = ner("प्रिया टाटा कंसल्टेंसी सर्विसेज मध्ये मुंबईत काम करते.")
check("longest alias wins ('...सर्विसेज' not '...सर्विस'), no duplicates",
      found(entities, "टाटा कंसल्टेंसी सर्विसेज", "ORG")
      and well_formed(entities, "प्रिया टाटा कंसल्टेंसी सर्विसेज मध्ये मुंबईत काम करते."),
      short(entities))

entities = ner("Aarav Sharma works at Infosys in Pune.")
check("model-only entities keep source 'Multilingual NER model'",
      all(e["Source"] == "Multilingual NER model" for e in entities), entities)

section("Fragment merging, overlaps and confidence")

sentence = "Rohan Verma and Priya visited Tata Consultancy Services in Mumbai."
for threshold in (0.6, 0.95):
    entities = ner(sentence, threshold)
    check(f"threshold {threshold}: all >= threshold, no 'Pri'/'ya' fragments",
          all(e["Confidence"] >= threshold for e in entities)
          and not any(e["Entity"] in ("Pri", "ya") for e in entities),
          short(entities))
entities = ner(sentence, 0.6)
priya = [e for e in entities if e["Entity"] == "Priya"]
check("'Pri'(0.994) + 'ya'(0.567) -> 'Priya' 0.8229 (weighted, not min 0.567)",
      len(priya) == 1 and priya[0]["Confidence"] == 0.8229, priya)

entities = ner("रतन टाटा टाटा समूह के अध्यक्ष थे।")
check("mixed labels inside one word ('टा'ORG+'टा'PER) unified to one entity",
      found(entities, "टाटा", "PER") and not any(e["Entity"] == "टा" for e in entities)
      and found(entities, "टाटा समूह", "ORG"), short(entities))

items = [
    {"Entity": "टा", "Entity Type": "ORG", "Confidence": 0.92, "Start": 7, "End": 9, "Source": "m"},
    {"Entity": "X", "Entity Type": "PER", "Confidence": 0.99, "Start": 5, "End": 8, "Source": "m"},
    {"Entity": "टाटा कंसल्टेंसी सर्विसेज", "Entity Type": "ORG", "Confidence": 1.0, "Start": 7, "End": 31, "Source": "a"},
]
kept = app.remove_overlapping_entities(items)
check("overlap: longer span wins over an earlier-starting shorter span",
      len(kept) == 1 and (kept[0]["Start"], kept[0]["End"]) == (7, 31), kept)

section("Labels: DATE and unknown labels are not displayed")

fake = [{"entity_group": "DATE", "score": 0.99, "start": 0, "end": 4},
        {"entity_group": "MISC", "score": 0.99, "start": 0, "end": 4},
        {"entity_group": "B-PERSON", "score": 0.99, "start": 0, "end": 4},
        {"entity_group": "GPE", "score": 0.99, "start": 0, "end": 4}]
converted = [app.convert_prediction("Pune city", p, "m", 0.0) for p in fake]
check("DATE / MISC dropped; PERSON->PER, GPE->LOC",
      converted[0] is None and converted[1] is None
      and converted[2]["Entity Type"] == "PER"
      and converted[3]["Entity Type"] == "LOC", converted)
entities = ner("The meeting was held on 12 January 2024 in Geneva.")
check("real sentence with a date: only PER/LOC/ORG returned",
      all(e["Entity Type"] in app.VALID_LABELS for e in entities)
      and found(entities, "Geneva", "LOC"), short(entities))

section("Empty / invalid input and malformed spans")

check("empty text -> []", ner("") == [])
check("whitespace -> []", ner("  \n\t ") == [])
check("punctuation only -> []", ner("... !!! ।।") == [])
check("no-entity sentence -> []", ner("आज हवामान छान आहे.") == [], ner("आज हवामान छान आहे."))
bad = [{"entity_group": "LOC", "score": 0.9, "start": 5, "end": 2},
       {"entity_group": "LOC", "score": 0.9, "start": None, "end": 3},
       {"entity_group": "LOC", "score": 0.9, "start": 0, "end": 999},
       {"entity_group": "LOC", "score": 0.9, "start": -1, "end": 3},
       {"entity_group": "LOC", "score": 0.9, "start": 2, "end": 5}]
converted = [app.convert_prediction("ab   ", p, "m", 0.0) for p in bad]
check("malformed spans (reversed, None, out of range, negative, blank) rejected",
      converted == [None] * 5, converted)
c = app.convert_prediction(" Geneva ", {"entity_group": "LOC", "score": 0.9, "start": 0, "end": 8}, "m", 0.0)
check("whitespace trimmed from span", c and (c["Start"], c["End"], c["Entity"]) == (1, 7, "Geneva"), c)
check("snap does not absorb space after word-final virama ('जगत् ')",
      app.snap_to_syllables("जगत् में", [{"Entity": "जगत्", "Entity Type": "LOC", "Confidence": 1, "Start": 0, "End": 4, "Source": "m"}])[0]["End"] == 4)

section("Sentence splitting (titles and initials)")

SPLIT_CASES = [
    ("Dr. A. P. J. Abdul Kalam was born in Rameswaram. He was a scientist.",
     ["Dr. A. P. J. Abdul Kalam was born in Rameswaram.", "He was a scientist."]),
    ("Mr. Narendra Modi spoke in Delhi. Prof. A. P. J. Abdul Kalam listened.",
     ["Mr. Narendra Modi spoke in Delhi.", "Prof. A. P. J. Abdul Kalam listened."]),
    ("Dr. Priya Mehta met Satya Nadella. डॉ. प्रिया मेहता जिनेव्हा येथे गेल्या.",
     ["Dr. Priya Mehta met Satya Nadella.", "डॉ. प्रिया मेहता जिनेव्हा येथे गेल्या."]),
    ("पहला वाक्य। दूसरा वाक्य॥ Third one!\nFourth line",
     ["पहला वाक्य।", "दूसरा वाक्य॥", "Third one!", "Fourth line"]),
    ("... !!!\n।।\n", []),
]
for text, expected in SPLIT_CASES:
    got = app.split_sentences(text)
    check(f"split {text[:40]!r}", got == expected, got)

for sentence, person in [
    ("Dr. A. P. J. Abdul Kalam was born in Rameswaram.", "A. P. J. Abdul Kalam"),
    ("Mr. Narendra Modi spoke in Delhi.", "Narendra Modi"),
    ("Dr. Priya Mehta met Satya Nadella.", "Priya Mehta"),
    ("Prof. A. P. J. Abdul Kalam listened.", "A. P. J. Abdul Kalam"),
]:
    entities = ner(sentence)
    check(f"NER after split: {person} -> PER (title not an entity)",
          found(entities, person, "PER")
          and not any(e["Entity"].casefold() in app.PERSON_TITLES for e in entities),
          short(entities))

section("File reading (read_file)")


class Upload:
    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._raw = f.read()

    def getvalue(self):
        return self._raw


def read(name, column=None):
    return app.read_file(Upload(os.path.join(SAMPLES, name)), column)


EN = "Dr. A. P. J. Abdul Kalam visited Geneva with Satya Nadella of Microsoft."
HI = "नरेंद्र मोदी ने दिल्ली में भारतीय रिज़र्व बैंक के अधिकारियों से मुलाकात की।"
MR = "विराट कोहली मुंबईत राहतो आणि जिनेव्हामध्ये इन्फोसिसचे कार्यालय आहे."
MIX = "Geneva और जिनेव्हा के बीच बैठक हुई."
ALL = "\n".join([EN, HI, MR, MIX])

check("TXT UTF-8 mixed", read("mixed_utf8.txt") == ALL)
check("TXT UTF-8 with BOM", read("mixed_utf8_bom.txt") == ALL)
check("TXT UTF-16", read("mixed_utf16.txt") == ALL)
check("TXT English / Hindi / Marathi",
      (read("english.txt"), read("hindi.txt"), read("marathi.txt")) == (EN, HI, MR))
check("TXT empty -> ''", read("empty.txt") == "")
bad_text = read("bad_encoding.txt")
check("TXT invalid bytes do not crash; control characters removed",
      "Geneva is a city." in bad_text
      and not any(ord(c) < 32 or 127 <= ord(c) < 160 for c in bad_text if c not in "\n\r\t"),
      repr(bad_text))
check("TXT cp1252 English", "Rohan Verma lives in Pune." in read("cp1252_english.txt"))
check("CSV normal (auto 'text' column)", read("normal.csv").splitlines() == [EN, HI, MR])
check("CSV UTF-16 ('sentence' column)", read("normal_utf16.csv") == MR)
check("CSV chosen column", read("other_column.csv", "body") == MR)
try:
    read("other_column.csv")
    check("CSV without text column -> friendly error", False)
except ValueError as exc:
    check("CSV without text column -> friendly ValueError", "column" in str(exc))
check("CSV empty file -> ''", read("empty.csv") == "")
check("CSV header only -> ''", read("header_only.csv") == "")
check("DOCX mixed (paragraphs + table)",
      read("mixed.docx") == ALL + "\nपुणे | Tata Consultancy Services")
check("DOCX empty -> ''", read("empty.docx") == "")
pdf_text = read("mixed.pdf")
check("PDF English line extracted exactly", pdf_text.splitlines()[0] == EN, pdf_text[:80])
check("PDF: no NUL characters in extracted text", "\x00" not in pdf_text)
with open(os.path.join(SAMPLES, "mixed.pdf"), "rb") as f:
    check("PDF: damaged Devanagari is detected (warning shown in UI)",
          app.pdf_devanagari_damaged(f.read()))
for name in ("corrupt.docx", "corrupt.pdf", "empty.pdf"):
    try:
        read(name)
        check(f"{name} -> friendly error", False)
    except ValueError as exc:
        check(f"{name} -> friendly ValueError", True)
    except Exception as exc:  # noqa: BLE001
        check(f"{name} -> friendly ValueError", False, repr(exc))

check("markdown escaping keeps text literal",
      app.escape_markdown("$5 *x* #1") == r"\$5 \*x\* \#1"
      and app.escape_markdown(MR) == MR.replace(".", r"\."))

# ------------------------------------------------------------------
# PART 2 - the real Streamlit app
# ------------------------------------------------------------------

from streamlit.testing.v1 import AppTest  # noqa: E402

section("Streamlit app: startup")

at = AppTest.from_file(APP_PATH, default_timeout=900)
at.run()
check("app starts without exceptions", not at.exception, [e.value for e in at.exception])
check("no sidebar error (transformers importable)", not at.sidebar.error,
      [e.value for e in at.sidebar.error])
check("headers present",
      [h.value for h in at.header][:3] == ["NER Extraction", "01 · Input", "04 · Entity Alignment"],
      [h.value for h in at.header])

section("Streamlit app: typed input -> sentences -> NER")

at.text_area(key="manual_text").set_value(
    "Dr. A. P. J. Abdul Kalam visited Geneva.\n"
    "जिनेव्हा येथे एक आंतरराष्ट्रीय परिषद आयोजित करण्यात आली.\n"
    "जेनेवा येथे परिषद झाली. जिनेवा येथे परिषद झाली.\n"
    "Geneva और जिनेव्हा के बीच बैठक हुई.\n"
    "Price is $5 and $10 *today*."
)
at.button(key="process_input").click().run()
check("process input: 6 sentences", any("Extracted 6 sentences" in s.value for s in at.success),
      [s.value for s in at.success])
check("sentence shown literally ($ and * escaped)",
      any(m.value == r"Price is \$5 and \$10 \*today\*\." for m in at.markdown))

at.button(key="run_ner").click().run()
check("NER runs without exceptions/errors", not at.exception and not at.error,
      [e.value for e in at.exception] + [e.value for e in at.error])
df = at.dataframe[0].value
print(df[["Sentence No.", "Entity", "Entity Type", "Confidence", "Source"]].to_string())
locs = df[df["Entity Type"] == "LOC"]["Entity"].tolist()
check("UI table: all Geneva spellings found as LOC",
      locs == ["Geneva", "जिनेव्हा", "जेनेवा", "जिनेवा", "Geneva", "जिनेव्हा"], locs)
# The monolingual workflow uses the English BiLSTM-CRF here (input language
# English); it tags "Abdul Kalam" without the initials.
pers = df[df["Entity Type"] == "PER"]["Entity"].tolist()
check("UI table: Kalam PER (monolingual English model)",
      any(p.endswith("Abdul Kalam") for p in pers), pers)
check("multilingual model keeps the full name 'A. P. J. Abdul Kalam' (PER)",
      any(e["Entity"] == "A. P. J. Abdul Kalam" and e["Entity Type"] == "PER"
          for e in ner("Dr. A. P. J. Abdul Kalam visited Geneva.")))
check("UI table: no duplicate rows", not df.duplicated(["Sentence No.", "Start", "End"]).any())
muril_notices = [i.value for i in at.info if "MuRIL" in i.value or "muril" in i.value]
check("no MuRIL notice (unusable generic-label model is no longer configured)",
      not muril_notices, muril_notices)

section("Streamlit app: language selection is honest")

at.selectbox(key="ner_language").set_value("English").run()
at.button(key="process_input").click().run()
check("language mismatch warning (English selected, Devanagari text)",
      any("Selected language is English" in w.value for w in at.warning),
      [w.value for w in at.warning])
check("detected script caption shown", any("Detected script" in c.value for c in at.caption))

section("Streamlit app: bilingual comparison")

at.segmented_control(key="bilingual_pair").set_value("English → Marathi").run()
at.text_area(key="bilingual_english").set_value("Geneva is an international city.")
at.text_area(key="bilingual_second").set_value("जिनेव्हा हे एक आंतरराष्ट्रीय शहर आहे.")
at.button(key="run_bilingual").click().run()
check("bilingual runs without exceptions", not at.exception, [e.value for e in at.exception])
tables = [d.value for d in at.dataframe]
check("bilingual: Geneva LOC and जिनेव्हा LOC",
      any(list(t["Entity"]) == ["Geneva"] for t in tables if "Entity" in t)
      and any(list(t["Entity"]) == ["जिनेव्हा"] for t in tables if "Entity" in t),
      [list(t["Entity"]) for t in tables if "Entity" in t])

section("Streamlit app: empty input")

at = AppTest.from_file(APP_PATH, default_timeout=900)
at.run()
at.button(key="process_input").click().run()
check("empty input -> warning, no crash",
      not at.exception and any("No readable text was found" in w.value for w in at.warning))

section("Streamlit app: file uploads")

MIME = {"txt": "text/plain", "csv": "text/csv", "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}

UPLOADS = [
    # (file, expected outcome: number of sentences | "warn" | "error")
    ("mixed_utf8.txt", 4), ("mixed_utf8_bom.txt", 4), ("mixed_utf16.txt", 4),
    ("english.txt", 1), ("hindi.txt", 1), ("marathi.txt", 1),
    ("no_entities.txt", 2), ("bad_encoding.txt", 2), ("cp1252_english.txt", 1),
    ("normal.csv", 3), ("normal_utf16.csv", 1),
    ("mixed.docx", 5), ("mixed.pdf", 4),
    ("empty.txt", "warn"), ("punctuation_only.txt", "warn"),
    ("empty.csv", "warn"), ("header_only.csv", "warn"), ("empty.docx", "warn"),
    ("other_column.csv", "error"), ("corrupt.docx", "error"),
    ("corrupt.pdf", "error"), ("empty.pdf", "error"),
]

for name, expected in UPLOADS:
    at = AppTest.from_file(APP_PATH, default_timeout=900)
    at.run()
    with open(os.path.join(SAMPLES, name), "rb") as f:
        at.file_uploader[0].upload(name, f.read(), MIME[name.rsplit(".", 1)[1]])
    at.run()
    at.button(key="process_input").click().run()

    crashed = bool(at.exception)
    successes = [s.value for s in at.success]
    errors = [e.value for e in at.error]
    warns = [w.value for w in at.warning]

    if isinstance(expected, int):
        ok = (not crashed and any(f"Extracted {expected} sentence" in s for s in successes))
        if ok and name in ("mixed_utf16.txt", "mixed.docx", "normal.csv"):
            at.button(key="run_ner").click().run()
            ok = not at.exception and not at.error and len(at.dataframe[0].value) > 0
    elif expected == "warn":
        ok = not crashed and any("No readable text was found" in w for w in warns)
    else:
        ok = not crashed and any("The input could not be processed" in e for e in errors)

    detail = {"exceptions": [e.value for e in at.exception], "success": successes,
              "errors": errors, "warnings": warns}
    check(f"upload {name} -> {expected}", ok, detail)

    if name == "mixed.pdf":
        check("PDF upload shows the damaged-Devanagari warning",
              any("Hindi/Marathi text in this PDF" in w for w in warns), warns)

# ------------------------------------------------------------------

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
