"""
Tests for anonymisation/ and the end-to-end pipeline (pipeline/).

    .venv\\Scripts\\python tests\\test_anonymisation.py

Run after anonymisation/evaluate_anonymisation.py (--part mono and --part cross)
and pipeline/run_pipeline.py. Exit code 0 only if every check passes.
"""

import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

from anonymisation.anonymise import PATTERNS, PLACEHOLDER, Anonymiser, Registry  # noqa: E402

failures, passes = [], 0


def check(name, condition, detail=""):
    global passes
    if condition:
        passes += 1
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}\n        -> {detail}")


def load(path):
    with open(os.path.join(PROJECT, path), encoding="utf-8") as f:
        return json.load(f)


print("\n=== Anonymiser behaviour")
reg = Registry()
check("same entity text -> same placeholder; different -> different",
      reg.placeholder("PER", "प्रिया मेहता") == reg.placeholder("PER", "प्रिया  मेहता")
      and reg.placeholder("PER", "राहुल") != reg.placeholder("PER", "प्रिया मेहता"))
reg.link("PER", "Priya Mehta", reg.placeholder("PER", "प्रिया मेहता"))
check("cross-lingual link reuses the placeholder", reg.placeholder("PER", "Priya Mehta") == "[PER_1]")
for text, kind in (("9876543210", "PHONE"), ("+91 98765 43210", "PHONE"), ("1234 5678 9012", "AADHAAR"),
                   ("ABCDE1234F", "PAN"), ("a.b@example.org", "EMAIL"), ("https://example.org/x", "URL")):
    check(f"regex detects {kind}: {text}", any(n == kind and p.search(text) for n, p in PATTERNS))
for text in ("2019", "Rs 25000", "450 pages"):
    check(f"regex leaves '{text}' alone", not any(p.search(text) for _, p in PATTERNS))
mr = Anonymiser("mr")
out, reps = mr.anonymise("देवेंद्र फडणवीस मुंबईत होते आणि देवेंद्र फडणवीस पुण्याला गेले.", Registry())
check("Marathi: suffix kept outside the placeholder, repeated name consistent",
      "]त " in out and out.count("[PER_1]") == 2, out)
check("placeholders only replace spans (text outside unchanged)", out.endswith("गेले.") and "होते आणि" in out, out)

print("\n=== Monolingual evaluation results")
mono = load("anonymisation/results/anonymisation_mono.json")
for lang in ("en", "hi", "mr"):
    r = mono[lang]
    m = r["metrics"]
    check(f"[{lang}] all metrics present and in [0, 1]",
          all(0 <= m[k] <= 1 for k in ("residual_identifier_rate", "missed_anonymisation_rate",
                                       "partial_anonymisation_rate", "false_positive_rate",
                                       "incorrect_anonymisation_rate", "non_entity_preservation")), m)
    check(f"[{lang}] identity consistency: no entity with two placeholders, no shared placeholder",
          m["identity_consistency"]["entity_texts_with_multiple_placeholders"] == 0
          and m["identity_consistency"]["placeholders_shared_by_different_texts"] == 0, m["identity_consistency"])
    s = r["synthetic_identifiers"]
    check(f"[{lang}] synthetic identifiers all detected, negative controls untouched",
          all(v == 1.0 for v in s["recall_by_type"].values()) and s["negative_controls_wrongly_anonymised"] == 0, s)

print("\n=== Cross-lingual anonymisation results")
cross = load("anonymisation/results/anonymisation_cross.json")
for lang in ("hi", "mr"):
    r = cross[lang]
    check(f"[en-{lang}] evaluated on every gold pair, rates in [0, 1]",
          r["gold_pairs"] > 0 and 0 <= r["cross_lingual_consistency"] <= 1 and 0 <= r["target_gold_left_visible"] <= 1, r.keys())

print("\n=== End-to-end pipeline")
summary = load("pipeline/results/pipeline_summary.json")
for lang in ("hi", "mr"):
    s = summary[lang]
    check(f"[{lang}] no crashes, no empty outputs, no corrupted text",
          s["crashes"] == 0 and s["examples_with_empty_output"] == 0 and s["examples_with_corrupted_text"] == 0, s)
    check(f"[{lang}] no leakage of detected entities into anonymised text or MT output",
          s["examples_with_leakage"] == 0, s["examples_with_leakage"])
    check(f"[{lang}] identity consistent across the whole run",
          s["identity"]["entity_texts_with_multiple_placeholders"] == 0, s["identity"])
    check(f"[{lang}] protected Hindi/Marathi NER files unchanged during the run", s["protected_ner_files_unchanged"])
    rows = [json.loads(l) for l in open(os.path.join(PROJECT, "pipeline", "results", f"{lang}_outputs.jsonl"), encoding="utf-8")]
    check(f"[{lang}] every example has anonymised text and an MT output",
          len(rows) == s["examples"] and all(r.get("anonymised") and r.get("mt") for r in rows))
    demo = [r for r in rows if r["source"] == "synthetic demo"]
    check(f"[{lang}] synthetic demo: phone/email removed from both sides",
          all("9876543210" not in r["anonymised"]["english"] + r["anonymised"]["target"]
              and "example.org" not in r["anonymised"]["english"] + r["anonymised"]["target"] for r in demo))
    def full(text):
        return [m.group() for m in PLACEHOLDER.finditer(text)]
    check(f"[{lang}] synthetic demo: the repeated name uses one placeholder on the English side",
          any(len(set(full(r["anonymised"]["english"]))) < len(full(r["anonymised"]["english"])) for r in demo),
          [r["anonymised"]["english"] for r in demo])

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
