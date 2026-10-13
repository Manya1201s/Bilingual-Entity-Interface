"""
Tests for the English -> Hindi / Marathi MT component (mt/).

    .venv\\Scripts\\python tests\\test_mt.py

Run after mt/prepare_parallel.py, mt/train_mt.py and mt/evaluate_mt.py.
Exit code 0 only if every check passes.
"""

import json
import os
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.stdout.reconfigure(encoding="utf-8")

import mt.evaluate_mt as E  # noqa: E402
from mt.model import Translator  # noqa: E402

failures, passes = [], 0


def check(name, condition, detail=""):
    global passes
    if condition:
        passes += 1
        print(f"  PASS  {name}")
    else:
        failures.append(name)
        print(f"  FAIL  {name}\n        -> {detail}")


def read(lang, split):
    with open(os.path.join(PROJECT, "mt", "data", lang, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


print("\n=== Parallel data")
result = subprocess.run([sys.executable, os.path.join(PROJECT, "mt", "validate_parallel.py")],
                        capture_output=True, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
check("validate_parallel.py: no problems (empty, duplicates, leakage, script, length, NER overlap)",
      result.returncode == 0 and "OK - no problems found" in result.stdout, result.stdout[-600:])
for lang in ("hi", "mr"):
    sizes = {s: len(read(lang, s)) for s in ("train", "validation", "test")}
    check(f"[{lang}] split sizes 50000 / 1000 / 1000", sizes == {"train": 50000, "validation": 1000, "test": 1000}, sizes)
    info = json.load(open(os.path.join(PROJECT, "mt", "data", lang, "dataset_info.json"), encoding="utf-8"))
    check(f"[{lang}] provenance recorded (source, licence, seed, row groups)",
          info["source"]["dataset"] == "ai4bharat/samanantar" and "CC-BY-NC" in info["source"]["license"]
          and info["seed"] == 2024 and len(info["row_groups_used"]) == info["stats"]["row_groups_read"])
    train_ids = {r["id"] for r in read(lang, "train")}
    held = {r["id"] for s in ("validation", "test") for r in read(lang, s)}
    check(f"[{lang}] no Samanantar row id shared between train and validation/test", not (train_ids & held))

print("\n=== Trained models")
translators = {}
for lang in ("hi", "mr"):
    model_dir = os.path.join(PROJECT, "models", f"mt_en_{lang}")
    files = set(os.listdir(model_dir)) if os.path.isdir(model_dir) else set()
    check(f"[{lang}] model folder complete (config, weights, both SentencePiece models, resume state)",
          {"config.json", "model.pt", "spm.en.model", f"spm.{lang}.model", "last_checkpoint.pt"} <= files, files)
    info = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    check(f"[{lang}] trained from scratch; best checkpoint chosen on validation loss",
          "from scratch" in info["model_type"] and info["best_epoch"] >= 1
          and info["best_val_loss"] == min(e["val_loss"] for e in info["log"]), info.get("best_epoch"))
    check(f"[{lang}] validation loss decreased from the first epoch",
          info["log"][0]["val_loss"] > info["best_val_loss"], [e["val_loss"] for e in info["log"]])
    translators[lang] = Translator(model_dir)
    out = translators[lang].translate(["The meeting was held in Delhi."], beam_size=4)
    check(f"[{lang}] model loads and translates (beam 4) into Devanagari",
          out and out[0].strip() and any(0x900 <= ord(c) < 0x980 for c in out[0]), out)
    greedy = translators[lang].translate(["The meeting was held in Delhi.", "India won the match."], beam_size=1)
    check(f"[{lang}] batched greedy decoding returns one output per input", len(greedy) == 2, greedy)

print("\n=== Entity-preservation categorisation (unit test)")


class FakeNER:
    def __init__(self, table):
        self.t = table

    def predict_batch(self, sentences):
        return [self.t.get(s, []) for s in sentences]


refs = ["प्रिया मेहता दिल्ली गईं", "आरव पुणे में", "मुंबई सुंदर है"]
hyps = ["प्रिया मेहता दिल्ली गईं", "आरव पुना में", "यह जगह सुंदर है टाटा"]
table = {refs[0]: [("PER", "प्रिया मेहता"), ("LOC", "दिल्ली")], hyps[0]: [("PER", "प्रिया मेहता"), ("LOC", "दिल्ली")],
         refs[1]: [("PER", "आरव"), ("LOC", "पुणे")], hyps[1]: [("PER", "आरव"), ("LOC", "पुना")],
         refs[2]: [("LOC", "मुंबई")], hyps[2]: [("ORG", "टाटा")]}
counts, rates, _, _ = E.entity_preservation(refs, hyps, FakeNER(table))
check("preserved / translated incorrectly / dropped / added counted correctly",
      counts == {"reference_entities": 5, "hypothesis_entities": 5, "preserved": 3,
                 "translated_incorrectly": 1, "dropped": 1, "hallucinated_added": 1}, counts)
check("whole-word matching (पुणे not found inside पुणेरी)", not E.contains_word("पुणेरी लोक", "पुणे"))

print("\n=== Evaluation results")
summary = json.load(open(os.path.join(PROJECT, "mt", "results", "mt_results.json"), encoding="utf-8"))
for lang in ("hi", "mr"):
    r = summary.get(lang, {})
    check(f"[{lang}] BLEU, chrF, chrF++ computed on all 1000 test pairs",
          r.get("test_pairs") == 1000 and all(isinstance(r.get(k), float) for k in ("BLEU", "chrF", "chrF++")), r.keys())
    check(f"[{lang}] COMET either computed or the reason is recorded",
          r.get("COMET") is not None or "not installed" in r.get("COMET_status", ""))
    ep = r.get("entity_preservation", {}).get("counts", {})
    check(f"[{lang}] entity categories add up to the reference entity count",
          ep and ep["preserved"] + ep["translated_incorrectly"] + ep["dropped"] == ep["reference_entities"], ep)
    folder = os.path.join(PROJECT, "mt", "results", lang)
    lens = [sum(1 for _ in open(os.path.join(folder, n), encoding="utf-8")) for n in
            ("sources.txt", "references.txt", "hypotheses.txt")]
    check(f"[{lang}] sources / references / hypotheses saved (1000 lines each)", lens == [1000, 1000, 1000], lens)

print("\n=== NER models untouched")
expected = {}
with open(os.path.join(PROJECT, "training", "logs", "checksums_before_option_a.txt"), encoding="utf-8") as f:
    for line in f:
        digest, path = line.split(maxsplit=1)
        expected[path.strip().lstrip("*")] = digest
import hashlib  # noqa: E402
changed = [p for p, d in expected.items() if p.startswith("models/")
           and hashlib.sha256(open(os.path.join(PROJECT, p), "rb").read()).hexdigest() != d]
check("all NER model files identical to their recorded SHA-256", not changed, changed)

print(f"\nRESULT: {passes} passed, {len(failures)} failed")
for name in failures:
    print("  failed:", name)
sys.exit(1 if failures else 0)
