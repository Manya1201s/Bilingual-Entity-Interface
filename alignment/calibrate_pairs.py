"""
Calibrate the entity-pair thresholds used by alignment/entity_pairs.py.

    python alignment/calibrate_pairs.py   ->  alignment/results/pair_calibration.json

Data: the Wikidata-verified gold sets (alignment/data/<lang>/{validation,
test}.jsonl, built by build_gold.py). For every gold item:
  positive pair  English entity  ↔  its gold Hindi/Marathi span
  negative pairs English entity  ↔  every other entity the project's
                 Hindi/Marathi BiLSTM-CRF finds in the same target sentence
                 (not overlapping a gold span) - the distractors the app
                 sees in section 04.
Score = pair_score (alignment/entity_pairs.py: translit_score, also with a
case suffix removed). No MT is used.

Thresholds are chosen on VALIDATION (hi + mr pooled) only:
  review_threshold   threshold with the best F1
  match_threshold    lowest threshold whose precision is >= 0.95
TEST metrics are reported once with those thresholds. Recall is limited
by translated names (e.g. "India" / "भारत"), which transliteration cannot
match; they are left unmatched in the app instead of guessed.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from alignment.aligner import NER_DIR  # noqa: E402
from alignment.entity_pairs import pair_score  # noqa: E402
from trained_ner import load_predictor  # noqa: E402

OUT = os.path.join(HERE, "results", "pair_calibration.json")
TARGET_PRECISION = 0.95
GRID = [round(0.30 + 0.01 * k, 2) for k in range(71)]  # 0.30 .. 1.00


def load(lang, split):
    with open(os.path.join(HERE, "data", lang, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pairs(lang, split, ner):
    """(score, is_positive) for every positive and negative pair."""
    out = []
    for item in load(lang, split):
        english, sentence = item["entity"]["text"], item["target"]
        gold = [tuple(g) for g in item["gold_spans"]]
        out.append((max(pair_score(english, sentence[a:b])[0] for a, b in gold), True))
        for entity in ner.predict(sentence):
            s, e = entity["start"], entity["end"]
            if not any(s < b and a < e for a, b in gold):
                out.append((pair_score(english, sentence[s:e])[0], False))
    return out


def prf(scored, threshold):
    tp = sum(1 for s, p in scored if p and s >= threshold)
    fp = sum(1 for s, p in scored if not p and s >= threshold)
    fn = sum(1 for s, p in scored if p and s < threshold)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    data = {}
    for lang in ("hi", "mr"):
        ner = load_predictor(os.path.join(PROJECT, "models", NER_DIR[lang]))
        for split in ("validation", "test"):
            data[(lang, split)] = pairs(lang, split, ner)
            print(f"{lang}/{split}: {sum(p for _, p in data[(lang, split)])} positive, "
                  f"{sum(not p for _, p in data[(lang, split)])} negative pairs", flush=True)

    validation = data[("hi", "validation")] + data[("mr", "validation")]
    review = max(GRID, key=lambda t: (prf(validation, t)["f1"], -t))
    match = next((t for t in GRID if t >= review
                  and prf(validation, t)["precision"] >= TARGET_PRECISION), 1.0)

    result = {
        "method": "translit_score (alignment/aligner.py) on Wikidata-verified gold pairs; "
                  "negatives = other target-side NER entities in the same sentence; "
                  "thresholds chosen on validation (hi+mr pooled), reported on test",
        "review_threshold": review,
        "match_threshold": match,
        "target_precision_for_match": TARGET_PRECISION,
        "validation": {"review": prf(validation, review), "match": prf(validation, match)},
        "test": {
            lang: {"review": prf(data[(lang, "test")], review),
                   "match": prf(data[(lang, "test")], match)}
            for lang in ("hi", "mr")
        },
        "pairs": {f"{lang}/{split}": {"positive": sum(p for _, p in v),
                                       "negative": sum(not p for _, p in v)}
                  for (lang, split), v in data.items()},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
