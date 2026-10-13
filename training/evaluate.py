"""
Evaluate the trained models and the baseline on the HELD-OUT test sets.

    python training/evaluate.py                   (all models)
    python training/evaluate.py --only hindi      (hindi | marathi | english | baseline)

Scoring: entity-level exact match (type and token span must both match),
micro-averaged precision / recall / F1 plus per-type F1 - the CoNLL
(conlleval / seqeval default) definition.

* Trained models are run on the gold tokens of training/data/<lang>/test.jsonl.
* The baseline Davlan/xlm-roberta-base-ner-hrl is run on the same sentences
  (tokens joined by spaces); its character spans are mapped back to tokens.
  Its DATE predictions are ignored (not part of the PER/ORG/LOC scope).
* No configured aliases and no post-processing are used: these are pure
  model scores.

Results are written to training/results/evaluation.json and .md.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ENTITY_TYPES, MODELS_DIR, RESULTS_DIR, entity_scores, read_jsonl, split_path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trained_ner import load_predictor  # noqa: E402

BASELINE = "Davlan/xlm-roberta-base-ner-hrl"

# key: (display name, dataset language, training method, model folder)
TRAINED = {
    "hindi": ("Hindi BiLSTM-CRF", "hindi", "BiLSTM-CRF trained from scratch", "hindi_ner"),
    "marathi": ("Marathi BiLSTM-CRF", "marathi", "BiLSTM-CRF trained from scratch", "marathi_ner"),
    "english_scratch": ("English BiLSTM-CRF", "english", "BiLSTM-CRF trained from scratch",
                        "english_ner_scratch"),
    # Historical experiment, kept for comparison (not the final English model).
    "english": ("English BERT", "english", "bert-base-cased fine-tuned (experiment)", "english_ner"),
}

TEST_SETS = {
    "hindi": "Naamapadam hi test (human-annotated)",
    "marathi": "Naamapadam mr test (human-annotated)",
    "english": "CoNLL-2003 test",
}


def evaluate_trained(key):
    name, language, method, folder = TRAINED[key]
    model_dir = os.path.join(MODELS_DIR, folder)
    rows = read_jsonl(split_path(language, "test"))
    predictor = load_predictor(model_dir)
    t0 = time.time()
    predictions = predictor.predict_tokens([r["tokens"] for r in rows])
    seconds = time.time() - t0
    scores = entity_scores([r["tags"] for r in rows], [tags for tags, _ in predictions])
    return result_row(name, language, method, scores, len(rows), seconds)


def evaluate_baseline(language, pipe):
    rows = read_jsonl(split_path(language, "test"))
    gold, pred = [], []
    t0 = time.time()
    for r in rows:
        tokens = r["tokens"]
        offsets, pos = [], 0
        for tok in tokens:
            offsets.append((pos, pos + len(tok)))
            pos += len(tok) + 1
        text = " ".join(tokens)
        tags = ["O"] * len(tokens)
        for ent in pipe(text):
            etype = ent["entity_group"]
            if etype not in ENTITY_TYPES:
                continue  # DATE etc.
            covered = [i for i, (s, e) in enumerate(offsets) if s < ent["end"] and e > ent["start"]]
            for k, i in enumerate(covered):
                if tags[i] == "O":
                    tags[i] = ("B-" if k == 0 else "I-") + etype
        gold.append(r["tags"])
        pred.append(tags)
    seconds = time.time() - t0
    return result_row("Baseline XLM-R", language, f"pretrained {BASELINE} (no training here)",
                      entity_scores(gold, pred), len(rows), seconds)


def result_row(name, language, method, scores, n, seconds):
    row = {"model": name, "language": language, "method": method, "test_set": TEST_SETS[language],
           "test_sentences": n, "seconds": round(seconds, 1),
           "precision": scores["micro"]["precision"], "recall": scores["micro"]["recall"],
           "f1": scores["micro"]["f1"], "support": scores["micro"]["support"]}
    for t in ENTITY_TYPES:
        row[f"{t}_f1"] = scores[t]["f1"]
        row[f"{t}_support"] = scores[t]["support"]
    print(f"  {name:20} {language:8} P {row['precision']:.4f} R {row['recall']:.4f} "
          f"F1 {row['f1']:.4f} | PER {row['PER_f1']:.4f} ORG {row['ORG_f1']:.4f} "
          f"LOC {row['LOC_f1']:.4f} ({n} sentences, {seconds:.0f}s)", flush=True)
    return row


def write_markdown(rows, path):
    lines = ["| Model | Language | Training method | Test set | Precision | Recall | F1 "
             "| PER F1 | ORG F1 | LOC F1 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    order = {"hindi": 0, "marathi": 1, "english": 2}
    for r in sorted(rows, key=lambda r: (order[r["language"]], r["model"].startswith("Baseline"))):
        lines.append(f"| {r['model']} | {r['language'].title()} | {r['method']} | {r['test_set']} "
                     f"({r['test_sentences']} sent.) | {r['precision']:.3f} | {r['recall']:.3f} "
                     f"| **{r['f1']:.3f}** | {r['PER_f1']:.3f} | {r['ORG_f1']:.3f} | {r['LOC_f1']:.3f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["hindi", "marathi", "english_scratch", "english", "baseline"])
    args = p.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "evaluation.json")
    results = {}
    if os.path.exists(json_path):
        with open(json_path, encoding="utf-8") as f:
            results = {(r["model"], r["language"]): r for r in json.load(f)}

    for key in TRAINED:
        if args.only in (None, key):
            row = evaluate_trained(key)
            results[(row["model"], row["language"])] = row

    if args.only in (None, "baseline"):
        from transformers import pipeline
        pipe = pipeline("token-classification", model=BASELINE, aggregation_strategy="simple")
        for language in ("hindi", "marathi", "english"):
            row = evaluate_baseline(language, pipe)
            results[(row["model"], row["language"])] = row

    rows = list(results.values())
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    write_markdown(rows, os.path.join(RESULTS_DIR, "evaluation.md"))
    print(f"written {json_path}")


if __name__ == "__main__":
    main()
