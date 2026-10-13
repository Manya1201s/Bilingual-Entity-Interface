"""
Evaluate English -> Hindi / Marathi entity alignment on the gold sets built
by alignment/build_gold.py.

    python alignment/evaluate_alignment.py

For every gold item the aligner ranks all candidate target spans
(alignment/aligner.py). Metrics (gold = Wikidata-verified target spans):

  Precision@1   top-1 candidate is exactly a gold span
  Accuracy      top-1 candidate overlaps a gold span (correct correspondence,
                boundary mismatches tolerated)
  Recall@K      a gold span is among the top K candidates (K = 1, 3, 5, 10)
  MRR           mean of 1 / rank of the first exact gold span (0 if absent)

Variants (identical candidates for all):
  random        expected value of a random ranking (seeded simulation)
  translit      transliteration similarity only
  mt            similarity to the project MT model's translation only
  ner           target-side NER agreement only
  combined      weighted sum; weights grid-searched on the VALIDATION gold
                set (best MRR), then applied once to the TEST set

Error analysis on TEST for the combined aligner:
  correct / boundary mismatch / wrong correspondence / missing (gold span not
  among the candidates) / items with multiple same-type candidate entities.

Outputs: alignment/results/<lang>/{metrics.json, examples.json} and
alignment/results/alignment_results.{json,md}
"""

import itertools
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
from alignment.aligner import Aligner  # noqa: E402

RESULTS = os.path.join(HERE, "results")
KS = (1, 3, 5, 10)
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def load(lang, split):
    with open(os.path.join(HERE, "data", lang, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def item_outcome(ranked, gold):
    gold = [tuple(g) for g in gold]
    rank = next((i + 1 for i, r in enumerate(ranked) if r["span"] in gold), None)
    top = ranked[0]["span"] if ranked else None
    return {"rank": rank, "top_exact": rank == 1,
            "top_overlap": top is not None and any(overlaps(top, g) for g in gold),
            "gold_in_candidates": rank is not None}


def metrics(outcomes):
    n = len(outcomes)
    out = {"items": n,
           "precision_at_1": sum(o["top_exact"] for o in outcomes) / n,
           "accuracy": sum(o["top_overlap"] for o in outcomes) / n,
           "mrr": sum(1 / o["rank"] for o in outcomes if o["rank"]) / n}
    for k in KS:
        out[f"recall_at_{k}"] = sum(bool(o["rank"]) and o["rank"] <= k for o in outcomes) / n
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}


def random_baseline(feats, golds, seed=0, trials=200):
    rng, outs = random.Random(seed), []
    for rows, gold in zip(feats, golds):
        for _ in range(trials):
            ranked = rows[:]
            rng.shuffle(ranked)
            outs.append(item_outcome(ranked, gold))
    return metrics(outs)


def evaluate_variant(aligner, feats, golds, weights):
    return [item_outcome(aligner.rank(rows, weights), gold) for rows, gold in zip(feats, golds)]


def tune(aligner, feats, golds):
    best, best_key = None, None
    for wt, wm, wn in itertools.product(GRID, GRID, GRID):
        if wt == wm == wn == 0:
            continue
        w = {"translit": wt, "mt": wm, "ner": wn}
        m = metrics(evaluate_variant(aligner, feats, golds, w))
        key = (m["mrr"], m["precision_at_1"], -(wt + wm + wn))
        if best_key is None or key > best_key:
            best, best_key = (w, m), key
    return best


def error_analysis(items, feats, extras, outcomes, ranked_all):
    cats = {"correct": 0, "boundary_mismatch": 0, "wrong_correspondence": 0}
    missing, examples = 0, {"correct": [], "boundary_mismatch": [], "wrong_correspondence": [], "missing": []}
    multi = {"items": 0, "correct": 0}
    single = {"items": 0, "correct": 0}
    by_type = {}
    for it, out, extra, ranked in zip(items, outcomes, extras, ranked_all):
        top = ranked[0]
        cat = "correct" if out["top_exact"] else ("boundary_mismatch" if out["top_overlap"] else "wrong_correspondence")
        cats[cat] += 1
        same_type = sum(1 for t, _, _ in extra["target_ner"] if t == it["entity"]["type"])
        group = multi if same_type >= 2 else single
        group["items"] += 1
        group["correct"] += out["top_exact"]
        t = by_type.setdefault(it["entity"]["type"], {"items": 0, "correct": 0})
        t["items"] += 1
        t["correct"] += out["top_exact"]
        record = {"english": it["entity"]["text"], "type": it["entity"]["type"],
                  "gold": [it["target"][a:b] for a, b in it["gold_spans"]],
                  "predicted": top["text"], "mt_translation": extra["mt_translation"],
                  "target": it["target"]}
        if not out["gold_in_candidates"]:
            missing += 1
            if len(examples["missing"]) < 8:
                examples["missing"].append(record)
        if len(examples[cat]) < 8:
            examples[cat].append(record)
    n = len(items)
    return {
        "categories": {k: {"count": v, "share": round(v / n, 4)} for k, v in cats.items()},
        "missing_gold_not_in_candidates": {"count": missing, "share": round(missing / n, 4)},
        "multiple_same_type_candidates": {**multi, "precision_at_1": round(multi["correct"] / max(multi["items"], 1), 4)},
        "single_or_no_same_type_candidate": {**single, "precision_at_1": round(single["correct"] / max(single["items"], 1), 4)},
        "by_type": {k: {**v, "precision_at_1": round(v["correct"] / v["items"], 4)} for k, v in by_type.items()},
    }, examples


def run(lang):
    t0 = time.time()
    aligner = Aligner(lang)
    data = {s: load(lang, s) for s in ("validation", "test")}
    feats, extras = {}, {}
    for split, items in data.items():
        feats[split], extras[split] = [], []
        for it in items:
            rows, extra = aligner.features(it)
            feats[split].append(rows)
            extras[split].append(extra)
    golds = {s: [it["gold_spans"] for it in items] for s, items in data.items()}

    weights, val_metrics = tune(aligner, feats["validation"], golds["validation"])
    variants = {"random": None, "translit": {"translit": 1}, "mt": {"mt": 1}, "ner": {"ner": 1},
                "combined": weights}
    test_metrics = {}
    for name, w in variants.items():
        if w is None:
            test_metrics[name] = random_baseline(feats["test"], golds["test"])
        else:
            test_metrics[name] = metrics(evaluate_variant(aligner, feats["test"], golds["test"], w))
    ranked = [aligner.rank(rows, weights) for rows in feats["test"]]
    outcomes = [item_outcome(r, g) for r, g in zip(ranked, golds["test"])]
    errors, examples = error_analysis(data["test"], feats["test"], extras["test"], outcomes, ranked)
    avg_cands = sum(len(r) for r in feats["test"]) / max(len(feats["test"]), 1)

    result = {"language_pair": f"en-{lang}",
              "gold_sets": {"validation_items": len(data["validation"]), "test_items": len(data["test"]),
                            "source": f"alignment/data/{lang}/{{validation,test}}.jsonl (Wikidata-verified)"},
              "average_candidates_per_item": round(avg_cands, 1),
              "tuned_weights_on_validation": weights, "validation_metrics_combined": val_metrics,
              "test_metrics": test_metrics, "test_error_analysis": errors,
              "models": {"ner": f"models/{'hindi_ner' if lang == 'hi' else 'marathi_ner'}",
                         "mt": f"models/mt_en_{lang}"},
              "seconds": round(time.time() - t0, 1)}
    folder = os.path.join(RESULTS, lang)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    c = test_metrics["combined"]
    print(f"[{lang}] test items {c['items']} | weights {weights} | Acc {c['accuracy']:.4f} P@1 {c['precision_at_1']:.4f} "
          f"R@5 {c['recall_at_5']:.4f} MRR {c['mrr']:.4f} | {result['seconds'] / 60:.1f} min", flush=True)
    return result


def main():
    results = {lang: run(lang) for lang in ("hi", "mr")}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "alignment_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    lines = ["| Language | Variant | Test items | Accuracy | Precision@1 | Recall@3 | Recall@5 | Recall@10 | MRR |",
             "|---|---|---|---|---|---|---|---|---|"]
    for lang, r in results.items():
        for name, m in r["test_metrics"].items():
            lines.append(f"| en→{lang} | {name} | {m['items']} | {m['accuracy']:.3f} | {m['precision_at_1']:.3f} "
                         f"| {m['recall_at_3']:.3f} | {m['recall_at_5']:.3f} | {m['recall_at_10']:.3f} | {m['mrr']:.3f} |")
    with open(os.path.join(RESULTS, "alignment_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
