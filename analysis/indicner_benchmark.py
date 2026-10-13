"""
TEMPORARY evaluation script (outside the project). Read-only on project files.

Evaluates the public ai4bharat/IndicNER checkpoint on the project's existing
Hindi/Marathi test files with the project's own scorer (training/common.py:
entity_scores = entity-level exact match, conlleval/seqeval semantics), and
compares with our saved BiLSTM-CRF models on the identical sentences.

Inference: gold tokens -> tokenizer(is_split_into_words=True) -> label of the
FIRST sub-word of each word (same method as the IndicNER model card / Colab).
Outputs are written only next to this script.
"""
import json, os, random, sys, time, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
ROOT = r"C:\Users\manya\OneDrive\Desktop\Bilingual_Entity_Interface"
OUT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "training"))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
from common import ENTITY_TYPES, entity_scores, get_entities, read_jsonl, split_path
from trained_ner import load_predictor

torch.set_num_threads(8)
REPO = "ai4bharat/IndicNER"
tokenizer = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForTokenClassification.from_pretrained(REPO).eval()
id2label = {int(k): v for k, v in model.config.id2label.items()}
print("loaded", REPO, "| fast tokenizer:", tokenizer.is_fast, "| labels:", id2label,
      "| parameters:", f"{sum(p.numel() for p in model.parameters()):,}")


@torch.no_grad()
def indicner_tags(sentences, batch_size=16):
    out, stats = [], {"truncated_sentences": 0, "words_without_subword": 0}
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    result = [None] * len(sentences)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        enc = tokenizer([sentences[i] for i in idx], is_split_into_words=True, truncation=True,
                        max_length=512, padding=True, return_tensors="pt")
        logits = model(**{n: enc[n] for n in ("input_ids", "attention_mask", "token_type_ids") if n in enc}).logits
        for b, i in enumerate(idx):
            tags, seen = ["O"] * len(sentences[i]), set()
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is None or wid in seen:
                    continue
                seen.add(wid)
                tags[wid] = id2label[int(logits[b, pos].argmax())]
            missing = len(sentences[i]) - len(seen)
            if missing:
                stats["words_without_subword"] += missing
                if max(seen, default=-1) < len(sentences[i]) - 1 and len(enc["input_ids"][b]) >= 512:
                    stats["truncated_sentences"] += 1
            result[i] = tags
    return result, stats


def per_sentence(gold, pred):
    rows = []
    for g, p in zip(gold, pred):
        g, p = set(get_entities(g)), set(get_entities(p))
        rows.append((len(g & p), len(g), len(p)))
    return rows


def f1(rows):
    tp, n_gold, n_pred = (sum(r[i] for r in rows) for i in range(3))
    P, R = (tp / n_pred if n_pred else 0), (tp / n_gold if n_gold else 0)
    return 2 * P * R / (P + R) if P + R else 0.0


results = {}
for lang in ("hindi", "marathi"):
    data = read_jsonl(split_path(lang, "test"))
    words = [r["tokens"] for r in data]
    gold = [r["tags"] for r in data]
    t0 = time.time()
    ind_pred, stats = indicner_tags(words)
    secs = time.time() - t0
    ours_pred = [t for t, _ in load_predictor(os.path.join(ROOT, "models", f"{lang}_ner")).predict_tokens(words)]
    s_ind, s_ours = entity_scores(gold, ind_pred), entity_scores(gold, ours_pred)
    rows_ind, rows_ours = per_sentence(gold, ind_pred), per_sentence(gold, ours_pred)
    rng = random.Random(0)
    boot_ind, boot_ours, boot_diff = [], [], []
    for _ in range(1000):
        sample = [rng.randrange(len(data)) for _ in data]
        a, b = f1([rows_ours[i] for i in sample]), f1([rows_ind[i] for i in sample])
        boot_ours.append(a); boot_ind.append(b); boot_diff.append(a - b)
    ci = lambda xs: (round(sorted(xs)[25], 4), round(sorted(xs)[974], 4))
    results[lang] = {
        "test_file": f"training/data/{lang}/test.jsonl", "sentences_scored": len(data),
        "gold_entities": s_ind["micro"]["support"], "indicner_seconds": round(secs, 1), "inference_stats": stats,
        "indicner": s_ind, "ours": s_ours,
        "indicner_f1_95ci": ci(boot_ind), "ours_f1_95ci": ci(boot_ours),
        "diff_ours_minus_indicner": round(s_ours["micro"]["f1"] - s_ind["micro"]["f1"], 4),
        "diff_95ci_paired_bootstrap": ci(boot_diff),
        "share_of_resamples_ours_better": round(sum(d > 0 for d in boot_diff) / len(boot_diff), 3),
    }
    m = s_ind["micro"]
    print(f"\n[{lang}] {len(data)} sentences scored, {m['support']} gold entities, {secs:.0f}s, stats {stats}")
    print(f"  IndicNER : P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} | "
          + " ".join(f"{t} {s_ind[t]['f1']:.4f}" for t in ENTITY_TYPES) + f" | 95% CI {ci(boot_ind)}")
    o = s_ours["micro"]
    print(f"  Ours     : P {o['precision']:.4f} R {o['recall']:.4f} F1 {o['f1']:.4f} | "
          + " ".join(f"{t} {s_ours[t]['f1']:.4f}" for t in ENTITY_TYPES) + f" | 95% CI {ci(boot_ours)}")
    print(f"  Ours - IndicNER = {results[lang]['diff_ours_minus_indicner']:+.4f}  paired-bootstrap 95% CI {ci(boot_diff)}"
          f"  (ours better in {results[lang]['share_of_resamples_ours_better']:.0%} of resamples)")

with open(os.path.join(OUT, "indicner_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print("\nwritten", os.path.join(OUT, "indicner_results.json"))
