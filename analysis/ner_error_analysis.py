"""Read-only error analysis of the final BiLSTM-CRF models on their test sets."""
import os, sys, collections
ROOT = r"C:\Users\manya\OneDrive\Desktop\Bilingual_Entity_Interface"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "training"))
sys.stdout.reconfigure(encoding="utf-8")
from common import get_entities, read_jsonl, split_path, entity_scores
from trained_ner import load_predictor, normalize_word

for lang, folder in (("hindi", "hindi_ner"), ("marathi", "marathi_ner"), ("english", "english_ner_scratch")):
    rows = read_jsonl(split_path(lang, "test"))
    pred_model = load_predictor(os.path.join(ROOT, "models", folder))
    preds = pred_model.predict_tokens([r["tokens"] for r in rows])
    vocab = pred_model.vocab.word2id
    cats = collections.Counter()
    confusion = collections.Counter()
    missed_by_type = collections.Counter()
    spurious_by_type = collections.Counter()
    oov = {"in": [0, 0], "oov": [0, 0]}          # [found, total]
    by_len = collections.defaultdict(lambda: [0, 0])
    examples = collections.defaultdict(collections.Counter)
    for r, (ptags, _) in zip(rows, preds):
        g = set(get_entities(r["tags"])); p = set(get_entities(ptags))
        toks = r["tokens"]
        text = lambda e: " ".join(toks[e[1]:e[2]])
        for e in g:
            hit = e in p
            key = "oov" if any(normalize_word(t) not in vocab for t in toks[e[1]:e[2]]) else "in"
            oov[key][0] += hit; oov[key][1] += 1
            n = min(e[2] - e[1], 3); by_len[n][0] += hit; by_len[n][1] += 1
            if hit:
                cats["correct"] += 1; continue
            over = [q for q in p if q[1] < e[2] and e[1] < q[2]]
            if not over:
                cats["missed"] += 1; missed_by_type[e[0]] += 1
                examples["missed " + e[0]][text(e)] += 1
            elif any((q[1], q[2]) == (e[1], e[2]) for q in over):
                q = [q for q in over if (q[1], q[2]) == (e[1], e[2])][0]
                cats["right span, wrong type"] += 1; confusion[(e[0], q[0])] += 1
                examples[f"{e[0]} -> {q[0]}"][text(e)] += 1
            elif any(q[0] == e[0] for q in over):
                cats["right type, wrong boundary"] += 1
                q = [q for q in over if q[0] == e[0]][0]
                examples["boundary " + e[0]][f"{text(e)}  => predicted: {text(q)}"] += 1
            else:
                cats["wrong boundary and type"] += 1
        for q in p:
            if not any(e[1] < q[2] and q[1] < e[2] for e in g):
                cats["spurious (no gold overlap)"] += 1; spurious_by_type[q[0]] += 1
                examples["spurious " + q[0]][text(q)] += 1
    total_gold = sum(v for k, v in cats.items() if not k.startswith("spurious"))
    s = entity_scores([r["tags"] for r in rows], [t for t, _ in preds])["micro"]
    print(f"\n######## {lang.upper()}  (recomputed F1 {s['f1']}, gold entities {total_gold})")
    for k in ("correct", "right span, wrong type", "right type, wrong boundary",
              "wrong boundary and type", "missed", "spurious (no gold overlap)"):
        v = cats[k]; print(f"  {k:28} {v:5}" + (f"  ({v / total_gold:.1%} of gold)" if not k.startswith("spurious") else ""))
    print("  type confusions (gold -> predicted):", dict(confusion.most_common()))
    print("  missed by type:", dict(missed_by_type), "| spurious by type:", dict(spurious_by_type))
    for k, (f, t) in oov.items():
        print(f"  recall, entities {'with an unseen word' if k == 'oov' else 'all words seen in training'}: {f}/{t} = {f / t:.1%}")
    for n in sorted(by_len):
        f, t = by_len[n]; print(f"  recall, {n}{'+' if n == 3 else ''}-word entities: {f}/{t} = {f / t:.1%}")
    for k in sorted(examples):
        if k.startswith(("ORG -> LOC", "LOC -> ORG", "PER -> ORG", "ORG -> PER", "missed ORG", "boundary ORG", "spurious")):
            print(f"  e.g. {k}: {[x for x, _ in examples[k].most_common(5)]}")
