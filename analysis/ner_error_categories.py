"""Read-only: categorise test-set errors of the final BiLSTM-CRF models, with real examples."""
import os, sys, collections, re
ROOT = r"C:\Users\manya\OneDrive\Desktop\Bilingual_Entity_Interface"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "training"))
sys.stdout.reconfigure(encoding="utf-8")
from common import get_entities, read_jsonl, split_path
from trained_ner import load_predictor, normalize_word

MR_SUFFIX = ("मध्ये", "मधील", "मधून", "च्या", "ची", "चे", "चा", "ला", "ना", "ने", "नी", "वर", "पासून", "पर्यंत", "त", "तील", "कडे", "साठी")

def span(toks, e): return " ".join(toks[e[1]:e[2]])
def ctx(toks, e, w=4):
    a, b = max(0, e[1] - w), min(len(toks), e[2] + w)
    return ("… " if a else "") + " ".join(toks[a:e[1]]) + " [" + span(toks, e) + "] " + " ".join(toks[e[2]:b]) + (" …" if b < len(toks) else "")

for lang, folder in (("hindi", "hindi_ner"), ("marathi", "marathi_ner"), ("english", "english_ner_scratch")):
    rows = read_jsonl(split_path(lang, "test"))
    model = load_predictor(os.path.join(ROOT, "models", folder))
    preds = model.predict_tokens([r["tokens"] for r in rows])
    vocab = model.vocab.word2id
    ex = collections.defaultdict(list)
    counts = collections.Counter()
    for sid, (r, (ptags, probs)) in enumerate(zip(rows, preds)):
        toks = r["tokens"]
        g, p = set(get_entities(r["tags"])), set(get_entities(ptags))
        conf = lambda q: sum(probs[q[1]:q[2]]) / (q[2] - q[1])
        for e in g:
            if e in p:
                continue
            over = [q for q in p if q[1] < e[2] and e[1] < q[2]]
            unseen = [t for t in toks[e[1]:e[2]] if normalize_word(t) not in vocab]
            if not over:
                kind, pred_desc = "missed", "O (nothing predicted)"
            else:
                q = max(over, key=lambda q: min(q[2], e[2]) - max(q[1], e[1]))
                pred_desc = f"{q[0]} '{span(toks, q)}' ({conf(q):.2f})"
                if (q[1], q[2]) == (e[1], e[2]):
                    kind = "wrong type"
                elif q[0] != e[0]:
                    kind = "boundary+type"
                elif q[1] <= e[1] and q[2] >= e[2]:
                    kind = "too long"
                elif q[1] >= e[1] and q[2] <= e[2]:
                    kind = "too short"
                else:
                    kind = "shifted boundary"
            counts[kind] += 1
            item = (sid, f"gold {e[0]} '{span(toks, e)}' -> pred {pred_desc} | {ctx(toks, e)}")
            ex[kind].append(item)
            if "ORG" in (e[0], pred_desc[:3]):
                ex["ORG-related"].append(item)
            if unseen:
                ex["unseen word"].append((sid, item[1] + f"  {{unseen: {unseen}}}"))
            if e[2] - e[1] >= 3:
                ex["long (3+)"].append(item)
            last = toks[e[2] - 1]
            if lang == "marathi" and last.endswith(MR_SUFFIX):
                ex["lang: Marathi suffix in gold span"].append(item)
            if lang == "hindi" and re.search(r"[A-Za-z]", span(toks, e)):
                ex["lang: Latin script in Hindi"].append(item)
            if lang == "hindi" and "़" in span(toks, e):
                ex["lang: nukta spelling"].append(item)
            if lang == "english" and span(toks, e).isupper():
                ex["lang: ALL-CAPS"].append(item)
        for q in p:
            if not any(e[1] < q[2] and q[1] < e[2] for e in g):
                counts["spurious"] += 1
                item = (sid, f"gold O -> pred {q[0]} '{span(toks, q)}' ({conf(q):.2f}) | {ctx(toks, q)}")
                ex["spurious"].append(item)
                if q[0] == "ORG":
                    ex["ORG-related"].append(item)
                if lang == "english" and span(toks, q).isupper():
                    ex["lang: ALL-CAPS"].append(item)
        # candidate gold-label problems: confident exact-span type disagreement
        for e in g:
            for q in p:
                if (q[1], q[2]) == (e[1], e[2]) and q[0] != e[0] and conf(q) >= 0.9:
                    ex["gold-label suspects (confident type disagreement)"].append(
                        (sid, f"gold {e[0]} '{span(toks, e)}' -> pred {q[0]} ({conf(q):.2f}) | {ctx(toks, e, 6)}"))
    print(f"\n################ {lang.upper()}  error counts: {dict(counts.most_common())}")
    for k in sorted(ex):
        items = ex[k]
        print(f"\n--- {k}: {len(items)}")
        for _, s in items[:8]:
            print("   ", s)
