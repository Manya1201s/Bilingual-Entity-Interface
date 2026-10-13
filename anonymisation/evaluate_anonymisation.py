"""
Evaluate entity anonymisation.

    python anonymisation/evaluate_anonymisation.py --part mono    (NER gold test sets + synthetic identifiers)
    python anonymisation/evaluate_anonymisation.py --part cross   (needs trained MT models + alignment results)

MONO - gold = human/CoNLL entity annotations of the NER test sets
(training/data/<language>/test.jsonl, read-only; sentence = tokens joined by
spaces). A token counts as anonymised if a placeholder span overlaps it.
  residual_identifier_rate      gold-entity tokens left visible / gold-entity tokens
  missed_anonymisation_rate     gold entities with NO token anonymised / gold entities
  partial_anonymisation_rate    gold entities only partly anonymised / gold entities
  false_positive_rate           anonymised tokens outside gold entities / anonymised tokens
  incorrect_anonymisation_rate  gold entities covered by an NER placeholder of the
                                WRONG type / gold entities covered by an NER placeholder
  non_entity_preservation       non-entity tokens left unchanged / non-entity tokens
  identity_consistency          over the whole test set with one registry: entity texts
                                that received >1 placeholder, placeholders shared by
                                >1 different entity text (both should be 0)
  synthetic identifiers         seeded template sentences with fake EMAIL / URL / PHONE /
                                AADHAAR / PAN (recall) + negative controls (years,
                                amounts, counts) that must stay unchanged

CROSS - gold = Wikidata-verified English->Hindi/Marathi entity pairs
(alignment/data/<lang>/test.jsonl). anonymise_pair() must give the English
entity and the gold target span the SAME placeholder.
  cross_lingual_consistency     pairs with identical placeholders / gold pairs
  target_gold_left_visible      gold target spans not anonymised / gold pairs

Results: anonymisation/results/anonymisation_{mono,cross}.json + .md
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
sys.path.insert(0, os.path.join(PROJECT, "training"))
from anonymisation.anonymise import PLACEHOLDER, Anonymiser, Registry, anonymise_pair, normalise  # noqa: E402
from common import get_entities, read_jsonl, split_path  # noqa: E402

RESULTS = os.path.join(HERE, "results")
LANGS = {"en": "english", "hi": "hindi", "mr": "marathi"}


def token_offsets(tokens):
    offsets, pos = [], 0
    for t in tokens:
        offsets.append((pos, pos + len(t)))
        pos += len(t) + 1
    return offsets


def evaluate_mono(lang):
    anon = Anonymiser(lang)
    rows = read_jsonl(split_path(LANGS[lang], "test"))
    registry = Registry()
    c = {"sentences": len(rows), "gold_entities": 0, "gold_entity_tokens": 0, "residual_tokens": 0,
         "missed_entities": 0, "partial_entities": 0, "anonymised_tokens": 0, "false_positive_tokens": 0,
         "ner_covered_entities": 0, "wrong_type_entities": 0, "non_entity_tokens": 0, "non_entity_changed": 0}
    text_to_ph, ph_to_text = {}, {}
    examples = {"missed": [], "false_positive": [], "wrong_type": []}
    for r in rows:
        tokens, text = r["tokens"], " ".join(r["tokens"])
        offs = token_offsets(tokens)
        spans = anon.detect(text)
        _, reps = anon.apply(text, spans, registry)
        for rep in reps:
            key = (rep["type"], normalise(rep["text"]))
            text_to_ph.setdefault(key, set()).add(rep["placeholder"])
            ph_to_text.setdefault(rep["placeholder"], set()).add(key)
        covered = [any(s["start"] < e and b < s["end"] for s in spans) for b, e in offs]
        gold_tok = [False] * len(tokens)
        for etype, a, b in get_entities(r["tags"]):
            c["gold_entities"] += 1
            idx = range(a, b)
            for i in idx:
                gold_tok[i] = True
            n_cov = sum(covered[i] for i in idx)
            c["gold_entity_tokens"] += b - a
            c["residual_tokens"] += (b - a) - n_cov
            if n_cov == 0:
                c["missed_entities"] += 1
                if len(examples["missed"]) < 8:
                    examples["missed"].append({"entity": " ".join(tokens[a:b]), "type": etype, "sentence": text})
            elif n_cov < b - a:
                c["partial_entities"] += 1
            ner_over = [s for s in spans if s["source"] == "ner" and s["start"] < offs[b - 1][1] and offs[a][0] < s["end"]]
            if ner_over:
                c["ner_covered_entities"] += 1
                if all(s["type"] != etype for s in ner_over):
                    c["wrong_type_entities"] += 1
                    if len(examples["wrong_type"]) < 8:
                        examples["wrong_type"].append({"entity": " ".join(tokens[a:b]), "gold": etype,
                                                       "placeholder_type": ner_over[0]["type"]})
        for i, cov in enumerate(covered):
            if cov:
                c["anonymised_tokens"] += 1
                if not gold_tok[i]:
                    c["false_positive_tokens"] += 1
                    if len(examples["false_positive"]) < 8:
                        examples["false_positive"].append({"token": tokens[i], "sentence": text})
            if not gold_tok[i]:
                c["non_entity_tokens"] += 1
                c["non_entity_changed"] += cov
    ratio = lambda a, b: round(a / b, 4) if b else 0.0  # noqa: E731
    metrics = {
        "residual_identifier_rate": ratio(c["residual_tokens"], c["gold_entity_tokens"]),
        "missed_anonymisation_rate": ratio(c["missed_entities"], c["gold_entities"]),
        "partial_anonymisation_rate": ratio(c["partial_entities"], c["gold_entities"]),
        "false_positive_rate": ratio(c["false_positive_tokens"], c["anonymised_tokens"]),
        "incorrect_anonymisation_rate": ratio(c["wrong_type_entities"], c["ner_covered_entities"]),
        "non_entity_preservation": ratio(c["non_entity_tokens"] - c["non_entity_changed"], c["non_entity_tokens"]),
        "identity_consistency": {
            "entity_texts_with_multiple_placeholders": sum(len(v) > 1 for v in text_to_ph.values()),
            "placeholders_shared_by_different_texts": sum(len(v) > 1 for v in ph_to_text.values()),
            "distinct_entities": len(text_to_ph)},
    }
    return {"language": lang, "test_set": f"training/data/{LANGS[lang]}/test.jsonl", "counts": c,
            "metrics": metrics, "examples": examples}


def synthetic(lang, n=200, seed=7):
    """Seeded synthetic identifiers (clearly fake values) + negative controls."""
    rng = random.Random(seed)
    templates = {
        "en": ["Contact {} for details.", "His number is {} now.", "Send the file to {} today."],
        "hi": ["विवरण के लिए {} पर संपर्क करें।", "उनका नंबर {} है।", "फ़ाइल {} पर भेजें।"],
        "mr": ["अधिक माहितीसाठी {} वर संपर्क करा.", "त्यांचा क्रमांक {} आहे.", "फाइल {} वर पाठवा."],
    }[lang]
    def value(kind):
        if kind == "EMAIL":
            return f"user{rng.randint(10, 999)}@example{rng.randint(1, 9)}.org"
        if kind == "URL":
            return f"https://www.example{rng.randint(1, 99)}.org/page{rng.randint(1, 9)}"
        if kind == "PHONE":
            first, rest = rng.randint(6, 9), f"{rng.randint(0, 10 ** 9 - 1):09d}"
            if rng.random() < 0.5:
                return f"{first}{rest}"                       # 9876543210
            return f"+91 {first}{rest[:4]} {rest[4:]}"        # +91 98765 43210
        if kind == "AADHAAR":
            return " ".join(f"{rng.randint(1000, 9999)}" for _ in range(3))
        return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(5)) + f"{rng.randint(1000, 9999)}" \
            + rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    anon = Anonymiser(lang)
    found = {k: [0, 0] for k in ("EMAIL", "URL", "PHONE", "AADHAAR", "PAN")}
    for i in range(n):
        kind = list(found)[i % 5]
        v = value(kind)
        text = rng.choice(templates).format(v)
        spans = anon.detect(text)
        start = text.index(v)
        hit = any(s["type"] == kind and s["start"] <= start and s["end"] >= start + len(v) for s in spans)
        found[kind][0] += hit
        found[kind][1] += 1
    controls = {"en": ["The match was played in 2019.", "He paid Rs 25000 for 3 tickets.", "Chapter 12 has 450 pages."],
                "hi": ["मैच 2019 में खेला गया।", "उन्होंने 25000 रुपये दिए।", "अध्याय 12 में 450 पृष्ठ हैं।"],
                "mr": ["सामना 2019 मध्ये खेळला गेला.", "त्यांनी 25000 रुपये दिले.", "प्रकरण 12 मध्ये 450 पाने आहेत."]}[lang]
    control_hits = sum(any(s["source"] == "regex" for s in anon.detect(t)) for t in controls)
    return {"recall_by_type": {k: round(f / t, 4) for k, (f, t) in found.items()},
            "items": n, "negative_controls": len(controls), "negative_controls_wrongly_anonymised": control_hits}


def evaluate_cross(lang):
    from alignment.aligner import Aligner
    weights = json.load(open(os.path.join(PROJECT, "alignment", "results", lang, "metrics.json"),
                             encoding="utf-8"))["tuned_weights_on_validation"]
    aligner = Aligner(lang)
    en_anon, tgt_anon = Anonymiser("en"), Anonymiser(lang)
    items = [json.loads(l) for l in open(os.path.join(PROJECT, "alignment", "data", lang, "test.jsonl"), encoding="utf-8")]
    same, visible, examples = 0, 0, []
    for it in items:
        out = anonymise_pair(it["en"], it["target"], lang, en_anon, tgt_anon, aligner, weights)
        ent = it["entity"]
        en_ph = next((r["placeholder"] for r in out["english_replacements"]
                      if r["start"] < ent["end"] and ent["start"] < r["end"]), None)
        tgt_ph = None
        for a, b in it["gold_spans"]:
            tgt_ph = tgt_ph or next((r["placeholder"] for r in out["target_replacements"]
                                     if r["start"] < b and a < r["end"]), None)
        ok = en_ph is not None and en_ph == tgt_ph
        same += ok
        visible += tgt_ph is None
        if len(examples) < 10:
            examples.append({"english": out["english"], "target": out["target"], "entity": ent["text"],
                             "gold": [it["target"][a:b] for a, b in it["gold_spans"]],
                             "english_placeholder": en_ph, "target_placeholder": tgt_ph, "consistent": ok})
    n = len(items)
    return {"language_pair": f"en-{lang}", "gold_pairs": n, "aligner_weights": weights,
            "cross_lingual_consistency": round(same / n, 4),
            "target_gold_left_visible": round(visible / n, 4), "examples": examples}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", choices=["mono", "cross"], required=True)
    args = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if args.part == "mono":
        results = {}
        for lang in ("en", "hi", "mr"):
            r = evaluate_mono(lang)
            r["synthetic_identifiers"] = synthetic(lang)
            results[lang] = r
            m = r["metrics"]
            print(f"[{lang}] residual {m['residual_identifier_rate']:.4f} missed {m['missed_anonymisation_rate']:.4f} "
                  f"partial {m['partial_anonymisation_rate']:.4f} FP {m['false_positive_rate']:.4f} "
                  f"wrong-type {m['incorrect_anonymisation_rate']:.4f} non-entity kept {m['non_entity_preservation']:.4f} "
                  f"| consistency {m['identity_consistency']} | synthetic {r['synthetic_identifiers']}", flush=True)
        lines = ["| Language | Test set | Residual identifier rate | Missed | Partial | False-positive rate "
                 "| Incorrect (wrong type) | Non-entity preserved | Synthetic ID recall (E/U/P/A/PAN) | Negative controls changed |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for lang, r in results.items():
            m, s = r["metrics"], r["synthetic_identifiers"]
            rec = "/".join(f"{s['recall_by_type'][k]:.2f}" for k in ("EMAIL", "URL", "PHONE", "AADHAAR", "PAN"))
            lines.append(f"| {lang} | {r['counts']['sentences']} sent., {r['counts']['gold_entities']} entities "
                         f"| {m['residual_identifier_rate']:.3f} | {m['missed_anonymisation_rate']:.3f} "
                         f"| {m['partial_anonymisation_rate']:.3f} | {m['false_positive_rate']:.3f} "
                         f"| {m['incorrect_anonymisation_rate']:.3f} | {m['non_entity_preservation']:.3f} | {rec} "
                         f"| {s['negative_controls_wrongly_anonymised']}/{s['negative_controls']} |")
        name = "anonymisation_mono"
    else:
        results = {lang: evaluate_cross(lang) for lang in ("hi", "mr")}
        for lang, r in results.items():
            print(f"[{lang}] cross-lingual consistency {r['cross_lingual_consistency']:.4f} "
                  f"gold target left visible {r['target_gold_left_visible']:.4f} ({r['gold_pairs']} pairs)", flush=True)
        lines = ["| Pair | Gold pairs | Same placeholder on both sides | Gold target span left visible |",
                 "|---|---|---|---|"]
        for lang, r in results.items():
            lines.append(f"| en-{lang} | {r['gold_pairs']} | {r['cross_lingual_consistency']:.3f} | {r['target_gold_left_visible']:.3f} |")
        name = "anonymisation_cross"
    with open(os.path.join(RESULTS, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS, f"{name}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
