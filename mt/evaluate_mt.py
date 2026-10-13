"""
Evaluate the trained English -> Hindi / Marathi models on the HELD-OUT test
split (mt/data/<lang>/test.jsonl).

    python mt/evaluate_mt.py                  (both languages)
    python mt/evaluate_mt.py --lang hi --beam 4

Metrics
  * BLEU, chrF and chrF++ with sacrebleu (corpus level; signatures saved).
  * COMET if the `comet` package (unbabel-comet) can be imported; otherwise
    the exact reason is recorded and BLEU/chrF are still reported.
  * Named-entity preservation, using the project's final NER models
    (BiLSTM-CRF, models/<language>_ner, NO aliases) on the reference and on
    the hypothesis. Anchored on reference entities:
        preserved            reference entity text appears verbatim (whole
                             word) in the hypothesis
        translated wrongly   not verbatim, but the hypothesis has an unmatched
                             entity of the same type (taken as its rendering)
        dropped              no same-type entity left in the hypothesis
        hallucinated/added   hypothesis entities left unmatched
    Rates for the first three are over all reference entities; the added
    rate is over all hypothesis entities. NER errors on either side affect
    these numbers (see README).

Outputs: mt/results/<lang>/{hypotheses.txt, references.txt, sources.txt,
metrics.json, entity_examples.json} and mt/results/mt_results.{json,md}.
"""

import argparse
import difflib
import json
import os
import re
import sys
import time
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from mt.model import Translator  # noqa: E402
from trained_ner import load_predictor  # noqa: E402

RESULTS = os.path.join(HERE, "results")
NER_DIR = {"hi": "hindi_ner", "mr": "marathi_ner", "en": "english_ner_scratch"}
LANG_NAME = {"hi": "Hindi", "mr": "Marathi"}


def comet_status():
    try:
        import comet  # noqa: F401
        return True, "available"
    except Exception as exc:  # noqa: BLE001
        return False, (f"package 'unbabel-comet' is not installed ({type(exc).__name__}: {exc}). "
                       "Installing it fails on this machine: it requires an older NumPy that has no "
                       "pre-built wheel for Python 3.14 on Windows ARM64, so pip tries to compile "
                       "NumPy and no C/C++ compiler (MSVC/gcc/clang) is installed "
                       "('meson: Unknown compiler(s)'). COMET can be computed elsewhere (e.g. "
                       "Colab) from the saved sources/hypotheses/references files.")


def contains_word(text, phrase):
    return re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text) is not None


def entity_preservation(refs, hyps, ner):
    counts = {"reference_entities": 0, "hypothesis_entities": 0, "preserved": 0,
              "translated_incorrectly": 0, "dropped": 0, "hallucinated_added": 0}
    by_type = {}
    examples = {"preserved": [], "translated_incorrectly": [], "dropped": [], "hallucinated_added": []}
    ref_ents_all = ner.predict_batch(refs)
    hyp_ents_all = ner.predict_batch(hyps)
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        ref_ents, hyp_ents = ref_ents_all[i], hyp_ents_all[i]
        counts["reference_entities"] += len(ref_ents)
        counts["hypothesis_entities"] += len(hyp_ents)
        unmatched_hyp = list(hyp_ents)
        pending = []
        for etype, text in ref_ents:
            t = by_type.setdefault(etype, {"reference": 0, "preserved": 0,
                                           "translated_incorrectly": 0, "dropped": 0})
            t["reference"] += 1
            if contains_word(hyp, text):
                counts["preserved"] += 1
                t["preserved"] += 1
                for h in unmatched_hyp:
                    if h[1] == text:
                        unmatched_hyp.remove(h)
                        break
                if len(examples["preserved"]) < 15:
                    examples["preserved"].append({"line": i, "type": etype, "reference": text})
            else:
                pending.append((etype, text))
        for etype, text in pending:
            same = [h for h in unmatched_hyp if h[0] == etype]
            if same:
                best = max(same, key=lambda h: difflib.SequenceMatcher(None, h[1], text).ratio())
                unmatched_hyp.remove(best)
                counts["translated_incorrectly"] += 1
                by_type[etype]["translated_incorrectly"] += 1
                if len(examples["translated_incorrectly"]) < 15:
                    examples["translated_incorrectly"].append(
                        {"line": i, "type": etype, "reference": text, "hypothesis": best[1]})
            else:
                counts["dropped"] += 1
                by_type[etype]["dropped"] += 1
                if len(examples["dropped"]) < 15:
                    examples["dropped"].append({"line": i, "type": etype, "reference": text,
                                                "hypothesis_sentence": hyp})
        for etype, text in unmatched_hyp:
            counts["hallucinated_added"] += 1
            if len(examples["hallucinated_added"]) < 15:
                examples["hallucinated_added"].append({"line": i, "type": etype, "hypothesis": text,
                                                       "reference_sentence": ref})
    n_ref = max(counts["reference_entities"], 1)
    n_hyp = max(counts["hypothesis_entities"], 1)
    rates = {"preserved_rate": counts["preserved"] / n_ref,
             "translated_incorrectly_rate": counts["translated_incorrectly"] / n_ref,
             "dropped_rate": counts["dropped"] / n_ref,
             "hallucinated_added_rate": counts["hallucinated_added"] / n_hyp}
    return counts, {k: round(v, 4) for k, v in rates.items()}, by_type, examples


class NER:
    def __init__(self, folder):
        self.predictor = load_predictor(os.path.join(PROJECT, "models", folder))

    def predict_batch(self, sentences):
        return [[(e["entity_group"], e["word"]) for e in self.predictor.predict(s)] for s in sentences]


def evaluate(lang, beam, limit):
    import sacrebleu
    rows = [json.loads(l) for l in open(os.path.join(HERE, "data", lang, "test.jsonl"), encoding="utf-8")]
    if limit:
        rows = rows[:limit]
    srcs, refs = [r["en"] for r in rows], [r[lang] for r in rows]
    model_dir = os.path.join(PROJECT, "models", f"mt_en_{lang}")
    translator = Translator(model_dir)
    t0 = time.time()
    hyps = translator.translate(srcs, beam_size=beam)
    decode_seconds = time.time() - t0

    bleu = sacrebleu.metrics.BLEU()
    chrf = sacrebleu.metrics.CHRF()
    chrfpp = sacrebleu.metrics.CHRF(word_order=2)
    scores = {"BLEU": bleu.corpus_score(hyps, [refs]), "chrF": chrf.corpus_score(hyps, [refs]),
              "chrF++": chrfpp.corpus_score(hyps, [refs])}
    comet_ok, comet_reason = comet_status()
    comet_score = None
    if comet_ok:
        from comet import download_model, load_from_checkpoint
        cm = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        out = cm.predict([{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)],
                         batch_size=16, gpus=0)
        comet_score = out.system_score

    counts, rates, by_type, examples = entity_preservation(refs, hyps, NER(NER_DIR[lang]))
    src_entities = sum(len(e) for e in NER(NER_DIR["en"]).predict_batch(srcs))

    folder = os.path.join(RESULTS, lang)
    os.makedirs(folder, exist_ok=True)
    for name, lines in (("sources.txt", srcs), ("references.txt", refs), ("hypotheses.txt", hyps)):
        with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    info = translator.info
    result = {
        "language_pair": f"en-{lang}", "test_file": f"mt/data/{lang}/test.jsonl", "test_pairs": len(rows),
        "decoding": f"beam search, beam {beam}, GNMT length penalty 0.6" if beam > 1 else "greedy",
        "decode_seconds": round(decode_seconds, 1),
        "BLEU": round(scores["BLEU"].score, 2), "chrF": round(scores["chrF"].score, 2),
        "chrF++": round(scores["chrF++"].score, 2),
        "signatures": {"BLEU": str(bleu.get_signature()), "chrF": str(chrf.get_signature()),
                       "chrF++": str(chrfpp.get_signature())},
        "COMET": round(comet_score, 4) if comet_score is not None else None,
        "COMET_status": "computed with Unbabel/wmt22-comet-da" if comet_ok else comet_reason,
        "entity_preservation": {"counts": counts, "rates": rates, "by_type": by_type,
                                "source_entities_english_ner": src_entities,
                                "ner_models": {"target": f"models/{NER_DIR[lang]}", "source": f"models/{NER_DIR['en']}"},
                                "aliases_used": False},
        "model": {"dir": f"models/mt_en_{lang}", "parameters": info["parameters"],
                  "architecture": info["model"], "best_epoch": info.get("best_epoch"),
                  "best_val_loss": info.get("best_val_loss"), "training": info["training"]},
    }
    with open(os.path.join(folder, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(os.path.join(folder, "entity_examples.json"), "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[{lang}] {len(rows)} test pairs | BLEU {result['BLEU']} chrF {result['chrF']} chrF++ {result['chrF++']} "
          f"| COMET {result['COMET']} | entities {counts} rates {rates} | decode {decode_seconds / 60:.1f} min", flush=True)
    return result


def write_summary(results):
    with open(os.path.join(RESULTS, "mt_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    lines = ["| Pair | Test pairs | BLEU | chrF | chrF++ | COMET | Entities preserved | "
             "Translated incorrectly | Dropped | Hallucinated/added |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for lang, r in sorted(results.items()):
        e, c = r["entity_preservation"]["rates"], r["entity_preservation"]["counts"]
        comet = f"{r['COMET']:.4f}" if r["COMET"] is not None else "not run (see metrics.json)"
        lines.append(f"| en→{lang} | {r['test_pairs']} | {r['BLEU']:.2f} | {r['chrF']:.2f} | {r['chrF++']:.2f} | {comet} "
                     f"| {c['preserved']}/{c['reference_entities']} ({e['preserved_rate']:.1%}) "
                     f"| {c['translated_incorrectly']} ({e['translated_incorrectly_rate']:.1%}) "
                     f"| {c['dropped']} ({e['dropped_rate']:.1%}) "
                     f"| {c['hallucinated_added']}/{c['hypothesis_entities']} ({e['hallucinated_added_rate']:.1%}) |")
    with open(os.path.join(RESULTS, "mt_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["hi", "mr"], nargs="*", default=["hi", "mr"])
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N test pairs (debugging)")
    args = p.parse_args()
    summary_path = os.path.join(RESULTS, "mt_results.json")
    results = json.load(open(summary_path, encoding="utf-8")) if os.path.exists(summary_path) else {}
    for lang in args.lang:
        results[lang] = evaluate(lang, args.beam, args.limit)
    if not args.limit:
        write_summary(results)


if __name__ == "__main__":
    main()
