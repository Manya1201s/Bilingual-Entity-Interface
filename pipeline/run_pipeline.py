"""
End-to-end research pipeline on representative English-Hindi / English-Marathi pairs.

    python pipeline/run_pipeline.py            (needs trained MT models + alignment results)

For every input parallel pair:
  1. NER / entity detection     English + target NER models, regex identifiers
  2. Entity alignment           alignment/aligner.py (weights tuned on validation)
  3. Entity anonymisation       anonymisation/anonymise.py, shared placeholders
                                for aligned entities, one registry per language
  4. MT / generation            models/mt_en_<lang>, beam 4, on the ANONYMISED
                                English sentence (and on the original, for 5)
  5. Evaluation                 placeholders preserved by MT; entity preservation of
                                the original translation (mt/evaluate_mt.py logic)
  6. Final output               pipeline/results/<lang>_outputs.jsonl

Automatic checks per example: no exception, no empty output, no corrupted text
(U+FFFD, control characters, a Devanagari vowel sign at a word start),
no leakage (a detected entity's original text still present in the anonymised
English/target or in the MT output of the anonymised sentence), and identity
consistency (one placeholder per entity text; aligned pairs share it).

Inputs: the first 15 items of each Wikidata-verified alignment TEST set
(real held-out pairs) + 2 clearly labelled synthetic demo pairs per language
(repeated names and fake phone/email values).
Summary: pipeline/results/pipeline_summary.json
"""

import hashlib
import json
import os
import re
import sys
import traceback
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)

from alignment.aligner import Aligner  # noqa: E402
from anonymisation.anonymise import PLACEHOLDER, Anonymiser, Registry, anonymise_pair, normalise  # noqa: E402
from mt.evaluate_mt import NER, entity_preservation  # noqa: E402
from mt.model import Translator  # noqa: E402

RESULTS = os.path.join(HERE, "results")
PROTECTED = ["models/hindi_ner/model.pt", "models/hindi_ner/vocab.json", "models/hindi_ner/config.json",
             "models/marathi_ner/model.pt", "models/marathi_ner/vocab.json", "models/marathi_ner/config.json"]

DEMO = {
    "hi": [("Priya Mehta met Rahul Gandhi in Delhi. Priya Mehta can be reached at 9876543210.",
            "प्रिया मेहता ने दिल्ली में राहुल गांधी से मुलाकात की। प्रिया मेहता से 9876543210 पर संपर्क किया जा सकता है।"),
           ("Write to aarav.sharma@example.org about the Infosys office in Pune.",
            "पुणे में इन्फोसिस कार्यालय के बारे में aarav.sharma@example.org पर लिखें।")],
    "mr": [("Priya Mehta met Devendra Fadnavis in Mumbai. Priya Mehta can be reached at 9876543210.",
            "प्रिया मेहता यांनी मुंबईत देवेंद्र फडणवीस यांची भेट घेतली. प्रिया मेहता यांच्याशी 9876543210 वर संपर्क साधता येईल."),
           ("Write to aarav.sharma@example.org about the Infosys office in Pune.",
            "पुण्यातील इन्फोसिस कार्यालयाबद्दल aarav.sharma@example.org वर लिहा.")],
}


def sha(path):
    return hashlib.sha256(open(os.path.join(PROJECT, path), "rb").read()).hexdigest()


def corrupted(text):
    problems = []
    if "�" in text:
        problems.append("replacement character")
    if any(unicodedata.category(c) == "Cc" for c in text):
        problems.append("control character")
    if re.search(r"(^|\s)[ा-्]", text):
        problems.append("vowel sign at word start")
    return problems


def leaked(original_texts, text):
    plain = PLACEHOLDER.sub(" ", text)
    return [t for t in original_texts if len(t) > 2 and t in plain]


def run(lang, before):
    weights = json.load(open(os.path.join(PROJECT, "alignment", "results", lang, "metrics.json"),
                             encoding="utf-8"))["tuned_weights_on_validation"]
    aligner, translator = Aligner(lang), Translator(os.path.join(PROJECT, "models", f"mt_en_{lang}"))
    en_anon, tgt_anon, tgt_ner = Anonymiser("en"), Anonymiser(lang), NER({"hi": "hindi_ner", "mr": "marathi_ner"}[lang])
    gold = [json.loads(l) for l in open(os.path.join(PROJECT, "alignment", "data", lang, "test.jsonl"), encoding="utf-8")]
    inputs, seen = [], set()
    for it in gold:
        if it["pair_id"] not in seen:
            seen.add(it["pair_id"])
            inputs.append(("held-out test pair " + it["pair_id"], it["en"], it["target"]))
        if len(inputs) == 15:
            break
    inputs += [("synthetic demo", en, tgt) for en, tgt in DEMO[lang]]

    registry = Registry()
    outputs, refs_for_eval, hyps_for_eval = [], [], []
    placeholder_total = placeholder_kept = 0
    for source, en, tgt in inputs:
        record = {"source": source, "input": {"english": en, "target": tgt}}
        try:
            pair = anonymise_pair(en, tgt, lang, en_anon, tgt_anon, aligner, weights, registry)
            mt_anon = translator.translate([pair["english"]], beam_size=4)[0]
            mt_plain = translator.translate([en], beam_size=4)[0]
            en_ph = PLACEHOLDER.findall(pair["english"])
            phs = [m.group() for m in PLACEHOLDER.finditer(pair["english"])]
            kept = [p for p in phs if p in mt_anon]
            placeholder_total += len(phs)
            placeholder_kept += len(kept)
            originals = [r["text"] for r in pair["english_replacements"] + pair["target_replacements"]]
            checks = {
                "empty_output": [k for k, v in (("anonymised_english", pair["english"]), ("anonymised_target", pair["target"]),
                                               ("mt_output", mt_anon)) if not v.strip()],
                "corrupted_text": {k: corrupted(v) for k, v in (("anonymised_target", pair["target"]), ("mt_output", mt_anon))
                                   if corrupted(v)},
                "leakage_anonymised_english": leaked(originals, pair["english"]),
                "leakage_anonymised_target": leaked(originals, pair["target"]),
                "leakage_mt_of_anonymised": leaked(originals, mt_anon),
            }
            record.update({
                "entities": {"english": [(r["type"], r["text"]) for r in pair["english_replacements"]],
                             "target": [(r["type"], r["text"]) for r in pair["target_replacements"]]},
                "alignment_links": pair["links"],
                "anonymised": {"english": pair["english"], "target": pair["target"]},
                "mt": {"of_anonymised_english": mt_anon, "of_original_english": mt_plain},
                "placeholders": {"in_english": len(phs), "preserved_in_mt": len(kept), "types": en_ph},
                "checks": checks,
                "ok": not any(checks[k] for k in checks),
            })
            refs_for_eval.append(tgt)
            hyps_for_eval.append(mt_plain)
        except Exception as exc:  # noqa: BLE001
            record.update({"ok": False, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()})
        outputs.append(record)

    # identity consistency over the whole run (one registry)
    text_to_ph, ph_to_text = {}, {}
    for (etype, key), ph in registry.ids.items():
        text_to_ph.setdefault((etype, key), set()).add(ph)
        ph_to_text.setdefault(ph, set()).add(key)
    multi_ph = [k for k, v in text_to_ph.items() if len(v) > 1]
    shared = {ph: sorted(v) for ph, v in ph_to_text.items() if len(v) > 1}
    counts, rates, _, _ = entity_preservation(refs_for_eval, hyps_for_eval, tgt_ner)
    after = {p: sha(p) for p in PROTECTED}

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, f"{lang}_outputs.jsonl"), "w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "language_pair": f"en-{lang}", "examples": len(outputs),
        "examples_ok": sum(r.get("ok", False) for r in outputs),
        "crashes": sum("error" in r for r in outputs),
        "examples_with_empty_output": sum(bool(r.get("checks", {}).get("empty_output")) for r in outputs),
        "examples_with_corrupted_text": sum(bool(r.get("checks", {}).get("corrupted_text")) for r in outputs),
        "examples_with_leakage": sum(any(r.get("checks", {}).get(k) for k in
                                         ("leakage_anonymised_english", "leakage_anonymised_target", "leakage_mt_of_anonymised"))
                                     for r in outputs),
        "alignment_links_shared_placeholders": sum(len(r.get("alignment_links", [])) for r in outputs),
        "identity": {"entity_texts_with_multiple_placeholders": len(multi_ph),
                     "placeholders_used_for_several_texts": shared},
        "placeholders_preserved_by_mt": {"preserved": placeholder_kept, "total": placeholder_total,
                                          "rate": round(placeholder_kept / max(placeholder_total, 1), 4)},
        "entity_preservation_original_translation": {"counts": counts, "rates": rates},
        "protected_ner_files_unchanged": all(before[p] == after[p] for p in PROTECTED),
    }
    print(f"[{lang}] {json.dumps({k: v for k, v in summary.items() if k != 'identity'}, ensure_ascii=False)}", flush=True)
    return summary


def main():
    before = {p: sha(p) for p in PROTECTED}
    summary = {lang: run(lang, before) for lang in ("hi", "mr")}
    with open(os.path.join(RESULTS, "pipeline_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
