"""
Build reproducible English-Hindi and English-Marathi parallel subsets from
Samanantar (ai4bharat/samanantar, CC-BY-NC-4.0) WITHOUT downloading the
whole corpus.

    python mt/prepare_parallel.py

Samanantar is stored as Parquet files made of 1,000-row groups. A seeded
random sample of row groups is read remotely (HTTP range requests); only
those blocks are transferred.

Cleaning (every rule is counted in dataset_info.json):
  * empty side, wrong script (English must be Latin, target Devanagari)
  * abnormal length (words) or abnormal target/source character ratio
  * digit mismatch (numbers present on one side only, or different numbers)
    - catches the many misaligned mined pairs
  * source == target copies
  * duplicates: a pair is dropped if its English OR its Indic side was
    already seen (so no sentence can appear in two splits)
  * overlap with the project's NER data (Naamapadam was built from
    Samanantar's Indic side; such sentences are removed so the NER models
    used for entity-preservation evaluation never saw them in training)

Output: mt/data/<lang>/{train,validation,test}.jsonl rows
{"id": "<file>#<row group>#<idx>", "en": ..., "<lang>": ...} + dataset_info.json
"""

import argparse
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
DATA_DIR = os.path.join(HERE, "data")
NER_DATA = os.path.join(PROJECT, "training", "data")

REPO = "datasets/ai4bharat/samanantar"
FILES = {"hi": [f"hi/train-0000{i}-of-00008.parquet" for i in range(8)],
         "mr": [f"mr/train-0000{i}-of-00002.parquet" for i in range(2)]}
NER_LANG = {"hi": "hindi", "mr": "marathi"}

LIMITS = {"min_words": 3, "max_en_words": 50, "max_tgt_words": 60,
          "min_char_ratio": 0.6, "max_char_ratio": 1.8,
          "min_script_share": 0.85}

DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def norm_key(text: str) -> str:
    """Letters/digits only, lower-case: robust key for duplicate/overlap checks."""
    return "".join(ch for ch in text.lower() if ch.isalnum() or ord(ch) in range(0x900, 0x980))


def script_share(text: str, devanagari: bool) -> float:
    letters = [c for c in text if c.isalpha() or 0x900 <= ord(c) < 0x980]
    if not letters:
        return 0.0
    if devanagari:
        good = sum(0x900 <= ord(c) < 0x980 for c in letters)
    else:
        good = sum(c.isascii() for c in letters)
    return good / len(letters)


def numbers(text: str):
    return sorted(re.findall(r"\d+", text.translate(DEVANAGARI_DIGITS)))


def reject_reason(en: str, tgt: str):
    if not en or not tgt:
        return "empty"
    if script_share(en, devanagari=False) < LIMITS["min_script_share"]:
        return "english_not_latin"
    if script_share(tgt, devanagari=True) < LIMITS["min_script_share"]:
        return "target_not_devanagari"
    ne, nt = len(en.split()), len(tgt.split())
    if ne < LIMITS["min_words"] or nt < LIMITS["min_words"]:
        return "too_short"
    if ne > LIMITS["max_en_words"] or nt > LIMITS["max_tgt_words"]:
        return "too_long"
    ratio = len(tgt) / len(en)
    if not LIMITS["min_char_ratio"] <= ratio <= LIMITS["max_char_ratio"]:
        return "length_ratio"
    if numbers(en) != numbers(tgt):
        return "number_mismatch"
    if norm_key(en) == norm_key(tgt):
        return "copy"
    return None


def ner_sentence_keys(lang: str) -> set:
    keys = set()
    folder = os.path.join(NER_DATA, NER_LANG[lang])
    for split in ("train", "validation", "test"):
        with open(os.path.join(folder, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    keys.add(norm_key("".join(json.loads(line)["tokens"])))
    return keys


def collect(lang: str, needed: int, seed: int, ner_keys: set):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    handles = {}
    groups = []
    for name in FILES[lang]:
        f = fs.open(f"{REPO}/{name}", "rb")
        pf = pq.ParquetFile(f)
        handles[name] = pf
        groups += [(name, g) for g in range(pf.metadata.num_row_groups)]
    random.Random(seed).shuffle(groups)

    stats = {"row_groups_read": 0, "raw_pairs": 0, "rejected": {}, "duplicate": 0,
             "overlap_with_ner_data": 0}
    seen_en, seen_tgt, kept, used_groups = set(), set(), [], []
    for name, g in groups:
        if len(kept) >= needed:
            break
        table = handles[name].read_row_group(g, columns=["idx", "src", "tgt"]).to_pylist()
        stats["row_groups_read"] += 1
        used_groups.append(f"{name}#{g}")
        for row in table:
            stats["raw_pairs"] += 1
            en = " ".join(str(row["src"] or "").split())
            tgt = " ".join(str(row["tgt"] or "").split())
            reason = reject_reason(en, tgt)
            if reason:
                stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1
                continue
            ke, kt = norm_key(en), norm_key(tgt)
            if ke in seen_en or kt in seen_tgt:
                stats["duplicate"] += 1
                continue
            seen_en.add(ke)
            seen_tgt.add(kt)
            if kt in ner_keys:
                stats["overlap_with_ner_data"] += 1
                continue
            kept.append({"id": f"{name}#{g}#{row['idx']}", "en": en, lang: tgt})
    return kept, stats, used_groups


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--languages", nargs="+", default=["hi", "mr"])
    p.add_argument("--train", type=int, default=50_000)
    p.add_argument("--validation", type=int, default=1_000)
    p.add_argument("--test", type=int, default=1_000)
    p.add_argument("--seed", type=int, default=2024)
    args = p.parse_args()

    for lang in args.languages:
        needed = args.train + args.validation + args.test
        kept, stats, used_groups = collect(lang, needed, args.seed, ner_sentence_keys(lang))
        if len(kept) < needed:
            sys.exit(f"[{lang}] only {len(kept)} clean pairs found, {needed} needed")
        kept = kept[:needed]
        random.Random(args.seed).shuffle(kept)
        splits = {"test": kept[:args.test],
                  "validation": kept[args.test:args.test + args.validation],
                  "train": kept[args.test + args.validation:]}
        folder = os.path.join(DATA_DIR, lang)
        os.makedirs(folder, exist_ok=True)
        for split, rows in splits.items():
            with open(os.path.join(folder, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        info = {
            "source": {"dataset": "ai4bharat/samanantar", "url": "https://huggingface.co/datasets/ai4bharat/samanantar",
                       "license": "CC-BY-NC-4.0 (non-commercial research use)",
                       "files": FILES[lang], "note": "mined parallel corpus; references are noisy"},
            "language_pair": f"en-{lang}", "seed": args.seed, "filters": LIMITS,
            "sampling": "seeded random order of 1,000-row Parquet row groups; read until enough clean pairs",
            "stats": stats, "row_groups_used": used_groups,
            "splits": {s: len(r) for s, r in splits.items()},
        }
        with open(os.path.join(folder, "dataset_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        kept_share = len(kept) / stats["raw_pairs"]
        print(f"[{lang}] row groups read {stats['row_groups_read']}, raw {stats['raw_pairs']:,}, "
              f"kept {len(kept):,} ({kept_share:.0%}) | rejected {stats['rejected']} | "
              f"duplicates {stats['duplicate']} | NER overlap removed {stats['overlap_with_ner_data']} | "
              f"splits {info['splits']}", flush=True)


if __name__ == "__main__":
    main()
