"""
Independently validate mt/data/<lang>/{train,validation,test}.jsonl.

    python mt/validate_parallel.py

Checks: malformed rows, empty samples, duplicate pairs / sentences inside a
split, sentences shared between splits (leakage), source-target mismatch
(wrong script, number mismatch, copies), abnormal lengths / length ratios,
overlap with the NER datasets, duplicate ids. Prints length statistics.
Exit code 1 if any problem is found.
"""

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_parallel import DATA_DIR, LIMITS, ner_sentence_keys, norm_key, reject_reason  # noqa: E402

SPLITS = ("train", "validation", "test")


def main():
    total_problems = 0
    for lang in ("hi", "mr"):
        folder = os.path.join(DATA_DIR, lang)
        if not os.path.isdir(folder):
            print(f"[{lang}] MISSING {folder}")
            total_problems += 1
            continue
        ner_keys = ner_sentence_keys(lang)
        keys = {}
        ids = set()
        for split in SPLITS:
            problems = []
            rows = []
            with open(os.path.join(folder, f"{split}.jsonl"), encoding="utf-8") as f:
                for n, line in enumerate(f, start=1):
                    try:
                        row = json.loads(line)
                        en, tgt, rid = row["en"], row[lang], row["id"]
                    except (json.JSONDecodeError, KeyError, TypeError) as exc:
                        problems.append(f"line {n}: malformed ({exc})")
                        continue
                    rows.append((n, en, tgt))
                    reason = reject_reason(en, tgt)
                    if reason:
                        problems.append(f"line {n}: {reason}")
                    if rid in ids:
                        problems.append(f"line {n}: duplicate id {rid}")
                    ids.add(rid)
                    for side, text in (("en", en), (lang, tgt)):
                        k = (side, norm_key(text))
                        if k in keys:
                            problems.append(f"line {n}: {side} sentence also in {keys[k]}")
                        else:
                            keys[k] = f"{split}:{n}"
                    if norm_key(tgt) in ner_keys:
                        problems.append(f"line {n}: target sentence occurs in NER data")
            en_len = [len(en.split()) for _, en, _ in rows]
            tg_len = [len(t.split()) for _, _, t in rows]
            ratio = [len(t) / len(e) for _, e, t in rows if e]
            print(f"[{lang}] {split:10} {len(rows):6} pairs | problems {len(problems)} | "
                  f"en words {min(en_len)}-{max(en_len)} (mean {statistics.mean(en_len):.1f}) | "
                  f"{lang} words {min(tg_len)}-{max(tg_len)} (mean {statistics.mean(tg_len):.1f}) | "
                  f"char ratio {min(ratio):.2f}-{max(ratio):.2f}")
            for p in problems[:8]:
                print("      ", p)
            total_problems += len(problems)
    print("limits:", LIMITS)
    print("RESULT:", "OK - no problems found" if total_problems == 0 else f"{total_problems} problem(s)")
    return 1 if total_problems else 0


if __name__ == "__main__":
    sys.exit(main())
