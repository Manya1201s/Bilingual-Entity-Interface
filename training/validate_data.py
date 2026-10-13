"""
Validate the prepared datasets in training/data/<language>/.

    python training/validate_data.py

Checks every sentence for: empty sentences, missing tokens or labels
(token/tag count mismatch), empty or whitespace tokens, unknown labels,
invalid BIO sequences (I-X without a preceding B-X/I-X), malformed rows
(wrong keys / types), and duplicate sentences shared between train and
validation/test (data leakage). Exit code 1 if any problem is found.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, LABEL2ID, bio_errors  # noqa: E402

SPLITS = ("train", "validation", "test")


def validate_file(path):
    problems, sentences = [], []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            where = f"{os.path.basename(path)}:{line_no}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                problems.append(f"{where} malformed JSON ({exc})")
                continue
            tokens, tags = row.get("tokens"), row.get("tags")
            if not isinstance(tokens, list) or not isinstance(tags, list):
                problems.append(f"{where} malformed example (needs 'tokens' and 'tags' lists)")
                continue
            if not tokens:
                problems.append(f"{where} empty sentence")
                continue
            if len(tokens) != len(tags):
                problems.append(f"{where} {len(tokens)} tokens but {len(tags)} labels")
            for i, token in enumerate(tokens):
                if not isinstance(token, str) or not token.strip():
                    problems.append(f"{where} missing/blank token at {i}")
            for i, tag in enumerate(tags):
                if tag not in LABEL2ID:
                    problems.append(f"{where} missing/unknown label {tag!r} at {i}")
            problems += [f"{where} {e}" for e in bio_errors(tags) if "unknown label" not in e]
            sentences.append(tuple(tokens))
    return problems, sentences


def main():
    total = 0
    for language in sorted(os.listdir(DATA_DIR)):
        folder = os.path.join(DATA_DIR, language)
        if not os.path.isdir(folder):
            continue
        seen = {}
        for split in SPLITS:
            path = os.path.join(folder, f"{split}.jsonl")
            if not os.path.exists(path):
                print(f"  {language:8} {split:10} MISSING FILE")
                total += 1
                continue
            problems, sentences = validate_file(path)
            seen[split] = set(sentences)
            total += len(problems)
            print(f"  {language:8} {split:10} {len(sentences):7d} sentences, "
                  f"{len(problems)} problem(s)")
            for p in problems[:10]:
                print("      ", p)
        if "train" in seen:
            for split in ("validation", "test"):
                leak = len(seen["train"] & seen.get(split, set()))
                total += leak
                print(f"  {language:8} train/{split} overlap: {leak} sentence(s)")
    print("RESULT:", "OK - no problems found" if total == 0 else f"{total} problem(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
