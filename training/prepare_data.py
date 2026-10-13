"""
Download and normalise the NER datasets into training/data/<language>/.

    python training/prepare_data.py

Sources (downloaded from the Hugging Face Hub, cached in ~/.cache):
  Hindi, Marathi : ai4bharat/naamapadam  (CC0-1.0)
                   train/validation = machine-projected ("silver") labels,
                   test = manually annotated ("gold") labels.
  English        : tner/conll2003 (CoNLL-2003; annotations by the
                   University of Antwerp, text (c) Reuters - research use only)

Output per language: train.jsonl, validation.jsonl, test.jsonl with rows
{"tokens": [...], "tags": [...]} using the common label set, plus
dataset_info.json recording source, license, label mapping and counts.

The Hindi/Marathi training sets are large (985k / 455k sentences); a fixed
random subset (seeded) is used so CPU training finishes in reasonable time.
Training sentences that also occur in validation/test are removed.
"""

import argparse
import json
import os
import random
import sys
import zipfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA_DIR, LABELS, normalize_tag, repair_bio, write_jsonl  # noqa: E402

NAAMAPADAM = {
    "repo": "ai4bharat/naamapadam",
    "url": "https://huggingface.co/datasets/ai4bharat/naamapadam",
    "license": "CC0-1.0 (public domain dedication)",
    "paper": "Mhaske et al., 2023, 'Naamapadam: A Large-Scale Named Entity "
             "Annotated Data for Indic Languages' (ACL 2023)",
    "annotation": "train/validation: entities projected from English via "
                  "word alignment on the Samanantar parallel corpus "
                  "(machine-generated, 'silver'); test: manually annotated",
}

CONLL = {
    "repo": "tner/conll2003",
    "url": "https://huggingface.co/datasets/tner/conll2003",
    "license": "CoNLL-2003 annotations (University of Antwerp); the text is "
               "Reuters Corpus RCV1, (c) Reuters, available for research "
               "purposes under the NIST/Reuters agreement. Use for research "
               "and education only; do not redistribute the raw text.",
    "paper": "Tjong Kim Sang & De Meulder, 2003, 'Introduction to the "
             "CoNLL-2003 Shared Task'",
    "annotation": "manually annotated (train/validation/test)",
}


def hf_file(repo, filename):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, filename, repo_type="dataset")


def convert(tokens, raw_tags, stats):
    """Normalise labels and repair BIO; returns None for unusable rows."""
    if not tokens or len(tokens) != len(raw_tags):
        stats["dropped_empty_or_malformed"] += 1
        return None
    tags = [normalize_tag(t) for t in raw_tags]
    for raw, new in zip(raw_tags, tags):
        if str(raw).upper() not in ("O", new):
            stats["mapped_to_O"][str(raw)] += 1
    tags, changes = repair_bio(tags)
    stats["bio_repairs"] += changes
    return {"tokens": [str(t) for t in tokens], "tags": tags}


def new_stats():
    return {"dropped_empty_or_malformed": 0, "bio_repairs": 0,
            "mapped_to_O": Counter(), "removed_train_overlap": 0}


def naamapadam(lang, train_size, seed):
    archive = zipfile.ZipFile(hf_file(NAAMAPADAM["repo"], f"data/{lang}_IndicNER_v1.0.zip"))
    stats = {split: new_stats() for split in ("train", "validation", "test")}
    files = {"train": f"{lang}_train.json", "validation": f"{lang}_val.json",
             "test": f"{lang}_test.json"}
    out, original = {}, {}
    for split in ("validation", "test", "train"):
        rows = []
        with archive.open(files[split]) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        original[split] = len(rows)
        if split == "train":
            held_out = {tuple(r["tokens"]) for s in ("validation", "test") for r in out[s]}
            random.Random(seed).shuffle(rows)
            kept = []
            for r in rows:
                if len(kept) >= train_size:
                    break
                row = convert(r["words"], r["ner"], stats[split])
                if row is None:
                    continue
                if tuple(row["tokens"]) in held_out:
                    stats[split]["removed_train_overlap"] += 1
                    continue
                kept.append(row)
            out[split] = kept
        else:
            out[split] = [x for x in (convert(r["words"], r["ner"], stats[split]) for r in rows) if x]
    return out, stats, original


def conll(seed):
    label_names = json.load(open(hf_file(CONLL["repo"], "dataset/label.json"), encoding="utf-8"))
    id2label = {v: k for k, v in label_names.items()}
    stats = {split: new_stats() for split in ("train", "validation", "test")}
    out, original = {}, {}
    for split, name in (("validation", "valid"), ("test", "test"), ("train", "train")):
        rows = [json.loads(l) for l in open(hf_file(CONLL["repo"], f"dataset/{name}.json"),
                                            encoding="utf-8") if l.strip()]
        original[split] = len(rows)
        converted = (convert(r["tokens"], [id2label[t] for t in r["tags"]], stats[split]) for r in rows)
        out[split] = [x for x in converted if x]
    # The official test/validation sets are kept unchanged; training
    # sentences that also occur there (mostly sports-table boilerplate such
    # as "W L PCT GB") are removed from train to avoid leakage.
    held_out = {tuple(r["tokens"]) for s in ("validation", "test") for r in out[s]}
    kept = [r for r in out["train"] if tuple(r["tokens"]) not in held_out]
    stats["train"]["removed_train_overlap"] = len(out["train"]) - len(kept)
    out["train"] = kept
    return out, stats, {k: original[k] for k in ("train", "validation", "test")}


def label_counts(rows):
    return dict(Counter(t for r in rows for t in r["tags"]))


def save(language, source, data, stats, original, note):
    info = {"language": language, "source": source, "labels": LABELS,
            "label_mapping": "B-/I- PER, ORG, LOC kept; every other entity "
                             "type (e.g. MISC) -> O; invalid I- tags -> B- (conlleval rule)",
            "note": note, "splits": {}}
    for split, rows in data.items():
        n = write_jsonl(os.path.join(DATA_DIR, language, f"{split}.jsonl"), rows)
        s = stats[split]
        info["splits"][split] = {
            "original_sentences": original[split], "sentences": n,
            "tokens": sum(len(r["tokens"]) for r in rows),
            "dropped_empty_or_malformed": s["dropped_empty_or_malformed"],
            "removed_train_overlap": s["removed_train_overlap"],
            "bio_repairs": s["bio_repairs"],
            "mapped_to_O": dict(s["mapped_to_O"]),
            "label_counts": label_counts(rows)}
        print(f"  {language:8} {split:10} {n:7d} sentences "
              f"(original {original[split]}, dropped {s['dropped_empty_or_malformed']}, "
              f"overlap removed {s['removed_train_overlap']}, BIO repairs {s['bio_repairs']})")
    with open(os.path.join(DATA_DIR, language, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hindi-train", type=int, default=100_000)
    parser.add_argument("--marathi-train", type=int, default=150_000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    for language, code, size in (("hindi", "hi", args.hindi_train),
                                 ("marathi", "mr", args.marathi_train)):
        data, stats, original = naamapadam(code, size, args.seed)
        save(language, NAAMAPADAM, data, stats, original,
             f"train = random subset (seed {args.seed}) of {original['train']} "
             f"silver sentences; validation = full silver set; test = full gold set")
    data, stats, original = conll(args.seed)
    save("english", CONLL, data, stats, original,
         "MISC entities mapped to O (outside the PER/ORG/LOC scope)")


if __name__ == "__main__":
    main()
