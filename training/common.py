"""
Shared definitions for dataset preparation, training and evaluation.

One label scheme is used by all three trained models and matches the
entity types app.py understands (PER / LOC / ORG):

    O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC
"""

import json
import os
from typing import Dict, Iterable, List, Sequence, Tuple

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ENTITY_TYPES = ["PER", "ORG", "LOC"]

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(TRAINING_DIR)
DATA_DIR = os.path.join(TRAINING_DIR, "data")
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
RESULTS_DIR = os.path.join(TRAINING_DIR, "results")


def normalize_tag(tag: str) -> str:
    """
    Map a dataset tag to the common scheme. Entity types outside
    PER/ORG/LOC (e.g. CoNLL MISC) become O; this is documented in README.
    """
    tag = str(tag).strip().upper()
    if tag in ("", "O"):
        return "O"
    if "-" not in tag:
        return "O"
    prefix, entity = tag.split("-", 1)
    entity = {"PERSON": "PER", "LOCATION": "LOC", "ORGANIZATION": "ORG",
              "GPE": "LOC"}.get(entity, entity)
    if prefix not in ("B", "I") or entity not in ENTITY_TYPES:
        return "O"
    return f"{prefix}-{entity}"


def repair_bio(tags: Sequence[str]) -> Tuple[List[str], int]:
    """
    Make a BIO sequence valid: an I-X that does not follow B-X/I-X is
    turned into B-X (the standard conlleval interpretation). Returns the
    repaired tags and how many tags were changed.
    """
    fixed, changes, previous = [], 0, "O"
    for tag in tags:
        if tag.startswith("I-"):
            entity = tag[2:]
            if previous not in (f"B-{entity}", f"I-{entity}"):
                tag = f"B-{entity}"
                changes += 1
        fixed.append(tag)
        previous = tag
    return fixed, changes


def bio_errors(tags: Sequence[str]) -> List[str]:
    """Describe every BIO violation in a tag sequence."""
    errors, previous = [], "O"
    for i, tag in enumerate(tags):
        if tag not in LABEL2ID:
            errors.append(f"unknown label {tag!r} at {i}")
        elif tag.startswith("I-") and previous not in (f"B-{tag[2:]}", f"I-{tag[2:]}"):
            errors.append(f"{tag} at {i} without preceding B-/I-{tag[2:]}")
        previous = tag
    return errors


def get_entities(tags: Sequence[str]) -> List[Tuple[str, int, int]]:
    """
    Entity spans (type, start, end_exclusive) from a BIO sequence, with
    conlleval semantics (a stray I-X starts a new entity).
    """
    entities, current = [], None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, entity = (tag.split("-", 1) + [""])[:2] if tag != "O" else ("O", "")
        continues = (current is not None and prefix == "I" and entity == current[0])
        if not continues:
            if current is not None:
                entities.append((current[0], current[1], i))
                current = None
            if prefix in ("B", "I"):
                current = (entity, i)
    return entities


def entity_scores(gold: Iterable[Sequence[str]], pred: Iterable[Sequence[str]]) -> Dict:
    """
    Entity-level exact-match precision / recall / F1 (micro), plus per
    type, computed the same way as conlleval / seqeval default mode.
    """
    counts = {t: {"tp": 0, "gold": 0, "pred": 0} for t in ENTITY_TYPES}
    for g_tags, p_tags in zip(gold, pred):
        g = set(get_entities(g_tags))
        p = set(get_entities(p_tags))
        for entity_type in ENTITY_TYPES:
            gt = {e for e in g if e[0] == entity_type}
            pt = {e for e in p if e[0] == entity_type}
            counts[entity_type]["tp"] += len(gt & pt)
            counts[entity_type]["gold"] += len(gt)
            counts[entity_type]["pred"] += len(pt)

    def prf(tp, n_gold, n_pred):
        precision = tp / n_pred if n_pred else 0.0
        recall = tp / n_gold if n_gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "support": n_gold}

    result = {t: prf(**{"tp": c["tp"], "n_gold": c["gold"], "n_pred": c["pred"]})
              for t, c in counts.items()}
    total = {k: sum(c[k] for c in counts.values()) for k in ("tp", "gold", "pred")}
    result["micro"] = prf(total["tp"], total["gold"], total["pred"])
    return result


def read_jsonl(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[Dict]) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def split_path(language: str, split: str) -> str:
    return os.path.join(DATA_DIR, language, f"{split}.jsonl")
