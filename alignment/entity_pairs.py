"""
Entity-to-entity alignment between an English sentence and a Hindi or
Marathi sentence whose entities were extracted separately (app section 04).

Reuses the transliteration signal of alignment/aligner.py (rule-based
Devanagari romanisation + Latin normalisation + consonant skeleton) and
the entity types predicted by NER on both sides:

  translit   translit_score(english entity, target entity), 0..1, also
             tried with a Hindi/Marathi case suffix removed (as in aligner.py)
  type       same PER/LOC/ORG type on both sides (a gate, not a score)

The aligner's MT signal (the project's English->Hindi/Marathi translation
model) is not used here: translation was removed from the app, and that
model is unevaluated (validation BLEU < 5). Translated names such as
"India" / "भारत" therefore stay unmatched rather than being guessed.

Thresholds are not hand-picked: alignment/calibrate_pairs.py derives them
from the Wikidata-verified gold sets (validation split) and writes
alignment/results/pair_calibration.json, which this module reads:

  review_threshold   F1-optimal translit score on validation
  match_threshold    lowest score with validation precision >= 0.95

Assignment is one-to-one: pairs are taken in descending score order and
each entity is used at most once. Status of an assigned pair:

  Matched                 same type, score >= match_threshold, and no
                          competing candidate also >= match_threshold
  Needs Review            score in [review, match), a type mismatch,
                          or a competing candidate >= match_threshold
Entities without a pair are reported as unmatched, never dropped.
"""

import json
import os
from typing import Any, Dict, List, Optional

from alignment.aligner import SUFFIXES, VIRAMA, translit_score

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION_PATH = os.path.join(HERE, "results", "pair_calibration.json")

MATCHED = "Matched"
REVIEW = "Needs Review"


def pair_score(english: str, target: str):
    """
    (score, removed_suffix): translit_score of the English entity against
    the target entity, also tried with a Hindi/Marathi case suffix removed
    from its last word, as alignment/aligner.py does for its candidates
    (नोएडामध्ये -> नोएडा). Offsets of the entity itself are not changed.
    """
    best, removed = translit_score(english, target), ""
    for suffix in SUFFIXES:
        stem = target[:-len(suffix)]
        if (target.endswith(suffix) and len(target.split()[-1]) - len(suffix) >= 2
                and stem[-1:] != VIRAMA):
            score = translit_score(english, stem)
            if score > best:
                best, removed = score, suffix
    return best, removed


def load_calibration(path: str = CALIBRATION_PATH) -> Optional[Dict[str, Any]]:
    """The calibrated thresholds, or None if calibrate_pairs.py was not run."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def align_entities(
    source: List[Dict[str, Any]],
    target: List[Dict[str, Any]],
    match_threshold: float,
    review_threshold: float,
    source_language: str = "English",
    target_language: str = "Hindi",
) -> List[Dict[str, Any]]:
    """
    source / target: entity dicts as produced by app.extract_ner
    ("Entity", "Entity Type", "Start", "End", "Confidence", "Source").

    Returns one row per pair or unmatched entity, each holding the original
    entity dicts (unchanged, so offsets/confidence/provenance are kept),
    the score, the evidence text and the status.
    """
    scored = {
        (i, j): pair_score(s["Entity"], t["Entity"])
        for i, s in enumerate(source)
        for j, t in enumerate(target)
    }
    scores = {pair: value[0] for pair, value in scored.items()}

    # Candidates above the review threshold, best evidence first; on equal
    # scores a same-type pair is preferred, then text order.
    candidates = sorted(
        (pair for pair, score in scores.items() if score >= review_threshold),
        key=lambda pair: (
            -scores[pair],
            source[pair[0]]["Entity Type"] != target[pair[1]]["Entity Type"],
            pair,
        ),
    )

    used_source, used_target, rows = set(), set(), []

    for i, j in candidates:
        if i in used_source or j in used_target:
            continue
        used_source.add(i)
        used_target.add(j)

        s, t, score = source[i], target[j], scores[(i, j)]
        same_type = s["Entity Type"] == t["Entity Type"]
        competitors = [
            pair for pair, other in scores.items()
            if pair != (i, j) and (pair[0] == i or pair[1] == j)
            and other >= match_threshold
        ]

        reasons = []
        if score < match_threshold:
            reasons.append(f"below the match threshold ({match_threshold:.2f})")
        if not same_type:
            reasons.append(
                f"entity types differ ({s['Entity Type']} vs {t['Entity Type']})"
            )
        if competitors:
            names = ", ".join(
                f"{source[a]['Entity']} ↔ {target[b]['Entity']}" for a, b in competitors
            )
            reasons.append(f"competing candidate above the threshold: {names}")

        suffix = scored[(i, j)][1]
        evidence = (
            f"Name similarity {score:.2f}"
            + (f" (case suffix '{suffix}' ignored)" if suffix else "")
            + " · "
            + (f"same type ({s['Entity Type']})" if same_type else
               f"type {s['Entity Type']} ≠ {t['Entity Type']}")
        )

        rows.append({
            "source": s,
            "target": t,
            "score": round(score, 3),
            "evidence": evidence + (" · " + "; ".join(reasons) if reasons else ""),
            "status": MATCHED if not reasons else REVIEW,
        })

    def unmatched(best, other_language, partner_taken):
        if best is None:
            return f"No {other_language} entities to compare"
        if partner_taken:
            return (f"Best candidate (name similarity {best:.2f}) "
                    "is already paired with another entity")
        return (f"Best name similarity {best:.2f} is below the review "
                f"threshold ({review_threshold:.2f})")

    for i, s in enumerate(source):
        if i not in used_source:
            options = [(scores[(i, j)], j) for j in range(len(target))]
            best = max(options, default=None)
            rows.append({
                "source": s, "target": None,
                "score": None if best is None else round(best[0], 3),
                "evidence": unmatched(
                    None if best is None else best[0], target_language,
                    best is not None and best[0] >= review_threshold),
                "status": f"Unmatched {source_language} Entity",
            })

    for j, t in enumerate(target):
        if j not in used_target:
            options = [(scores[(i, j)], i) for i in range(len(source))]
            best = max(options, default=None)
            rows.append({
                "source": None, "target": t,
                "score": None if best is None else round(best[0], 3),
                "evidence": unmatched(
                    None if best is None else best[0], source_language,
                    best is not None and best[0] >= review_threshold),
                "status": f"Unmatched {target_language} Entity",
            })

    return rows
