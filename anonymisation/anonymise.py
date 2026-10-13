"""
Entity anonymisation for English / Hindi / Marathi text.

Detection
  * named entities (PER / LOC / ORG) from the project's final BiLSTM-CRF NER
    models (models/<language>_ner, no aliases)
  * structured identifiers by regular expression:
    EMAIL, URL, PHONE (Indian mobile / 3-3-4 numbers), AADHAAR (12 digits),
    PAN (AAAAA9999A). Latin and Devanagari digits are both recognised.
  Regex matches take precedence over overlapping NER spans.

Replacement
  * every detected span becomes a typed placeholder: [PER_1], [LOC_2], ...
  * for Hindi/Marathi a case suffix attached to the entity's last word stays
    outside the placeholder (दिल्लीत -> [LOC_1]त), so grammar is preserved
  * consistent identity: a Registry maps (type, normalised text) to one
    placeholder, so repeated mentions of the same entity get the same id,
    within a sentence and across all sentences anonymised with that registry
  * parallel pairs: anonymise_pair() uses the cross-lingual aligner
    (alignment/aligner.py) to give an English entity and its Hindi/Marathi
    counterpart the SAME placeholder.
"""

import os
import re
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
from trained_ner import load_predictor  # noqa: E402

NER_DIR = {"en": "english_ner_scratch", "hi": "hindi_ner", "mr": "marathi_ner"}
SUFFIXES = sorted(["मध्ये", "मधील", "मधून", "च्या", "ची", "चे", "चा", "ला", "ना", "ने", "नी",
                   "वर", "पासून", "पर्यंत", "कडे", "साठी", "तील", "त"], key=len, reverse=True)
D = "[0-9०-९]"
PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("URL", re.compile(r"(?:https?://|www\.)\S+")),
    ("AADHAAR", re.compile(rf"(?<![0-9०-९]){D}{{4}}[ -]?{D}{{4}}[ -]?{D}{{4}}(?![0-9०-९])")),
    # Indian mobile: 10 digits, optionally +91 and a 5+5 grouping ("98765 43210")
    ("PHONE", re.compile(rf"(?<![0-9०-९])(?:\+91[ -]?)?[6-9६-९]{D}{{4}}[ -]?{D}{{5}}(?![0-9०-९])")),
    ("PHONE", re.compile(rf"(?<![0-9०-९]){D}{{3}}[ -]{D}{{3}}[ -]{D}{{4}}(?![0-9०-९])")),
    ("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")),
]
PLACEHOLDER = re.compile(r"\[(PER|LOC|ORG|EMAIL|URL|PHONE|AADHAAR|PAN)_\d+\]")


def normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(text.replace("़", "").split())


class Registry:
    """Consistent placeholder ids: one id per (type, normalised entity text)."""

    def __init__(self):
        self.ids = {}
        self.counters = {}

    def placeholder(self, etype: str, text: str) -> str:
        key = (etype, normalise(text))
        if key not in self.ids:
            self.counters[etype] = self.counters.get(etype, 0) + 1
            self.ids[key] = f"[{etype}_{self.counters[etype]}]"
        return self.ids[key]

    def link(self, etype: str, text: str, placeholder: str):
        """Force `text` to use an existing placeholder (cross-lingual identity)."""
        self.ids.setdefault((etype, normalise(text)), placeholder)


def strip_suffix(text: str, start: int, end: int, lang: str) -> int:
    if lang not in ("hi", "mr"):
        return end
    word = text[start:end].split()[-1] if text[start:end].split() else ""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 2 and text[end - len(suffix) - 1] != "्":
            return end - len(suffix)
    return end


class Anonymiser:
    def __init__(self, lang: str):
        self.lang = lang
        self.ner = load_predictor(os.path.join(PROJECT, "models", NER_DIR[lang]))

    def detect(self, text: str):
        """Non-overlapping spans: [{'start', 'end', 'type', 'source'}] sorted by start."""
        spans = []
        for etype, pattern in PATTERNS:
            for m in pattern.finditer(text):
                spans.append({"start": m.start(), "end": m.end(), "type": etype, "source": "regex"})
        for e in self.ner.predict(text):
            end = strip_suffix(text, e["start"], e["end"], self.lang)
            spans.append({"start": e["start"], "end": end, "type": e["entity_group"], "source": "ner"})
        kept = []
        for s in sorted(spans, key=lambda s: (s["source"] != "regex", -(s["end"] - s["start"]), s["start"])):
            if s["end"] > s["start"] and not any(s["start"] < k["end"] and k["start"] < s["end"] for k in kept):
                kept.append(s)
        return sorted(kept, key=lambda s: s["start"])

    def apply(self, text: str, spans, registry: Registry):
        out, pos, replacements = [], 0, []
        for s in sorted(spans, key=lambda s: s["start"]):
            original = text[s["start"]:s["end"]]
            ph = s.get("placeholder") or registry.placeholder(s["type"], original)
            out.append(text[pos:s["start"]])
            out.append(ph)
            pos = s["end"]
            replacements.append({**s, "text": original, "placeholder": ph})
        out.append(text[pos:])
        return "".join(out), replacements

    def anonymise(self, text: str, registry: Registry = None):
        registry = registry or Registry()
        return self.apply(text, self.detect(text), registry)


def anonymise_pair(english: str, target: str, lang: str, en_anon: Anonymiser, tgt_anon: Anonymiser,
                   aligner, weights, registry: Registry = None):
    """
    Anonymise a parallel pair with shared identities. Each English entity is
    aligned (top-1 of the aligner); if that target span overlaps a detected
    target span of the same type, both receive the same placeholder.
    """
    registry = registry or Registry()
    en_spans = en_anon.detect(english)
    tgt_spans = tgt_anon.detect(target)
    links = []
    for s in en_spans:
        if s["source"] != "ner":
            continue
        item = {"en": english, "target": target,
                "entity": {"text": english[s["start"]:s["end"]], "type": s["type"],
                           "start": s["start"], "end": s["end"]}}
        rows, _ = aligner.features(item)
        if not rows:
            continue
        top = aligner.rank(rows, weights)[0]["span"]
        for t in tgt_spans:
            if t["type"] == s["type"] and t["start"] < top[1] and top[0] < t["end"] and "placeholder" not in t:
                ph = registry.placeholder(s["type"], english[s["start"]:s["end"]])
                s["placeholder"] = t["placeholder"] = ph
                registry.link(t["type"], target[t["start"]:t["end"]], ph)
                links.append((english[s["start"]:s["end"]], target[t["start"]:t["end"]], ph))
                break
    en_text, en_reps = en_anon.apply(english, en_spans, registry)
    tgt_text, tgt_reps = tgt_anon.apply(target, tgt_spans, registry)
    return {"english": en_text, "target": tgt_text, "english_replacements": en_reps,
            "target_replacements": tgt_reps, "links": links}
