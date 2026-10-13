"""
Cross-lingual entity aligner: English entity -> span in its Hindi / Marathi
translation.

For an English entity mention and the target sentence, every candidate span
is scored and the candidates are ranked:

  candidates  every run of 1-4 consecutive target words (not crossing
              punctuation) + the same span with a Hindi/Marathi case suffix
              removed from its last word (दिल्लीत -> दिल्ली)
  translit    similarity between the English name and a rule-based
              romanisation of the candidate (handles transliterated names)
  mt          similarity between the candidate and the English mention
              translated by the project's own MT model (models/mt_en_<lang>;
              handles translated names such as "Supreme Court")
  ner         1 if the candidate is exactly a same-type entity predicted by
              the project's Hindi/Marathi BiLSTM-CRF (no aliases),
              0.5 if it overlaps one, else 0
  position    tiny tie-breaker: relative position of the candidate vs the
              mention in its sentence

  score = w_translit*translit + w_mt*mt + w_ner*ner + 0.01*position
"""

import difflib
import os
import sys
import unicodedata

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
from trained_ner import load_predictor, tokenize_with_offsets  # noqa: E402

SUFFIXES = sorted(["मध्ये", "मधील", "मधून", "च्या", "ची", "चे", "चा", "ला", "ना", "ने", "नी",
                   "वर", "पासून", "पर्यंत", "कडे", "साठी", "तील", "त"], key=len, reverse=True)
NER_DIR = {"hi": "hindi_ner", "mr": "marathi_ner"}

# --- rule-based Devanagari romanisation -----------------------------------------
VOWELS = {"अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai",
          "ओ": "o", "औ": "au", "ऑ": "o", "ॲ": "e"}
SIGNS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o",
         "ौ": "au", "ॉ": "o", "ॅ": "e"}
CONSONANTS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j",
              "झ": "jh", "ञ": "n", "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t",
              "थ": "th", "द": "d", "ध": "dh", "न": "n", "प": "p", "फ": "f", "ब": "b", "भ": "bh",
              "म": "m", "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s",
              "ह": "h", "ळ": "l", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z", "ड़": "r", "ढ़": "rh",
              "फ़": "f", "य़": "y"}
NASAL = {"ं": "n", "ँ": "n", "ः": "h"}
VIRAMA, NUKTA = "्", "़"


def romanize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out, i = [], 0
    while i < len(text):
        ch = text[i]
        if i + 1 < len(text) and text[i + 1] == NUKTA and ch + NUKTA in CONSONANTS:
            ch, i = ch + NUKTA, i + 1
        if ch in CONSONANTS:
            out.append(CONSONANTS[ch])
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if nxt in SIGNS:
                out.append(SIGNS[nxt])
                i += 1
            elif nxt == VIRAMA:
                i += 1
            elif nxt and (nxt in CONSONANTS or nxt in NASAL or nxt == NUKTA):
                out.append("a")  # inherent vowel inside the word (schwa kept)
        elif ch in VOWELS:
            out.append(VOWELS[ch])
        elif ch in SIGNS:
            out.append(SIGNS[ch])
        elif ch in NASAL:
            out.append(NASAL[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
        elif ch.isspace():
            out.append(" ")
        i += 1
    return "".join(out)


def latin_key(text: str) -> str:
    s = "".join(c for c in text.lower() if c.isalpha() or c == " ")
    for a, b in (("ph", "f"), ("w", "v"), ("q", "k"), ("x", "ks"), ("ck", "k"), ("c", "k"), ("ee", "i"),
                 ("oo", "u"), ("y", "i")):
        s = s.replace(a, b)
    return " ".join(s.split())


def skeleton(text: str) -> str:
    s = "".join(c for c in text if c not in "aeiou ")
    return "".join(c for k, c in enumerate(s) if k == 0 or c != s[k - 1]).replace("h", "")


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio() if a and b else 0.0


def translit_score(english: str, candidate: str) -> float:
    e, c = latin_key(english), latin_key(romanize(candidate))
    return max(similarity(e, c), similarity(skeleton(e), skeleton(c)))


def deva_key(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", text) if c != NUKTA and not c.isspace())


# --- candidates --------------------------------------------------------------------
def candidates(sentence: str, max_words: int = 4):
    tokens = [t for t in tokenize_with_offsets(sentence)]
    spans = set()
    for i in range(len(tokens)):
        if not tokens[i][0][0].isalnum() and not unicodedata.category(tokens[i][0][0]).startswith("L"):
            continue
        for j in range(i, min(i + max_words, len(tokens))):
            word = tokens[j][0]
            if not (word[0].isalnum() or unicodedata.category(word[0]).startswith(("L", "M"))):
                break  # punctuation ends the run
            start, end = tokens[i][1], tokens[j][2]
            spans.add((start, end))
            for suffix in SUFFIXES:
                if word.endswith(suffix) and len(word) - len(suffix) >= 2:
                    cut = end - len(suffix)
                    if sentence[cut - 1] != VIRAMA:
                        spans.add((start, cut))
    return sorted(spans)


class Aligner:
    def __init__(self, lang: str, use_mt: bool = True):
        self.lang = lang
        self.ner = load_predictor(os.path.join(PROJECT, "models", NER_DIR[lang]))
        self.translator = None
        if use_mt:
            from mt.model import Translator
            self.translator = Translator(os.path.join(PROJECT, "models", f"mt_en_{lang}"))
        self._mt_cache = {}

    def mt_name(self, english: str) -> str:
        if self.translator is None:
            return ""
        if english not in self._mt_cache:
            self._mt_cache[english] = self.translator.translate([english], beam_size=4)[0]
        return self._mt_cache[english]

    def features(self, item):
        """Feature dicts for every candidate span of one evaluation item."""
        sentence, ent = item["target"], item["entity"]
        target_ents = [(e["entity_group"], e["start"], e["end"]) for e in self.ner.predict(sentence)]
        mt_name = deva_key(self.mt_name(ent["text"]))
        rel_e = ent["start"] / max(len(item["en"]), 1)
        rows = []
        for start, end in candidates(sentence):
            text = sentence[start:end]
            ner = 0.0
            for etype, s, e in target_ents:
                if etype == ent["type"] and (s, e) == (start, end):
                    ner = 1.0
                    break
                if etype == ent["type"] and s < end and start < e:
                    ner = max(ner, 0.5)
            rows.append({"span": (start, end), "text": text,
                         "translit": translit_score(ent["text"], text),
                         "mt": similarity(mt_name, deva_key(text)) if mt_name else 0.0,
                         "ner": ner,
                         "position": 1.0 - abs(start / max(len(sentence), 1) - rel_e)})
        return rows, {"target_ner": target_ents, "mt_translation": self._mt_cache.get(ent["text"], "")}

    @staticmethod
    def rank(rows, weights):
        def score(r):
            return (weights.get("translit", 0) * r["translit"] + weights.get("mt", 0) * r["mt"]
                    + weights.get("ner", 0) * r["ner"] + 0.01 * r["position"])
        return sorted(rows, key=lambda r: (-score(r), r["span"]))
