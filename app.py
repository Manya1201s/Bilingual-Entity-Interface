import io
import logging
import os
import re
import unicodedata
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

from ui_style import APP_CSS
from sentence_generation import (
    ENTITY_TYPES, build_docx, build_txt, entity_key, generate_sentences,
)

# Technical details (exceptions, missing files) go to the log; the page
# shows a short explanation.
LOGGER = logging.getLogger("indic_entity_lab")

# Cross-lingual entity alignment (alignment/entity_pairs.py). If it cannot
# be imported, NER still works and section 04 says why alignment is missing.
try:
    from alignment.entity_pairs import (
        MATCHED, REVIEW, align_entities, load_calibration,
    )
    ALIGNMENT_IMPORT_ERROR = None
except Exception as exc:
    ALIGNMENT_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# transformers (+ torch) is required for NER. If it is missing, the UI
# still starts and shows install instructions instead of a traceback.
try:
    from transformers import AutoConfig, pipeline
    TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:
    AutoConfig = None
    pipeline = None
    TRANSFORMERS_IMPORT_ERROR = exc

# Optional document readers
try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "Davlan/xlm-roberta-base-ner-hrl"
# A Hindi MuRIL check (MichaelHuang/muril_base_cased_hindi_ner) used to run
# next to the baseline. It is no longer configured: its config.json has 23
# generic labels (LABEL_0 … LABEL_22) and no documented mapping to entity
# types, so it could never be used and only produced a warning. For Hindi,
# choose "Language-specific trained models" (models/hindi_ner).

# Models trained for this project (see training/ and README.md).
# With "Language-specific trained models" the selected input language
# routes to one of them; the baseline option keeps the multilingual model
# above ("Auto Detect" route: Hindi and Marathi share
# the Devanagari script, so they cannot be told apart automatically).
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINED_MODELS = {
    # English BERT (models/english_ner) is kept only as a reference
    # experiment in training/results; it is not routed in the app.
    "English": ("english_ner_scratch", "English BiLSTM-CRF (scratch)"),
    "Hindi": ("hindi_ner", "Hindi BiLSTM-CRF (scratch)"),
    "Marathi": ("marathi_ner", "Marathi BiLSTM-CRF (scratch)"),
}
BASELINE_SOURCE = "Multilingual NER model"

# NER model choice shared by every extraction section; the first option is
# the default.
NER_ENGINES = [
    "Baseline multilingual model",
    "Language-specific trained models",
]
TRAINED_ENGINE = NER_ENGINES[1]

# Input languages offered by the NER sections.
NER_LANGUAGES = ["English", "Hindi", "Marathi"]

# Language pairs of the entity alignment section: label -> (language of
# the first sentence, language of the second sentence). Both sentences are
# typed by the user; nothing is translated.
LANGUAGE_PAIRS = {
    "English → Hindi": ("English", "Hindi"),
    "English → Marathi": ("English", "Marathi"),
    "Hindi → English": ("Hindi", "English"),
    "Marathi → English": ("Marathi", "English"),
}

# Example sentence per language (input placeholders).
EXAMPLE_SENTENCES = {
    "English": "Aarav Sharma works at Infosys in Pune.",
    "Hindi": "आरव शर्मा पुणे में इन्फोसिस में काम करते हैं।",
    "Marathi": "आरव शर्मा पुण्यात इन्फोसिसमध्ये काम करतात।",
}

# Entity types shown by the app. The default model
# (Davlan/xlm-roberta-base-ner-hrl) predicts DATE, PER, ORG and LOC.
# DATE is intentionally discarded: the project scope is PER/LOC/ORG and
# the model's DATE output is unreliable for Hindi/Marathi. MONEY/EVENT
# are not predicted by the model at all, so they are not offered.
VALID_LABELS = {"PER", "LOC", "ORG"}

st.set_page_config(
    page_title="Indic Entity Lab",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL LOADING
# ============================================================

TRANSFORMERS_INSTALL_HINT = (
    "Install it in the Python environment that runs Streamlit: "
    "python -m pip install transformers torch sentencepiece"
)


@st.cache_resource(show_spinner="Loading multilingual NER model...")
def load_model(model_name: str):
    if pipeline is None:
        raise ImportError(
            f"transformers is not installed ({TRANSFORMERS_IMPORT_ERROR}). "
            + TRANSFORMERS_INSTALL_HINT
        )

    # Check the label names first (config only, no weights): a model whose
    # labels are LABEL_0 ... LABEL_N cannot be mapped to PER/LOC/ORG, and
    # guessing a mapping would turn its output into fabricated labels.
    config = AutoConfig.from_pretrained(model_name)

    if not model_has_entity_labels(config.id2label or {}):
        raise ValueError(
            f"{model_name} is not usable: its labels are unnamed "
            "(LABEL_0 … LABEL_N), so they cannot be mapped to PER/LOC/ORG. "
            "Use a model whose config.json names its entity labels."
        )

    return pipeline(
        "token-classification",
        model=model_name,
        aggregation_strategy="simple",
    )


@st.cache_resource(show_spinner="Loading trained NER model...")
def load_trained_model(language: str):
    """
    Returns (predictor, None) for the trained model of `language`, or
    (None, reason) when it is missing or cannot be loaded, so the caller
    can fall back to the baseline model instead of crashing.
    """
    folder, _ = TRAINED_MODELS[language]
    model_dir = os.path.join(PROJECT_DIR, "models", folder)

    if not os.path.isdir(model_dir):
        script = (
            "train_english_scratch.py" if language == "English"
            else f"train_{language.lower()}.py"
        )
        return None, (
            f"models/{folder} was not found (train it with "
            f"training/{script})"
        )

    try:
        from trained_ner import load_predictor
        return load_predictor(model_dir), None
    except Exception as exc:
        return None, f"it could not be loaded ({type(exc).__name__}: {exc})"


def model_label(language: str) -> str:
    """Human-readable name of the model that processes `language`."""
    if language in TRAINED_MODELS:
        return TRAINED_MODELS[language][1]
    return f"Baseline {MODEL_NAME}"


def ner_route(language: str, engine: str) -> str:
    """Language key passed to extract_ner for `language` with `engine`."""
    return language if engine == TRAINED_ENGINE else "Auto Detect"


def model_has_entity_labels(id2label: Dict[Any, str]) -> bool:
    """True when at least one label name maps to PER/LOC/ORG."""
    return any(
        normalize_label(label) in VALID_LABELS
        for label in id2label.values()
    )


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(raw_label: str) -> str:
    label = str(raw_label or "").upper().strip()

    for prefix in ("B-", "I-", "E-", "S-", "U-"):
        if label.startswith(prefix):
            label = label[len(prefix):]

    mapping = {
        "PERSON": "PER",
        "PER": "PER",
        "LOCATION": "LOC",
        "LOC": "LOC",
        "GPE": "LOC",
        "ORGANIZATION": "ORG",
        "ORG": "ORG",
    }

    return mapping.get(label, label)


# ============================================================
# LANGUAGE / SCRIPT HELPERS
# ============================================================

def contains_devanagari(text: str) -> bool:
    return bool(re.search(r"[\u0900-\u097F]", text))


def escape_markdown(text: str) -> str:
    """
    Show user text literally. st.write/st.markdown would otherwise
    interpret characters such as $ * _ # as Markdown/LaTeX
    (e.g. "$5 and $10" rendered as a formula). Devanagari is untouched.
    """
    return re.sub(r"([!-/:-@\[-`{-~])", r"\\\1", text)


LANGUAGE_HELP = (
    "The language of the input text. With language-specific models, it "
    "selects the model trained for that language."
)

def language_selector(key: str, label: str, options=NER_LANGUAGES) -> str:
    """Input-language dropdown used by the NER sections."""
    return st.selectbox(
        label,
        options,
        key=key,
        help=LANGUAGE_HELP,
    )


def fixed_model(engine: str) -> str:
    """
    Show the NER model a workflow uses as static text (no dropdown) and
    return that NER_ENGINES value, so the text always names the model that
    is actually used.
    """
    st.markdown(
        f'<div class="static-field">{engine}</div>',
        unsafe_allow_html=True,
    )
    return engine


def language_mismatch(selected_language: str, script: str) -> bool:
    """True when the chosen language clearly contradicts the script."""
    if selected_language == "English":
        return script.startswith("Hindi/Marathi")

    if selected_language in ("Hindi", "Marathi"):
        return script.startswith("English")

    return False


def detect_script(text: str) -> str:
    devanagari_count = sum(
        "\u0900" <= char <= "\u097F" for char in text
    )
    latin_count = sum(
        ("A" <= char <= "Z") or ("a" <= char <= "z")
        for char in text
    )

    if devanagari_count > latin_count:
        return "Hindi/Marathi script (Devanagari)"

    if latin_count:
        return "English/Latin script"

    return "Unknown"


# ============================================================
# ENTITY POST-PROCESSING
# ============================================================

# Targeted aliases for the examples being tested.
# This is NOT a complete dictionary of all people, places or companies.
# Add verified aliases here only when required by your dataset.
ENTITY_ALIASES = {
    "PER": [
        "प्रिया",
    ],
    "ORG": [
        "इन्फोसिस लिमिटेड",
        "इन्फोसिस",
        "माइक्रोसॉफ्ट",
        "टाटा कंसल्टेंसी सर्विसेज",
        "टाटा कंसल्टेंसी सर्विस",
        # Marathi spellings
        "मायक्रोसॉफ्ट",
        "टाटा कन्सल्टन्सी सर्व्हिसेस",
    ],
    "LOC": [
        # Geneva: English, Hindi and Marathi spellings
        "Geneva",
        "जेनेवा",
        "जिनेवा",
        "जिनेव्हा",
        "जिनिव्हा",
    ],
}


# Titles should not be treated as person names on their own.
PERSON_TITLES = {
    "dr",
    "dr.",
    "mr",
    "mr.",
    "mrs",
    "mrs.",
    "ms",
    "ms.",
    "prof",
    "prof.",
    "श्री",
    "श्रीमती",
    "डॉ",
    "डॉ.",
}


def is_joining_char(char: str) -> bool:
    """
    True for characters that belong to the same written word as the
    previous character: letters/digits, Devanagari vowel signs, virama,
    anusvara, nukta (Unicode category M*) and ZWJ/ZWNJ.
    """
    return (
        char.isalnum()
        or unicodedata.category(char).startswith("M")
        or char in "‌‍"
    )


def find_alias_entities(text: str) -> List[Dict[str, Any]]:
    """
    Find only the explicitly configured aliases.
    Offsets always refer to the original sentence.

    Word boundaries:
    - An alias must not start in the middle of a word
      (e.g. "प्रिया" inside "सुप्रिया").
    - Latin aliases must also end at a word boundary.
    - Devanagari aliases may be followed by an attached Hindi/Marathi
      suffix ("जिनेव्हात", "इन्फोसिसमध्ये"), but not by a vowel sign /
      virama, which would mean the alias ends mid-syllable
      (e.g. "प्रिया" inside "प्रियांका").
    """
    found = []

    for entity_type, aliases in ENTITY_ALIASES.items():
        for alias in aliases:
            if not alias:
                continue

            start_pos = 0

            while True:
                start = text.find(alias, start_pos)

                if start == -1:
                    break

                end = start + len(alias)

                before_ok = (
                    start == 0
                    or not is_joining_char(text[start - 1])
                )

                if end == len(text):
                    after_ok = True
                elif alias.isascii():
                    after_ok = not is_joining_char(text[end])
                else:
                    after_ok = not (
                        unicodedata.category(text[end]).startswith("M")
                        or text[end] in "‌‍"
                    )

                if before_ok and after_ok:
                    found.append({
                        "Entity": text[start:end],
                        "Entity Type": entity_type,
                        "Confidence": 1.0,
                        "Start": start,
                        "End": end,
                        "Source": "Configured alias — review",
                    })

                start_pos = end

    return found


def remove_title_only_entities(entities: List[Dict[str, Any]]):
    """
    Remove standalone Dr./डॉ. predictions.
    If the model predicts the title together with the name,
    retain the complete span.
    """
    cleaned = []

    for item in entities:
        value = item["Entity"].strip().casefold()

        if value in PERSON_TITLES:
            continue

        cleaned.append(item)

    return cleaned


def merge_organization_fragments(
    text: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge adjacent ORG fragments when they are separated only by
    whitespace. This helps with outputs such as:
    'टाटा कंसल्टेंसी' + 'सर्विसेज'.

    It deliberately does not merge separate PER or LOC predictions.
    """
    ordered = sorted(
        entities,
        key=lambda item: (item["Start"], item["End"])
    )

    merged = []

    for item in ordered:
        current = item.copy()

        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]

        gap = text[previous["End"]:current["Start"]]

        can_merge_org = (
            previous["Entity Type"] == "ORG"
            and current["Entity Type"] == "ORG"
            and current["Start"] >= previous["End"]
            and gap.strip() == ""
            and len(gap) <= 3
        )

        # Merge touching fragments of the same entity type.
        # This handles token splits such as "रो" + "हन".
        # Touching pieces INSIDE one word are merged even when the model
        # labelled them differently (e.g. "टा"/ORG + "टा"/PER inside
        # "टाटा"); the word gets the label covering most of it.
        touching = current["Start"] == previous["End"]
        inside_one_word = (
            touching
            and is_joining_char(text[previous["End"] - 1])
            and is_joining_char(text[current["Start"]])
        )
        can_merge_touching = touching and (
            previous["Entity Type"] == current["Entity Type"]
            or inside_one_word
        )

        if can_merge_org or can_merge_touching:
            total_len = current["End"] - previous["Start"]

            if can_merge_touching:
                # Sub-word pieces of one word ("Pri" + "ya"): confidence
                # is the length-weighted score of the winning label, so
                # one short low-score piece does not dominate, and pieces
                # that carried a different label count against it.
                weights = label_weights(previous)
                weights[current["Entity Type"]] = (
                    weights.get(current["Entity Type"], 0.0)
                    + current["Confidence"]
                    * (current["End"] - current["Start"])
                )
                label = max(weights, key=weights.get)
                confidence = weights[label] / total_len
            else:
                # Separate words of an organisation name: conservative.
                label = "ORG"
                confidence = min(
                    previous["Confidence"],
                    current["Confidence"]
                )
                weights = {label: confidence * total_len}

            previous["End"] = current["End"]
            previous["_weights"] = weights
            previous["Entity Type"] = label
            previous["Entity"] = text[
                previous["Start"]:previous["End"]
            ]
            previous["Confidence"] = round(confidence, 4)

            if previous["Source"] != current["Source"]:
                previous["Source"] = "Merged model predictions"

        else:
            merged.append(current)

    for item in merged:
        item.pop("_weights", None)

    return merged


def label_weights(item: Dict[str, Any]) -> Dict[str, float]:
    """Per-label sum of (score x characters) for a (merged) prediction."""
    if "_weights" in item:
        return dict(item["_weights"])

    return {
        item["Entity Type"]:
            item["Confidence"] * (item["End"] - item["Start"])
    }


VIRAMA = "्"


def snap_to_syllables(
    text: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    The tokenizer can cut a Devanagari word inside a syllable, e.g.
    "पुण्यात" -> "पुण्य" + "ात", leaving the entity "पुण्य" without its
    vowel sign. Extend such spans to complete syllables ("पुण्या").

    Only vowel signs / virama / anusvara / nukta (and a consonant joined
    by a virama) are added, never a following letter, so attached Marathi
    suffixes such as -त, -ला, -मध्ये, -च्या are NOT absorbed.
    """
    snapped = []

    for item in entities:
        start = item["Start"]
        end = item["End"]

        while start > 0 and (
            not is_word_start(text[start])
            or (text[start - 1] == VIRAMA and text[start].isalpha())
        ):
            start -= 1

        while end < len(text) and (
            not is_word_start(text[end])
            or (text[end - 1] == VIRAMA and text[end].isalpha())
        ):
            end += 1

        if (start, end) != (item["Start"], item["End"]):
            item = item.copy()
            item["Start"] = start
            item["End"] = end
            item["Entity"] = text[start:end]

        snapped.append(item)

    return snapped


def is_word_start(char: str) -> bool:
    """False for combining marks (vowel signs, virama, ...) and ZWJ/ZWNJ,
    which can never begin a syllable."""
    return not (
        unicodedata.category(char).startswith("M")
        or char in "‌‍"
    )


def remove_partial_word_entities(
    text: str,
    entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Drop model spans that cover only part of a word.

    The "simple" aggregation returns sub-word pieces. Pieces of one word
    that carry entity labels are joined by merge_organization_fragments,
    so a span that still starts or ends inside a word means the model
    tagged the rest of that word O, e.g. "Sentence Pairs" -> B-PER on
    "▁Sent" and I-PER on "▁Pai" with "ence"/"rs" tagged O. Such a span is
    not an entity of the text, whatever its confidence.

    - Start inside a word (any script): rejected.
    - End inside a word: rejected for Latin script. Devanagari spans may
      end inside a word because Hindi/Marathi case suffixes are attached
      ("पुण्यात", "इन्फोसिसमध्ये"); snap_to_syllables has already completed
      the syllable.
    Whole words are never removed, however short ("Li", "Bo").
    """
    kept = []

    for item in entities:
        start, end = item["Start"], item["End"]

        starts_inside = (
            start > 0
            and is_joining_char(text[start - 1])
            and is_joining_char(text[start])
        )

        ends_inside = (
            end < len(text)
            and is_joining_char(text[end - 1])
            and is_joining_char(text[end])
            and not contains_devanagari(text[end - 1:end + 1])
        )

        if not (starts_inside or ends_inside):
            kept.append(item)

    return kept


def confirm_with_aliases(
    candidates: List[Dict[str, Any]],
    alias_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Model spans below the confidence threshold that contain a configured
    alias of the same type, e.g. a weak model span "प्रिया मेहता" (PER)
    around the alias "प्रिया". The alias confirms the type and the model
    supplies the full boundary, so the span is kept with the model's own
    confidence. Without this, only the shorter alias survived and the
    name was truncated. No surname is ever added without a model span.
    """
    confirmed = []

    for item in candidates:
        if any(
            alias["Entity Type"] == item["Entity Type"]
            and item["Start"] <= alias["Start"]
            and alias["End"] <= item["End"]
            for alias in alias_entities
        ):
            item = item.copy()
            item["Source"] += " + configured alias"
            confirmed.append(item)

    return confirmed


def remove_overlapping_entities(
    entities: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Keep the longer / higher-confidence prediction when spans overlap.

    Candidates are considered longest first (then most confident),
    so a short fragment that merely starts earlier cannot block a
    longer, better span.
    """
    ordered = sorted(
        entities,
        key=lambda item: (
            -(item["End"] - item["Start"]),
            -item["Confidence"],
            item["Start"],
        ),
    )

    kept = []

    for item in ordered:
        overlaps = any(
            not (
                item["End"] <= previous["Start"]
                or item["Start"] >= previous["End"]
            )
            for previous in kept
        )

        if not overlaps:
            kept.append(item)

    return sorted(kept, key=lambda item: item["Start"])


def combine_alias_predictions(
    model_entities: List[Dict[str, Any]],
    alias_entities: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    When a configured alias covers exactly the same span and type as a
    model prediction, keep ONE entity: the model's real confidence,
    with the source marked as confirmed by the alias list.
    Without this, the alias (confidence 1.0) silently replaced the
    model prediction and its real score was lost.

    Alias-only matches are kept as review candidates (confidence 1.0).
    """
    combined = [item.copy() for item in model_entities]

    by_span = {
        (item["Start"], item["End"], item["Entity Type"]): item
        for item in combined
    }

    for alias in alias_entities:
        key = (alias["Start"], alias["End"], alias["Entity Type"])

        if key in by_span:
            match = by_span[key]

            if "configured alias" not in match["Source"]:
                match["Source"] += " + configured alias"

        else:
            combined.append(alias)
            by_span[key] = alias

    return combined


# ============================================================
# NER EXTRACTION
# ============================================================

def convert_prediction(
    text: str,
    prediction: Dict[str, Any],
    source: str,
    threshold: float,
):
    label = normalize_label(
        prediction.get(
            "entity_group",
            prediction.get("entity", "")
        )
    )

    start = prediction.get("start")
    end = prediction.get("end")
    score = float(prediction.get("score", 0.0))

    if label not in VALID_LABELS:
        return None

    if start is None or end is None:
        return None

    if score < threshold:
        return None

    start = int(start)
    end = int(end)

    if start < 0 or end > len(text) or start >= end:
        return None

    # Shrink the span to exclude surrounding whitespace so that
    # text[Start:End] is always exactly the displayed Entity.
    while start < end and text[start].isspace():
        start += 1

    while end > start and text[end - 1].isspace():
        end -= 1

    entity_text = text[start:end]

    if not entity_text:
        return None

    return {
        "Entity": entity_text,
        "Entity Type": label,
        "Confidence": round(score, 4),
        "Start": start,
        "End": end,
        "Source": source,
    }


# Messages already shown during this script run. Streamlit re-executes the
# script on every interaction, so this resets per run and stops the same
# notice from repeating once per sentence.
_SHOWN_NOTICES = set()


def notify_once(message: str):
    if message not in _SHOWN_NOTICES:
        _SHOWN_NOTICES.add(message)
        st.info(message)


def trained_predictions(text: str, language: str):
    """
    Predictions of the project's trained model for `language`, converted
    like baseline predictions. Returns None when the model is unavailable
    (the caller then falls back to the baseline model).
    """
    predictor, problem = load_trained_model(language)

    if predictor is None:
        LOGGER.warning("%s unavailable: %s", model_label(language), problem)
        notify_once(
            f"The {language} model could not be loaded, so the baseline "
            "multilingual model was used instead."
        )
        return None

    predictions = []

    for prediction in predictor.predict(text):
        converted = convert_prediction(
            text,
            prediction,
            model_label(language),
            0.0,
        )

        if converted:
            predictions.append(converted)

    return predictions


def baseline_predictions(text: str, model_name: str):
    """Baseline multilingual model (Davlan/xlm-roberta-base-ner-hrl)."""
    predictions = []

    for prediction in load_model(model_name)(text) or []:
        converted = convert_prediction(
            text,
            prediction,
            BASELINE_SOURCE,
            0.0,
        )

        if converted:
            predictions.append(converted)

    return predictions


def extract_ner(
    text: str,
    threshold: float,
    model_name: str,
    language: str = "Auto Detect",
) -> List[Dict[str, Any]]:
    """
    Extract PER, LOC and ORG entities.

    Model routing by the selected language:
        English -> English BiLSTM-CRF (trained from scratch)
        Hindi   -> Hindi BiLSTM-CRF (trained from scratch)
        Marathi -> Marathi BiLSTM-CRF (trained from scratch)
        Auto Detect / anything else -> baseline
                   Davlan/xlm-roberta-base-ner-hrl

    Targeted aliases:
        Used only for configured example spellings, after the model.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    # The confidence threshold is applied AFTER fragment merging, so a
    # low-scoring sub-word piece cannot truncate an entity
    # (e.g. "Pri" + "ya" becoming just "Pri").

    predictions = None

    if language in TRAINED_MODELS:
        predictions = trained_predictions(text, language)

    if predictions is None:
        predictions = baseline_predictions(text, model_name)

    # Remove exact duplicates.
    unique = {}

    for item in predictions:
        key = (
            item["Start"],
            item["End"],
            item["Entity Type"],
        )

        if key not in unique or (
            item["Confidence"] > unique[key]["Confidence"]
        ):
            unique[key] = item

    predictions = list(unique.values())

    # Merge fragments before adding aliases.
    predictions = merge_organization_fragments(text, predictions)

    # Complete Devanagari syllables cut by the tokenizer ("पुण्य" -> "पुण्या").
    predictions = snap_to_syllables(text, predictions)

    # Sub-word pieces whose word the model otherwise tagged O ("Sent").
    predictions = remove_partial_word_entities(text, predictions)

    below_threshold = [
        item for item in predictions
        if item["Confidence"] < threshold
    ]
    predictions = [
        item for item in predictions
        if item["Confidence"] >= threshold
    ]

    # Add configured aliases as review candidates. An alias that matches
    # a model entity exactly is merged into it instead of duplicated; a
    # weaker model span that contains the alias keeps its full boundary.
    alias_predictions = find_alias_entities(text)

    predictions = combine_alias_predictions(
        predictions + confirm_with_aliases(below_threshold, alias_predictions),
        alias_predictions,
    )

    # Remove standalone honorifics such as "Dr.".
    predictions = remove_title_only_entities(predictions)

    # Resolve overlaps and return sorted output.
    predictions = remove_overlapping_entities(predictions)

    return predictions


# ============================================================
# DOCUMENT INPUT
# ============================================================

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str) or not text.strip():
        return []

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    parts = re.split(
        r"(?<=[.!?।॥])\s+|\n+",
        text,
    )

    # Skip fragments with no letters/digits (e.g. "...", "।।", "| |").
    parts = [
        re.sub(r"\s+", " ", part).strip()
        for part in parts
        if any(char.isalnum() for char in part)
    ]

    # Re-join splits made after a title or an initial, e.g.
    # "Dr." + "Priya Mehta ...", "डॉ." + "प्रिया ...", "A." + "P. J. Kalam".
    sentences = []

    for part in parts:
        if sentences and ends_with_abbreviation(sentences[-1]):
            sentences[-1] += " " + part
        else:
            sentences.append(part)

    return sentences


def ends_with_abbreviation(sentence: str) -> bool:
    last_word = sentence.rsplit(" ", 1)[-1]

    if last_word.casefold() in PERSON_TITLES and last_word.endswith("."):
        return True

    # Single capital-letter initial such as "A."
    return bool(re.fullmatch(r"[A-Z]\.", last_word))


def decode_text_bytes(raw: bytes) -> str:
    """
    Decode uploaded text without losing Devanagari:
    UTF-16 (Windows Notepad "Unicode") is detected by its BOM,
    otherwise UTF-8 (with or without BOM) is used.
    """
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    else:
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            # Legacy 8-bit file. Devanagari cannot be stored in latin-1,
            # so this is only for English text.
            text = raw.decode("latin-1")

    # Drop control characters (NUL etc.) from damaged files; keep
    # newlines/tabs. ZWJ/ZWNJ used in Devanagari are not affected.
    return "".join(
        char for char in text
        if char in "\n\r\t" or unicodedata.category(char) != "Cc"
    )


def pdf_devanagari_damaged(raw: bytes) -> bool:
    """
    PDF text extraction often breaks Devanagari: ligatures become NUL
    characters and the short-i sign "ि" is stored before its consonant.
    A vowel sign at the start of a word is impossible in real text, so
    either signal means the extracted Hindi/Marathi is unreliable.
    """
    if PdfReader is None:
        return False

    try:
        text = "\n".join(
            page.extract_text() or ""
            for page in PdfReader(io.BytesIO(raw)).pages
        )
    except Exception:
        return False

    if not contains_devanagari(text):
        return False

    return "\x00" in text or bool(
        re.search(r"(^|\s)[ा-्]", text)
    )


# Column names recognised as "the text column" in CSV and Excel uploads.
CSV_TEXT_COLUMN_NAMES = {"text", "sentence", "sentences", "content", "document"}


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(decode_text_bytes(raw)))
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except pd.errors.ParserError as exc:
        LOGGER.warning("CSV parse error: %s", exc)
        raise ValueError(
            "The CSV file could not be read because its rows have "
            "different numbers of columns."
        ) from exc


def read_file(uploaded_file, csv_column=None) -> str:
    filename = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    if filename.endswith(".txt"):
        return decode_text_bytes(raw)

    if filename.endswith(".docx"):
        if Document is None:
            LOGGER.error("python-docx is not installed")
            raise ImportError(
                "DOCX files are not supported on this installation."
            )

        try:
            doc = Document(io.BytesIO(raw))
        except Exception as exc:
            LOGGER.warning("DOCX open error: %s", exc)
            raise ValueError(
                "The DOCX file could not be opened. It may be empty, "
                "damaged, or saved in the older .doc format."
            ) from exc

        parts = [
            paragraph.text.strip()
            for paragraph in doc.paragraphs
            if paragraph.text.strip()
        ]

        for table in doc.tables:
            for row in table.rows:
                cells = [
                    cell.text.strip()
                    for cell in row.cells
                ]

                if any(cells):
                    parts.append(" | ".join(cells))

        return "\n".join(parts)

    if filename.endswith(".pdf"):
        if PdfReader is None:
            LOGGER.error("pypdf is not installed")
            raise ImportError(
                "PDF files are not supported on this installation."
            )

        try:
            reader = PdfReader(io.BytesIO(raw))

            pages = [
                page.extract_text() or ""
                for page in reader.pages
            ]
        except Exception as exc:
            LOGGER.warning("PDF read error: %s", exc)
            raise ValueError(
                "The PDF file could not be read. It may be empty, damaged, "
                "or password-protected."
            ) from exc

        # NUL characters appear where a font ligature has no Unicode
        # mapping; they are never valid text.
        extracted = "\n".join(pages).replace("\x00", "").strip()

        if not extracted:
            raise ValueError(
                "No readable text was found in the PDF. Scanned or "
                "image-only PDFs need OCR, which is not supported."
            )

        return extracted

    if filename.endswith(".csv"):
        df = read_csv_bytes(raw)

        if df.empty:
            return ""

        if csv_column and csv_column in df.columns:
            selected_column = csv_column
        else:
            matching_columns = [
                column
                for column in df.columns
                if str(column).strip().casefold()
                in CSV_TEXT_COLUMN_NAMES
            ]

            if not matching_columns:
                raise ValueError(
                    "The text column could not be detected automatically. "
                    "Select the column that contains the sentences."
                )

            selected_column = matching_columns[0]

        return "\n".join(
            df[selected_column].dropna().astype(str).tolist()
        )

    if filename.endswith(EXCEL_TYPES):
        cells, _ = excel_cells(read_excel_bytes(raw, filename))
        return "\n".join(text for text, _ in cells)

    raise ValueError(
        "Unsupported file type. Upload a TXT, DOCX, PDF, CSV or Excel "
        "(XLSX/XLS) file."
    )


# ============================================================
# EXCEL INPUT
# ============================================================

EXCEL_TYPES = (".xlsx", ".xls")
ALL_SHEETS = "All sheets"
ALL_COLUMNS = "All text columns"


def column_letter(index: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (Excel column names)."""
    letters, index = "", index + 1
    while index:
        index, remainder = divmod(index - 1, 26)
        letters = chr(65 + remainder) + letters
    return letters


def read_excel_bytes(raw: bytes, filename: str) -> Dict[str, pd.DataFrame]:
    """
    Every sheet of an .xlsx (openpyxl) or .xls (xlrd) workbook as a
    DataFrame of strings. As for CSV files, the first row holds the column
    headers; blank cells become "". Unnamed header cells are shown by their
    Excel column letter.
    """
    engine = "xlrd" if filename.lower().endswith(".xls") else "openpyxl"

    try:
        sheets = pd.read_excel(
            io.BytesIO(raw),
            sheet_name=None,
            engine=engine,
            dtype=str,
            keep_default_na=False,
        )
    except ImportError as exc:
        LOGGER.error("Excel engine %s is not installed: %s", engine, exc)
        raise ImportError(
            "Excel files are not supported on this installation."
        ) from exc
    except Exception as exc:
        LOGGER.warning("Excel read error (%s): %s", filename, exc)
        raise ValueError(
            "The Excel file could not be read. It may be damaged, "
            "password-protected, or not a real Excel workbook."
        ) from exc

    for name, df in sheets.items():
        df.columns = [
            f"Column {column_letter(i)}"
            if not str(column).strip() or str(column).startswith("Unnamed:")
            else str(column).strip()
            for i, column in enumerate(df.columns)
        ]

    return sheets


def clear_prepared_input():
    """Forget prepared sentences and results of sections 01-03 (failed or
    changed input). Entity Alignment has its own input and results."""
    for key in ("sentences", "sentence_origins", "ner_results",
                "generated_sentences"):
        st.session_state.pop(key, None)


def input_source(uploaded) -> str:
    """
    The single input-source rule: an uploaded document always wins, and
    typed text is then ignored completely (never merged, never a
    fallback). Without a document, the typed text is the source.
    """
    return "document" if uploaded is not None else "manual"


def spreadsheet_pair(sheets, sheet, english_column, indic_column,
                     row_number, languages):
    """
    One spreadsheet row as alignment input: ([(language, sentence)] in the
    order of `languages` (the selected pair), None), or (None, message)
    when a cell is empty. `row_number` is the spreadsheet row (row 1 =
    headers).
    """
    row = sheets[sheet].iloc[row_number - 2]

    def cell(column):
        value = row[column]
        return "" if pd.isna(value) else str(value).strip()

    indic_language = next(lang for lang in languages if lang != "English")
    texts = {"English": cell(english_column), indic_language: cell(indic_column)}

    if not all(texts.values()):
        return None, (
            f"Row {row_number} has an empty English or {indic_language} cell. "
            "Choose another row."
        )

    return [(language, texts[language]) for language in languages], None


ALIGNMENT_DOCUMENT_TYPES = ["xlsx", "xls", "csv", "docx", "pdf", "txt"]


def load_alignment_document(uploaded_file):
    """
    Content of a document uploaded in Entity Alignment, read with the
    same readers as 01 · Input:
      ("table", {sheet: DataFrame})  for Excel (sheets with data) and CSV
      ("sentences", [sentence, ...]) for DOCX, PDF and TXT
    Raises ValueError / ImportError with a user-readable message.
    """
    name = uploaded_file.name
    raw = uploaded_file.getvalue()

    if name.lower().endswith(EXCEL_TYPES):
        sheets = read_excel_bytes(raw, name)
        return "table", {n: df for n, df in sheets.items() if not df.empty}

    if name.lower().endswith(".csv"):
        df = read_csv_bytes(raw)
        return "table", ({name: df} if not df.empty else {})

    return "sentences", split_sentences(read_file(uploaded_file))


def written_in(language: str, text: str) -> bool:
    """True when `text` is in the script of `language` (Latin / Devanagari)."""
    script = detect_script(text)
    return script.startswith("English" if language == "English" else "Hindi/Marathi")


def default_sentence_index(sentences, language, exclude=None) -> int:
    """First sentence written in `language`'s script (not `exclude`), else 0."""
    return next(
        (i for i, s in enumerate(sentences) if i != exclude and written_in(language, s)),
        0,
    )


def sentence_pair(sentences, first_index, second_index, languages):
    """
    Two sentences chosen from a DOCX/PDF/TXT document as alignment input:
    ([(language, sentence)], None) or (None, message). The user picks the
    pair; nothing is assumed to be aligned sentence by sentence.
    """
    if first_index == second_index:
        return None, "Choose two different sentences."

    chosen = [(languages[0], sentences[first_index]),
              (languages[1], sentences[second_index])]

    for language, text in chosen:
        if not written_in(language, text):
            script = "Latin" if language == "English" else "Devanagari"
            return None, (
                f"The sentence chosen as {language} is not written in {script} "
                "script. Choose a matching sentence."
            )

    return chosen, None


def default_text_column(columns) -> str:
    """A column named like the CSV text columns, else all text columns."""
    for column in columns:
        if str(column).strip().casefold() in CSV_TEXT_COLUMN_NAMES:
            return column
    return ALL_COLUMNS


def excel_cells(
    sheets: Dict[str, pd.DataFrame],
    sheet: str = ALL_SHEETS,
    column: str = ALL_COLUMNS,
):
    """
    ([(cell text, location)], stats) for the chosen sheet(s) and column(s),
    row by row. Blank cells and cells without letters (IDs, numbers) are
    skipped. Location is "Sheet!B3 (header)", with Excel row numbers.
    """
    chosen = sheets if sheet == ALL_SHEETS else {sheet: sheets[sheet]}
    cells, used_sheets, used_rows = [], 0, 0

    for name, df in chosen.items():
        columns = [
            (i, c) for i, c in enumerate(df.columns)
            if column == ALL_COLUMNS or c == column
        ]
        sheet_rows = 0

        for row_number, (_, row) in enumerate(df.iterrows(), start=2):
            found = False

            for i, header in columns:
                text = str(row.iloc[i]).strip()

                if text and any(char.isalpha() for char in text):
                    cells.append((
                        text,
                        f"{name}!{column_letter(i)}{row_number} ({header})",
                    ))
                    found = True

            sheet_rows += found

        used_sheets += sheet_rows > 0
        used_rows += sheet_rows

    return cells, {"sheets": used_sheets, "rows": used_rows}


# ============================================================
# DATAFRAME / EXPORT HELPERS
# ============================================================

def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def run_for_sentences(
    sentences: List[str],
    threshold: float,
    model_name: str,
    language: str = "Auto Detect",
) -> List[Dict[str, Any]]:
    rows = []

    for sentence_no, sentence in enumerate(sentences, start=1):
        entities = extract_ner(
            sentence,
            threshold,
            model_name,
            language,
        )

        for entity in entities:
            rows.append({
                "Sentence No.": sentence_no,
                "Sentence": sentence,
                **entity,
            })

    return rows


def span_text(entity) -> str:
    return "" if entity is None else f"{entity['Start']}–{entity['End']}"


def plural(count: int, word: str, many: str = "") -> str:
    """'1 sentence', '2 sentences'; `many` for irregular plurals."""
    return f"{count} {word if count == 1 else (many or word + 's')}"


# Subtle status colours in result tables (text colour only).
STATUS_COLOURS = {
    "Matched": "color: #5eead4; font-weight: 600",
    "Needs Review": "color: #fbbf24; font-weight: 600",
}


def status_style(value: str) -> str:
    if value in STATUS_COLOURS:
        return STATUS_COLOURS[value]
    return "color: #8ea3bd" if str(value).startswith("Unmatched") else ""


ENTITY_COLUMNS = {
    "Confidence": st.column_config.ProgressColumn(
        "Confidence",
        help="The model's own score for the entity (0–1). Not a verified accuracy.",
        min_value=0.0,
        max_value=1.0,
        format="%.2f",
    ),
    "Start": st.column_config.NumberColumn(
        "Start", help="Character offset where the entity starts in the sentence."
    ),
    "End": st.column_config.NumberColumn(
        "End", help="Character offset where the entity ends in the sentence."
    ),
}


def show_entity_table(df: pd.DataFrame):
    st.dataframe(
        df,
        width="stretch",
        hide_index=True,
        column_config=ENTITY_COLUMNS,
    )


def english_first(sides):
    """
    The alignment module compares English entities with Hindi/Marathi
    entities, so the English side goes first whatever the input order.
    Each side keeps its own sentence, entities and offsets.
    """
    return sorted(sides, key=lambda side: side[0] != "English")


def render_alignment(bilingual):
    """
    Alignment results: English entities ↔ Hindi/Marathi entities, computed
    by alignment/entity_pairs.py (name similarity + entity type, thresholds
    calibrated on the Wikidata-verified gold sets).
    """
    (english, _, english_entities), (other, _, other_entities) = bilingual

    st.subheader("Alignment Results")

    if ALIGNMENT_IMPORT_ERROR is not None:
        LOGGER.error("Alignment import failed: %s", ALIGNMENT_IMPORT_ERROR)
        st.warning(
            "Entity alignment is currently unavailable. The extracted "
            "entities above are still valid."
        )
        return

    calibration = load_calibration()

    if calibration is None:
        LOGGER.error(
            "Alignment calibration missing; run alignment/calibrate_pairs.py"
        )
        st.warning(
            "Entity alignment is currently unavailable because its matching "
            "thresholds are not configured. The extracted entities above are "
            "still valid."
        )
        return

    rows = align_entities(
        english_entities,
        other_entities,
        calibration["match_threshold"],
        calibration["review_threshold"],
        english,
        other,
    )

    table = pd.DataFrame([
        {
            f"{english} Entity": "" if r["source"] is None else r["source"]["Entity"],
            f"{other} Entity": "" if r["target"] is None else r["target"]["Entity"],
            "Type": " / ".join(dict.fromkeys(
                e["Entity Type"] for e in (r["source"], r["target"]) if e is not None
            )),
            "Alignment Score": r["score"],
            "Status": r["status"],
            "Evidence": r["evidence"],
            f"{english} Offsets": span_text(r["source"]),
            f"{other} Offsets": span_text(r["target"]),
        }
        for r in rows
    ])

    statuses = [r["status"] for r in rows]
    matched = statuses.count(MATCHED)
    review = statuses.count(REVIEW)
    unmatched_english = statuses.count(f"Unmatched {english} Entity")
    unmatched_other = statuses.count(f"Unmatched {other} Entity")

    columns = st.columns(5)
    columns[0].metric(f"{english} Entities", len(english_entities))
    columns[1].metric(f"{other} Entities", len(other_entities))
    columns[2].metric("Matched Pairs", matched)
    columns[3].metric("Needs Review", review)
    columns[4].metric(
        f"Unmatched ({english} / {other})",
        f"{unmatched_english} / {unmatched_other}",
    )

    if not english_entities or not other_entities:
        st.info(
            "No entities could be paired because one of the sentences has no "
            "detected entities. All detected entities are listed as unmatched."
        )
    elif review or unmatched_english or unmatched_other:
        st.warning(
            f"Alignment complete: {matched} matched, {review} need review, "
            f"{unmatched_english + unmatched_other} unmatched. Check the pairs "
            "marked Needs Review before using them."
        )
    else:
        st.success(
            f"Alignment complete: all {plural(matched, 'entity pair')} matched. "
            "Matches are automatic and have not been verified by a reviewer."
        )

    if rows:
        st.dataframe(
            table.style.map(status_style, subset=["Status"]),
            width="stretch",
            hide_index=True,
            column_config={
                "Alignment Score": st.column_config.NumberColumn(
                    "Alignment Score",
                    help="Name similarity across scripts (0–1).",
                    format="%.2f",
                ),
                "Evidence": st.column_config.TextColumn("Evidence", width="large"),
            },
        )

        st.download_button(
            "Download Alignment Results (CSV)",
            dataframe_csv(table),
            "entity_alignment.csv",
            "text/csv",
            key="download_alignment",
            icon=":material/download:",
        )

    test = calibration["test"]

    with st.expander("How entity alignment works"):
        st.markdown(
            f"- Each English entity is compared with each {other} entity by "
            "**name similarity**: the Devanagari name is romanised and "
            "compared with the English spelling (score 0–1). Case suffixes "
            "such as मध्ये are ignored.\n"
            "- Each entity is paired at most once, best score first.\n"
            f"- **Matched**: same entity type, score ≥ "
            f"{calibration['match_threshold']:.2f}, and no competing candidate.\n"
            f"- **Needs Review**: score ≥ {calibration['review_threshold']:.2f} "
            "but below the match threshold, different entity types, or a "
            "competing candidate.\n"
            "- **Unmatched**: no candidate with enough evidence. Names that "
            "are translated rather than transliterated (for example, "
            "*India* and *भारत*) cannot be matched by this method.\n"
            "- The thresholds were calibrated on Wikidata-verified "
            "English–Hindi and English–Marathi name pairs. On held-out test "
            f"pairs, Matched results were correct "
            f"{test['hi']['match']['precision']:.1%} (Hindi) and "
            f"{test['mr']['match']['precision']:.1%} (Marathi) of the time, "
            f"and found {test['hi']['match']['recall']:.1%} and "
            f"{test['mr']['match']['recall']:.1%} of the true pairs."
        )


ENTITY_TYPE_NAMES = {"PER": "person", "ORG": "organization", "LOC": "location"}


def highlighted(sentence: str, start: int, end: int) -> str:
    """Markdown for `sentence` with characters start..end highlighted."""
    return (
        escape_markdown(sentence[:start])
        + f"**:orange-background[{escape_markdown(sentence[start:end])}]**"
        + escape_markdown(sentence[end:])
    )


def render_sentence_generation(rows):
    """
    Generate New Sentences: replace extracted entities (from the existing
    NER results) with new entities of the same type, one new sentence per
    replacement. Logic in sentence_generation.py.
    """
    st.divider()
    st.header("Generate New Sentences")
    st.caption(
        "Replace an extracted entity with a new entity of the same type. Each "
        "replacement creates one new sentence; the rest of the sentence stays "
        "exactly as it is. Leave a field empty to skip that entity."
    )

    by_sentence = {}
    for row in rows:
        by_sentence.setdefault(row["Sentence No."], []).append(row)

    def field_key(row):
        return f"gen_{entity_key(row)}_{row['Entity']}"

    for position, (number, sentence_rows) in enumerate(sorted(by_sentence.items())):
        sentence = sentence_rows[0]["Sentence"]

        with st.expander(f"Sentence {number}: {escape_markdown(sentence)}",
                         expanded=position == 0):
            for entity_type in ENTITY_TYPES:
                typed = sorted(
                    (r for r in sentence_rows if r["Entity Type"] == entity_type),
                    key=lambda r: r["Start"],
                )
                if not typed:
                    continue

                st.markdown(f"**{entity_type}** · {ENTITY_TYPE_NAMES[entity_type]}")
                for row in typed:
                    st.text_input(
                        f"Replace “{row['Entity']}” with",
                        key=field_key(row),
                        placeholder=f"New {ENTITY_TYPE_NAMES[entity_type]}",
                    )

    if st.button(
        "Generate New Sentences",
        type="primary",
        key="generate_sentences",
        icon=":material/auto_awesome:",
    ):
        replacements = {
            entity_key(row): st.session_state.get(field_key(row), "")
            for row in rows
        }

        try:
            generated = generate_sentences(rows, replacements)
        except ValueError:
            LOGGER.exception("Sentence generation failed")
            generated = None
            st.error(
                "New sentences could not be generated because the extracted "
                "entities no longer match the sentences. Extract the entities "
                "again and retry."
            )

        if generated == []:
            st.warning(
                "Enter at least one replacement that differs from the "
                "original entity."
            )

        st.session_state["generated_sentences"] = generated or None

    generated = st.session_state.get("generated_sentences")

    if not generated:
        return

    st.success(f"Generated {plural(len(generated), 'new sentence')}.")
    current = None

    for item in generated:
        if item["Original Sentence"] != current:
            current = item["Original Sentence"]
            st.markdown(f"**Original:** {escape_markdown(current)}")

        st.markdown(
            f"- `{item['Entity Type']}` {escape_markdown(item['Original Entity'])} "
            f"→ {escape_markdown(item['Replacement'])}: "
            + highlighted(item["Generated Sentence"], item["Start"], item["End"])
        )

    with st.expander("Generated Sentences Table"):
        st.dataframe(
            pd.DataFrame(generated).drop(columns=["Start", "End"]),
            width="stretch",
            hide_index=True,
        )

    word_col, text_col = st.columns(2)
    word_col.download_button(
        "Download Sentences (Word)",
        build_docx(generated),
        "generated_sentences.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key="download_generated_docx",
        icon=":material/download:",
        width="stretch",
    )
    text_col.download_button(
        "Download Sentences (TXT)",
        build_txt(generated).encode("utf-8"),
        "generated_sentences.txt",
        "text/plain",
        key="download_generated_txt",
        icon=":material/download:",
        width="stretch",
    )


# ============================================================
# UI STYLE
# ============================================================

st.markdown(APP_CSS, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌐 Indic Entity Lab")
    st.caption("Named entity extraction and alignment for English, Hindi and Marathi")
    st.page_link(
        "pages/1_Research_Results.py",
        label="Research Results",
        icon="📊",
    )
    st.divider()

    st.markdown("### Workflow")

    st.markdown(
        "1. **Input**: enter text or upload a document\n"
        "2. **Sentence Preview**: check the extracted sentences\n"
        "3. **Entity Extraction**: find PER, LOC and ORG entities\n"
        "4. **Entity Alignment**: pair entities across two languages"
    )

    st.divider()

    st.markdown("### Models")

    model_name = st.text_input(
        "Multilingual Model ID",
        value=MODEL_NAME,
        help=(
            "Hugging Face token-classification model used by the baseline "
            "option. Its labels must name the entity types."
        ),
    )

    st.markdown(
        "**Language-specific:** BiLSTM-CRF models trained for this project "
        "(English, Hindi, Marathi).  \n"
        "**Baseline:** the multilingual model above."
    )

    st.caption(
        "Entity types: PER (person), LOC (location), ORG (organization)."
    )

    if TRANSFORMERS_IMPORT_ERROR is not None:
        LOGGER.error("transformers import failed: %s", TRANSFORMERS_IMPORT_ERROR)
        st.error(
            "The multilingual model is unavailable because a required "
            "package is not installed. Language-specific models can still "
            "be used."
        )

    st.info(
        "Entities are model predictions. Review them before using them as "
        "dataset labels.",
        icon=":material/info:",
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="tag">RESEARCH PROTOTYPE · NLP</div>
        <div class="hero-title">Indic Entity Lab</div>
        <div class="hero-sub">
            Extract person, location and organization names from English,
            Hindi and Marathi text, and align them across languages.
        </div>
        <div class="chips">
            <span class="chip">English · हिन्दी · मराठी</span>
            <span class="chip">PER · LOC · ORG</span>
            <span class="chip">Entity Extraction · Entity Alignment</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# NER EXTRACTION: WORKFLOW NAVIGATION
# ============================================================
# Both workflows live on this page as tabs. The cards switch the active
# tab through its session-state key; both tabs stay rendered, so typed
# text, uploads and results are kept when switching.

WORKFLOW_TABS = ["Monolingual NER Extraction", "Bilingual NER Extraction"]
WORKFLOW_CARDS = [
    (WORKFLOW_TABS[0], ":material/article:", "open_monolingual",
     "Extract PER, LOC and ORG entities from English, Hindi or Marathi "
     "text or documents."),
    (WORKFLOW_TABS[1], ":material/compare_arrows:", "open_bilingual",
     "Extract entities from two corresponding sentences in different "
     "languages and align them."),
]


def open_workflow(name: str):
    """on_click: show the chosen workflow's tab."""
    st.session_state["ner_workflow"] = name


st.header("NER Extraction")

active_workflow = st.session_state.get("ner_workflow", WORKFLOW_TABS[0])

for column, (name, icon, key, description) in zip(st.columns(2), WORKFLOW_CARDS):
    with column, st.container(border=True):
        st.markdown(f"**{name}**")
        st.caption(description)
        st.button(
            f"Open {name}",
            key=key,
            icon=icon,
            type="primary" if name == active_workflow else "secondary",
            on_click=open_workflow,
            args=(name,),
            width="stretch",
        )

monolingual_tab, bilingual_tab = st.tabs(
    WORKFLOW_TABS,
    key="ner_workflow",
    default=WORKFLOW_TABS[0],
    on_change="rerun",
)


with monolingual_tab:
    # ============================================================
    # INPUT
    # ============================================================

    st.header("01 · Input")

    language_col, engine_col = st.columns(2)

    with language_col:
        ner_language = language_selector("ner_language", "Input Language")

    with engine_col:
        # Fixed: the monolingual workflow uses the language-specific models.
        ner_engine = fixed_model(TRAINED_ENGINE)

    tab_text, tab_file = st.tabs([
        "Enter Text",
        "Upload Document",
    ])

    with tab_text:
        manual_text = st.text_area(
            f"Enter one or more {ner_language} sentences",
            placeholder=EXAMPLE_SENTENCES[ner_language],
            height=170,
            key="manual_text",
        )


    with tab_file:
        uploaded = st.file_uploader(
            "Upload a TXT, DOCX, PDF, CSV or Excel file",
            type=["txt", "docx", "pdf", "csv", "xlsx", "xls"],
            help=(
                "Scanned (image-only) PDFs are not supported. In CSV and Excel "
                "files, the first row is read as column headers."
            ),
        )

        csv_column = None
        excel_sheet, excel_column = ALL_SHEETS, ALL_COLUMNS

        if uploaded is not None and uploaded.name.lower().endswith(EXCEL_TYPES):
            try:
                workbook = read_excel_bytes(uploaded.getvalue(), uploaded.name)
                non_empty = {name: df for name, df in workbook.items() if not df.empty}

                if not non_empty:
                    st.info("This workbook has no data rows.")

                else:
                    sheet_col, column_col = st.columns(2)

                    with sheet_col:
                        excel_sheet = st.selectbox(
                            "Sheet",
                            ([ALL_SHEETS] if len(non_empty) > 1 else []) + list(non_empty),
                            key="excel_sheet",
                            help="Process one sheet or every sheet with data.",
                        )

                    sheet_columns = list(dict.fromkeys(
                        column
                        for name, df in non_empty.items()
                        if excel_sheet in (ALL_SHEETS, name)
                        for column in df.columns
                    ))
                    column_options = [ALL_COLUMNS] + sheet_columns
                    # Sheets can have different layouts, so a named text column
                    # is only pre-selected for a single sheet.
                    default_column = (
                        ALL_COLUMNS if excel_sheet == ALL_SHEETS
                        else default_text_column(sheet_columns)
                    )

                    with column_col:
                        excel_column = st.selectbox(
                            "Text Column",
                            column_options,
                            index=column_options.index(default_column),
                            key=f"excel_text_column_{excel_sheet}",
                            help=(
                                "The column that contains the sentences, or every "
                                "column with text."
                            ),
                        )

                    if len(workbook) > len(non_empty):
                        st.caption(
                            f"{plural(len(workbook) - len(non_empty), 'empty sheet')} "
                            "skipped."
                        )

            except (ValueError, ImportError) as exc:
                st.error(f"The Excel file could not be previewed. {exc}")

            except Exception:
                LOGGER.exception("Excel preview failed")
                st.error("The Excel file could not be previewed.")

        if uploaded is not None and uploaded.name.lower().endswith(".csv"):
            try:
                preview_df = read_csv_bytes(uploaded.getvalue())

                if len(preview_df.columns):
                    csv_column = st.selectbox(
                        "Text Column",
                        ["Auto-detect"] + list(preview_df.columns),
                        key="csv_text_column",
                        help="The CSV column that contains the sentences.",
                    )

                    if csv_column == "Auto-detect":
                        csv_column = None

            except ValueError as exc:
                st.error(f"The CSV file could not be previewed. {exc}")

            except Exception:
                LOGGER.exception("CSV preview failed")
                st.error("The CSV file could not be previewed.")


    # ============================================================
    # INPUT SOURCE
    # ============================================================

    active_source = input_source(uploaded)
    source_key = (
        ("document", uploaded.name, uploaded.size)
        if active_source == "document" else ("manual",)
    )

    # A new, changed or removed document makes earlier sentences and results
    # stale: drop them so nothing from the previous source can be run.
    if st.session_state.get("input_source_key") not in (None, source_key):
        clear_prepared_input()
    st.session_state["input_source_key"] = source_key

    if active_source == "document":
        st.caption(f"Input source: Uploaded document ({uploaded.name})")

        if manual_text.strip():
            st.info(
                "A document is uploaded, so the text in the text box is ignored. "
                "Remove the document to use the text box."
            )
    else:
        st.caption("Input source: Manual sentences")


    # ============================================================
    # PROCESS INPUT
    # ============================================================

    if st.button(
        "Prepare Sentences",
        type="primary",
        key="process_input",
        icon=":material/segment:",
        help=(
            "Splits the text or document into sentences. Entities are "
            "extracted in the next step."
        ),
    ):
        try:
            if active_source == "manual":
                raw_text = manual_text
                source_name = "the text box"

            elif uploaded.name.lower().endswith(EXCEL_TYPES):
                # Split cell by cell so every sentence keeps its cell location.
                cells, excel_stats = excel_cells(
                    read_excel_bytes(uploaded.getvalue(), uploaded.name),
                    excel_sheet,
                    excel_column,
                )
                raw_text = None
                excel_sentences, excel_origins = [], []

                for text, location in cells:
                    for sentence in split_sentences(text):
                        excel_sentences.append(sentence)
                        excel_origins.append(location)

                source_name = uploaded.name

            else:
                raw_text = read_file(
                    uploaded,
                    csv_column,
                )
                source_name = uploaded.name

                if (
                    uploaded.name.lower().endswith(".pdf")
                    and pdf_devanagari_damaged(uploaded.getvalue())
                ):
                    st.warning(
                        "Hindi/Marathi text in this PDF was not extracted "
                        "correctly (a common PDF font limitation), so entity "
                        "results may be unreliable. For best results, upload the "
                        "same content as DOCX or TXT, or paste the text."
                    )

            if raw_text is None:
                sentences, origins = excel_sentences, excel_origins
            else:
                sentences, origins = split_sentences(raw_text), None

            if not sentences:
                clear_prepared_input()
                st.warning(
                    "No readable text was found in the uploaded document. Please "
                    "upload a valid document."
                    if active_source == "document" else
                    "No readable text was found. Enter text or upload a document."
                )

            else:
                st.session_state["sentences"] = sentences
                st.session_state["sentence_origins"] = origins
                st.session_state["source_name"] = source_name
                st.session_state["ner_results"] = None

                if origins is None:
                    st.success(
                        f"Extracted {plural(len(sentences), 'sentence')} from {source_name}."
                    )
                else:
                    st.success(
                        f"Extracted {plural(len(sentences), 'sentence')} from "
                        f"{plural(excel_stats['rows'], 'row')} in "
                        f"{plural(excel_stats['sheets'], 'sheet')} of {source_name}."
                    )

        except (ValueError, ImportError) as exc:
            # Raised by read_file with a user-readable explanation.
            clear_prepared_input()
            st.error(f"The input could not be processed. {exc}")

        except Exception:
            clear_prepared_input()
            LOGGER.exception("Input processing failed")
            st.error(
                "The input could not be processed because of an unexpected "
                "error. Details were written to the application log."
            )


    # ============================================================
    # SENTENCE PREVIEW
    # ============================================================

    if "sentences" in st.session_state:
        sentences = st.session_state["sentences"]

        st.divider()
        st.header("02 · Sentence Preview")

        c1, c2, c3 = st.columns(3)

        c1.metric("Sentences", len(sentences))

        c2.metric(
            "Input Language",
            ner_language,
        )

        c3.metric(
            "Source",
            st.session_state.get("source_name", "Input"),
        )

        # Show what the text actually contains so a wrong language choice does
        # not silently mislead the user.
        detected_script = detect_script(" ".join(sentences))
        route_language = ner_route(ner_language, ner_engine)

        st.caption(
            f"Detected script: {detected_script} · "
            f"NER model for this input: {model_label(route_language)}"
        )

        if language_mismatch(ner_language, detected_script):
            st.warning(
                f"Selected language is {ner_language}, but the text appears to be "
                f"in {detected_script}. Extraction will still run, but results "
                "are likely to be poor. Select the language that matches the text."
            )

        with st.expander(
            "Extracted Sentences",
            expanded=True,
        ):
            origins = st.session_state.get("sentence_origins")

            for index, sentence in enumerate(sentences, start=1):
                location = f" · {origins[index - 1]}" if origins else ""
                st.markdown(f"**Sentence {index}**{escape_markdown(location)}")
                st.markdown(escape_markdown(sentence))

        sentence_df = pd.DataFrame({
            "Sentence No.": range(1, len(sentences) + 1),
            **({"Location": origins} if origins else {}),
            "Sentence": sentences,
        })

        st.download_button(
            "Download Sentences (CSV)",
            dataframe_csv(sentence_df),
            "extracted_sentences.csv",
            "text/csv",
            key="download_sentences",
            icon=":material/download:",
        )


        # ========================================================
        # NER EXTRACTION
        # ========================================================

        st.divider()
        st.header("03 · Entity Extraction")

        st.caption(
            "Entities are extracted from each sentence. Start and End are "
            "character offsets in the original sentence."
        )

        threshold = st.slider(
            "Minimum Confidence",
            min_value=0.0,
            max_value=0.95,
            value=0.30,
            step=0.05,
            help=(
                "Lower values return more entities but may include more false "
                "positives."
            ),
            key="ner_threshold",
        )

        if st.button(
            "Extract Entities",
            type="primary",
            key="run_ner",
            icon=":material/manage_search:",
            help="Runs the selected NER model on every sentence.",
        ):
            if not model_name.strip():
                st.error("Enter a multilingual model ID in the sidebar.")

            else:
                try:
                    with st.spinner(
                        f"Extracting entities from {plural(len(sentences), 'sentence')}…"
                    ):
                        language = ner_route(ner_language, ner_engine)

                        ner_rows = run_for_sentences(
                            sentences,
                            threshold,
                            model_name.strip(),
                            language,
                        )

                        # Spreadsheet input: add the source cell of each row.
                        origins = st.session_state.get("sentence_origins")
                        if origins:
                            ner_rows = [
                                {
                                    "Sentence No.": row["Sentence No."],
                                    "Location": origins[row["Sentence No."] - 1],
                                    **row,
                                }
                                for row in ner_rows
                            ]

                        st.session_state["ner_results"] = ner_rows
                        # New entities: earlier generated sentences are stale.
                        st.session_state.pop("generated_sentences", None)
                        st.session_state["used_model"] = (
                            model_label(language)
                            if language in TRAINED_MODELS
                            else model_name.strip()
                        )

                    st.success(
                        "Entity extraction complete: "
                        f"{plural(len(ner_rows), 'entity', 'entities')} found in "
                        f"{plural(len(sentences), 'sentence')}."
                    )

                except ValueError as exc:
                    # e.g. a model ID whose labels cannot be mapped to PER/LOC/ORG
                    st.error(f"Entity extraction failed. {exc}")

                except Exception:
                    LOGGER.exception("Entity extraction failed")
                    st.error(
                        "Entity extraction failed: the model could not be loaded "
                        "or run. Check the model ID and your connection, then try "
                        "again. Details were written to the application log."
                    )


        # ========================================================
        # NER RESULTS
        # ========================================================

        rows = st.session_state.get("ner_results")

        if rows is not None:
            st.subheader("Extracted Entities")

            if rows:
                result_df = pd.DataFrame(rows)

                p, l, o, total = st.columns(4)

                p.metric(
                    "PER",
                    sum(row["Entity Type"] == "PER" for row in rows),
                )

                l.metric(
                    "LOC",
                    sum(row["Entity Type"] == "LOC" for row in rows),
                )

                o.metric(
                    "ORG",
                    sum(row["Entity Type"] == "ORG" for row in rows),
                )

                total.metric(
                    "Total Entities",
                    len(rows),
                )

                show_entity_table(result_df)

                st.download_button(
                    "Download Results (CSV)",
                    dataframe_csv(result_df),
                    "ner_results.csv",
                    "text/csv",
                    key="download_ner",
                    icon=":material/download:",
                )

            else:
                st.warning(
                    "No entities were detected. This does not necessarily mean "
                    "the text contains none: the model may have missed them, or "
                    "the minimum confidence may be too high."
                )

            st.caption(
                "Model: " + st.session_state.get("used_model", model_name)
            )

            with st.expander("How to read the results"):
                st.markdown(
                    "- **PER** = person, **LOC** = location, **ORG** = organization.\n"
                    "- **Confidence** is the model's own score (0–1), not a "
                    "verified accuracy.\n"
                    "- **Start / End** are character offsets in the sentence.\n"
                    "- **Source** names the model that found the entity. "
                    "*+ configured alias* means the result also matches a known "
                    "spelling in the alias list (the model's confidence is kept). "
                    "*Configured alias — review* means it was found only through "
                    "the alias list (shown as 1.0); please check these entries."
                )

            if rows:
                render_sentence_generation(rows)


with bilingual_tab:
    # ============================================================
    # ENTITY ALIGNMENT: INPUT + EXTRACTION
    # ============================================================
    st.header("04 · Entity Alignment")

    st.caption(
        "Choose the language pair, then enter two corresponding sentences or "
        "upload a document that contains them. Named entities are extracted "
        "independently from each sentence and then aligned."
    )

    MANUAL_METHOD = "Enter Sentences Manually"
    DOCUMENT_METHOD = "Upload a Document"

    # Keep typed sentences while their fields are hidden in document mode
    # (Streamlit forgets the state of widgets that are not rendered).
    for kept in ("bilingual_english", "bilingual_second"):
        if kept in st.session_state:
            st.session_state[kept] = st.session_state[kept]

    input_method = st.segmented_control(
        "Input Method",
        [MANUAL_METHOD, DOCUMENT_METHOD],
        default=MANUAL_METHOD,
        required=True,
        key="align_input_method",
        help="Only the selected input is processed.",
    )


    pair_col, engine_col = st.columns(2)

    with pair_col:
        language_pair = st.segmented_control(
            "Language Pair",
            list(LANGUAGE_PAIRS),
            default="English → Hindi",
            required=True,
            key="bilingual_pair",
            help="The languages of the first and second sentence, in that order.",
        )

    with engine_col:
        # Fixed: the bilingual workflow uses the baseline multilingual model.
        bilingual_engine = fixed_model(NER_ENGINES[0])

    first_language, second_language = LANGUAGE_PAIRS[language_pair]
    indic_language = second_language if first_language == "English" else first_language


    align_document = None
    if input_method == DOCUMENT_METHOD:
        align_document = st.file_uploader(
            "Upload a document with corresponding sentences",
            type=ALIGNMENT_DOCUMENT_TYPES,
            key="align_document",
            help=(
                "Excel or CSV: one column per language, one sentence pair per "
                "row (row 1 = headers). Word, PDF or TXT: choose the two "
                "corresponding sentences from the extracted text."
            ),
        )

    # Results belong to one input: a new mode or document makes them stale.
    align_source_key = (
        ("manual",) if input_method == MANUAL_METHOD
        else ("document", align_document.name, align_document.size) if align_document
        else ("document",)
    )
    if st.session_state.get("align_source_key") not in (None, align_source_key):
        st.session_state.pop("bilingual_results", None)
    st.session_state["align_source_key"] = align_source_key

    document_sides, document_problem, document_label = None, None, None
    kind = None  # "table" / "sentences" once a document was read

    if input_method == DOCUMENT_METHOD:
        if align_document is None:
            document_problem = "Upload a document to align its sentences."
            st.info(document_problem)

        else:
            try:
                kind, content = load_alignment_document(align_document)
            except (ValueError, ImportError) as exc:
                kind, content = None, None
                document_problem = f"The document could not be read. {exc}"
                st.error(document_problem)
            except Exception:
                LOGGER.exception("Alignment document could not be read")
                kind, content = None, None
                document_problem = "The document could not be read."
                st.error(document_problem)

            if kind == "table" and not content:
                document_problem = (
                    "No data rows were found in the uploaded document. Please "
                    "upload a valid document."
                )

            elif kind == "table":
                # One sentence pair per row: pick the sheet, the two language
                # columns and the row.
                table_cols = st.columns(4)
                doc_sheet = table_cols[0].selectbox(
                    "Sheet", list(content), key="align_sheet",
                    disabled=len(content) == 1,
                )
                sheet_df = content[doc_sheet]
                columns = list(sheet_df.columns)

                def guess(language, fallback):
                    return next((i for i, c in enumerate(columns)
                                 if language.casefold() in str(c).casefold()), fallback)

                doc_english = table_cols[1].selectbox(
                    "English Column", columns, index=guess("English", 0),
                    key=f"align_english_{doc_sheet}",
                )
                doc_indic = table_cols[2].selectbox(
                    f"{indic_language} Column", columns,
                    index=guess(indic_language, min(1, len(columns) - 1)),
                    key=f"align_indic_{indic_language}_{doc_sheet}",
                )
                doc_row = table_cols[3].number_input(
                    "Row", min_value=2, max_value=len(sheet_df) + 1, value=2,
                    step=1, key=f"align_row_{doc_sheet}",
                    help="Spreadsheet row number (row 1 holds the headers).",
                )

                if doc_english == doc_indic:
                    document_problem = (
                        f"Choose two different columns for English and {indic_language}."
                    )
                else:
                    preview = sheet_df[[doc_english, doc_indic]].copy()
                    preview.insert(0, "Row", range(2, len(sheet_df) + 2))
                    st.dataframe(preview, width="stretch", hide_index=True, height=210)

                    document_sides, document_problem = spreadsheet_pair(
                        content, doc_sheet, doc_english, doc_indic,
                        int(doc_row), (first_language, second_language),
                    )
                    document_label = (
                        f"{align_document.name} · {doc_sheet}, row {int(doc_row)}"
                        if len(content) > 1 else
                        f"{align_document.name} · row {int(doc_row)}"
                    )

            elif kind == "sentences" and len(content) < 2:
                document_problem = (
                    "The document does not contain enough sentences to form a "
                    "sentence pair. Please upload a valid document."
                    if content else
                    "No readable text was found in the uploaded document. Please "
                    "upload a valid document."
                )

            elif kind == "sentences":
                # Arbitrary text is not assumed to be aligned: the user picks
                # the sentence for each language.
                st.dataframe(
                    pd.DataFrame({
                        "No.": range(1, len(content) + 1),
                        "Sentence": content,
                        "Script": [detect_script(s) for s in content],
                    }),
                    width="stretch", hide_index=True, height=210,
                )

                def numbered(i):
                    text = content[i]
                    return f"{i + 1}. {text[:90]}{'…' if len(text) > 90 else ''}"

                pick_cols = st.columns(2)
                first_index = pick_cols[0].selectbox(
                    f"{first_language} Sentence", range(len(content)),
                    index=default_sentence_index(content, first_language),
                    format_func=numbered, key=f"align_first_{first_language}",
                )
                second_index = pick_cols[1].selectbox(
                    f"{second_language} Sentence", range(len(content)),
                    index=default_sentence_index(content, second_language, first_index),
                    format_func=numbered, key=f"align_second_{second_language}",
                )
                document_sides, document_problem = sentence_pair(
                    content, first_index, second_index,
                    (first_language, second_language),
                )
                document_label = (
                    f"{align_document.name} · sentences {first_index + 1} and "
                    f"{second_index + 1}"
                )

            if document_sides:
                preview_cols = st.columns(2)
                for column, (language, text) in zip(preview_cols, document_sides):
                    with column, st.container(border=True):
                        st.markdown(f"**{language} Sentence**")
                        st.markdown(escape_markdown(text))
            elif kind is not None and document_problem:
                # Read errors were shown above; this covers content problems.
                st.warning(document_problem)


    def sentence_box(language: str):
        """Sentence input for `language`. The English box and the Hindi/Marathi
        box keep their own keys, so text stays with its language when the pair
        order changes."""
        return st.text_area(
            f"{language} Sentence",
            placeholder=EXAMPLE_SENTENCES[language],
            key="bilingual_english" if language == "English" else "bilingual_second",
        )


    if input_method == MANUAL_METHOD:
        first_col, second_col = st.columns(2)

        with first_col:
            first_sentence = sentence_box(first_language)

        with second_col:
            second_sentence = sentence_box(second_language)

    bilingual_threshold = st.slider(
        "Minimum Confidence",
        min_value=0.0,
        max_value=0.95,
        value=0.30,
        step=0.05,
        key="bilingual_threshold",
        help=(
            "Lower values return more entities but may include more false "
            "positives."
        ),
    )


    if st.button(
        "Extract and Align Entities",
        type="primary",
        key="run_bilingual",
        icon=":material/compare_arrows:",
    ):
        sides = None

        if input_method == DOCUMENT_METHOD:
            # Document only: never the (hidden) manual fields.
            if document_sides:
                sides = document_sides
                source_label = document_label
            else:
                st.session_state.pop("bilingual_results", None)
                st.warning(document_problem)

        else:
            missing = [
                language
                for language, text in (
                    (first_language, first_sentence),
                    (second_language, second_sentence),
                )
                if not text.strip()
            ]

            if missing:
                st.warning(
                    f"Enter the {missing[0]} sentence."
                    if len(missing) == 1 else
                    f"Enter both the {first_language} and the {second_language} sentence."
                )
            else:
                # Input order; each sentence goes to its own language's model.
                sides = [
                    (first_language, first_sentence),
                    (second_language, second_sentence),
                ]
                source_label = "Manually entered sentences"

        if sides is not None and not model_name.strip():
            st.error("Enter a multilingual model ID in the sidebar.")

        elif sides is not None:
            for language_name, text in sides:
                if language_mismatch(language_name, detect_script(text)):
                    st.warning(
                        f"The {language_name} sentence appears to be written in "
                        f"{detect_script(text)}. Check the inputs and the "
                        "selected language pair."
                    )

            try:
                with st.spinner(
                    "Extracting entities from both sentences…"
                ):
                    st.session_state["bilingual_pair_used"] = language_pair
                    st.session_state["bilingual_source_used"] = source_label
                    st.session_state["bilingual_results"] = [
                        (
                            language_name,
                            text,
                            extract_ner(
                                text,
                                bilingual_threshold,
                                model_name.strip(),
                                ner_route(language_name, bilingual_engine),
                            ),
                        )
                        for language_name, text in sides
                    ]

                st.success(
                    "Entities extracted from both sentences. Alignment results "
                    "are shown below."
                )

            except ValueError as exc:
                st.error(f"Entity extraction failed. {exc}")

            except Exception:
                LOGGER.exception("Bilingual entity extraction failed")
                st.error(
                    "Entity extraction failed: the model could not be loaded or "
                    "run. Details were written to the application log."
                )


    # ============================================================
    # ENTITY ALIGNMENT: RESULTS
    # ============================================================

    if st.session_state.get("bilingual_results"):
        bilingual = st.session_state["bilingual_results"]

        st.caption(
            "Results for "
            + st.session_state.get("bilingual_pair_used", "the entered sentences")
        )
        st.caption(
            "Source: "
            + st.session_state.get("bilingual_source_used", "Manually entered sentences")
        )

        export_rows = []

        for column, (language_name, sentence, entities) in zip(
            st.columns(2),
            bilingual,
        ):
            with column:
                st.subheader(f"{language_name} Entities")

                st.markdown(escape_markdown(sentence))

                if entities:
                    show_entity_table(pd.DataFrame(entities))

                else:
                    st.info(
                        "No entities detected."
                    )

                for entity in entities:
                    export_rows.append({
                        "Language": language_name,
                        "Sentence": sentence,
                        **entity,
                    })

        if export_rows:
            st.download_button(
                "Download Entities (CSV)",
                dataframe_csv(pd.DataFrame(export_rows)),
                "bilingual_entity_results.csv",
                "text/csv",
                key="download_bilingual",
                icon=":material/download:",
            )

        render_alignment(english_first(bilingual))


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Indic Entity Lab · Research prototype · English, Hindi and Marathi "
    "named entity extraction and alignment"
)
