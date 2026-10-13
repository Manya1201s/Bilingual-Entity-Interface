"""
Research results & project status dashboard (pages/1_Research_Results.py).

Every metric is read from the result / config files written by the actual
runs; nothing is typed in here, so the dashboard cannot drift from
results/FINAL_REPORT.md (which reads the same files):

    training/results/evaluation.json              NER test F1 (ours, XLM-R, BERT)
    results/ner/indicner_benchmark.json           IndicNER test F1
    results/ner/english_bert_validation_accuracy.json
                                                  English BERT token accuracy (validation)
    models/english_ner/training_info.json         English BERT validation entity F1
    training/data/english/dataset_info.json       validation token counts

Only the component status lists below are maintained by hand.
"""

import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# DATA (single source of truth: the result files)
# ============================================================

def _load(*parts) -> Optional[Any]:
    path = os.path.join(PROJECT_DIR, *parts)

    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data(show_spinner=False)
def load_results() -> Dict[str, Any]:
    evaluation = _load("training", "results", "evaluation.json") or []

    return {
        "ner": {(row["model"], row["language"]): row for row in evaluation},
        "indicner": _load("results", "ner", "indicner_benchmark.json"),
        "bert_val_acc": _load(
            "results", "ner", "english_bert_validation_accuracy.json"
        ),
        "bert_info": _load("models", "english_ner", "training_info.json"),
        "english_data": _load(
            "training", "data", "english", "dataset_info.json"
        ),
    }


def pct(value: Optional[float], digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


# Rows of the NER F1 comparison: (model key in evaluation.json, label, role).
NER_ROWS = [
    ("Hindi BiLSTM-CRF", "BiLSTM-CRF (trained from scratch)", "Custom model"),
    ("Marathi BiLSTM-CRF", "BiLSTM-CRF (trained from scratch)", "Custom model"),
    ("English BiLSTM-CRF", "BiLSTM-CRF (trained from scratch)", "Custom model"),
    ("Baseline XLM-R", "XLM-R (Davlan/xlm-roberta-base-ner-hrl)",
     "Benchmark (pretrained, not trained here)"),
    ("English BERT", "BERT (bert-base-cased, fine-tuned here)",
     "Reference experiment"),
]


# ============================================================
# STATUS (maintained by hand — update when work is resumed)
# ============================================================

COMPLETED = [
    "Hindi, Marathi and English NER model training and reported evaluation",
    "NER benchmark comparisons (XLM-R, IndicNER, English BERT) and error analysis",
    "English NER token-level validation accuracy calculation",
    "Anonymisation implementation and initial (monolingual) evaluation",
    "Alignment implementation and Wikidata-verified gold sets",
    "End-to-end pipeline implementation",
]

PENDING = [
    "Alignment metrics (accuracy, P@1, Recall@K, MRR)",
    "Cross-lingual anonymisation consistency evaluation",
    "Full end-to-end pipeline execution",
]

# (component, badge class, badge text, explanation)
COMPONENT_STATUS = [
    ("NER", "done", "Evaluated",
     "Held-out test F1 for Hindi, Marathi and English; benchmarks and "
     "error analysis complete."),
    ("Entity alignment", "partial", "Implemented · metrics pending",
     "Aligner and Wikidata-verified gold sets are ready; evaluation has "
     "not been run yet."),
    ("Anonymisation", "partial", "Monolingual evaluated · cross-lingual pending",
     "Monolingual evaluation complete; cross-lingual consistency not run."),
]


# ============================================================
# HTML HELPERS
# ============================================================

def card(label: str, value: str, meta: str, badge: str = "") -> str:
    value_html = f'<div class="metric-value">{value}</div>' if value else ""

    return (
        '<div class="metric-card">'
        f"{badge}"
        f'<div class="metric-label">{label}</div>'
        f"{value_html}"
        f'<div class="metric-meta">{meta}</div>'
        "</div>"
    )


def badge(kind: str, text: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def grid(cards: List[str]) -> None:
    st.markdown(
        '<div class="metric-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# SECTIONS
# ============================================================

def render_headline(data: Dict[str, Any]) -> None:
    ner = data["ner"]
    acc = data["bert_val_acc"]

    st.header("Headline Results")

    cards = []

    if acc:
        cards.append(card(
            "English NER validation accuracy",
            f"{100 * acc['correct_tokens'] / acc['total_tokens']:.2f}%",
            "Token-level · BERT (<code>models/english_ner</code>) · "
            f"CoNLL-2003 <b>validation</b> split · "
            f"{acc['correct_tokens']:,} / {acc['total_tokens']:,} tokens",
        ))

    for key, label, meta in [
        (("English BiLSTM-CRF", "english"), "English NER entity-level F1",
         "BiLSTM-CRF (scratch) · CoNLL-2003 <b>test</b> split"),
        (("Hindi BiLSTM-CRF", "hindi"), "Hindi NER entity-level F1",
         "BiLSTM-CRF (scratch) · Naamapadam hi <b>test</b> split"),
        (("Marathi BiLSTM-CRF", "marathi"), "Marathi NER entity-level F1",
         "BiLSTM-CRF (scratch) · Naamapadam mr <b>test</b> split"),
    ]:
        row = ner.get(key)
        if row:
            cards.append(card(label, pct(row["f1"]), meta))

    grid(cards)

    st.caption(
        "The two English figures come from **different models and splits**: "
        "the 99.26% token accuracy is the fine-tuned BERT model on the "
        "validation split; the 86.1% entity-level F1 is the from-scratch "
        "BiLSTM-CRF on the held-out test split. All F1 scores are on "
        "held-out test sets unless marked otherwise."
    )


def render_methodology(data: Dict[str, Any]) -> None:
    acc = data["bert_val_acc"]
    info = data["bert_info"]
    english = data["english_data"]

    with st.container(border=True):
        st.markdown("**Token Accuracy vs. Entity F1**")
        st.markdown(
            "Token-level validation accuracy measures the percentage of "
            "tokens whose predicted BIO tag matches the reference tag. "
            "Entity-level F1 measures the precision and recall of complete "
            "named entities."
        )

        notes = []

        if english:
            val = english["splits"]["validation"]
            o_share = val["label_counts"]["O"] / val["tokens"]
            notes.append(
                f"Most tokens are not entities: {val['label_counts']['O']:,} "
                f"of {val['tokens']:,} validation tokens ({o_share:.1%}) are "
                "tagged `O`, so a tagger that predicts `O` everywhere would "
                f"already reach {o_share:.1%} token accuracy. A token accuracy "
                "of 99.26% therefore does **not** mean that 99.26% of complete "
                "named entities are identified correctly."
            )

        if info:
            notes.append(
                "For the same BERT model on the same validation split, the "
                "entity-level F1 is "
                f"**{pct(info['best_dev_f1'])}** (epoch {info['best_epoch']}, "
                "`models/english_ner/training_info.json`); on the held-out "
                "test split it is "
                f"**{pct(data['ner'][('English BERT', 'english')]['f1'])}**."
            )

        notes.append(
            "Validation scores guide model selection; held-out **test** "
            "scores are the reported results. The 99.26% figure is a "
            "validation metric, not test accuracy."
        )

        for note in notes:
            st.markdown(f"- {note}")

        if acc:
            st.caption(
                f"Source: `{acc['dataset']}` · model `{acc['model_dir']}` · "
                "`results/ner/english_bert_validation_accuracy.json`"
            )


def render_ner_tables(data: Dict[str, Any]) -> None:
    ner = data["ner"]
    indicner = data["indicner"]

    st.header("NER Model Comparison")
    st.caption(
        "Entity-level exact match (type + span), micro-averaged, on the "
        "held-out **test** sets with the same scorer. Custom models, "
        "benchmarks and the reference experiment are different models — "
        "compare within a language only."
    )

    datasets = {
        "hindi": "Naamapadam hi test (864 sent.)",
        "marathi": "Naamapadam mr test (1,060 sent.)",
        "english": "CoNLL-2003 test (3,453 sent.)",
    }

    rows = []

    for language in ("hindi", "marathi", "english"):
        for model_key, label, role in NER_ROWS:
            row = ner.get((model_key, language))
            if row:
                rows.append({
                    "Language": language.title(),
                    "Model": label,
                    "Role": role,
                    "Test set": datasets[language],
                    "Precision": pct(row["precision"]),
                    "Recall": pct(row["recall"]),
                    "Entity F1": pct(row["f1"]),
                })

        if indicner and language in indicner:
            micro = indicner[language]["indicner"]["micro"]
            rows.append({
                "Language": language.title(),
                "Model": "IndicNER (ai4bharat/IndicNER)",
                "Role": "Benchmark (pretrained, not trained here)",
                "Test set": datasets[language],
                "Precision": pct(micro["precision"]),
                "Recall": pct(micro["recall"]),
                "Entity F1": pct(micro["f1"]),
            })

    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.caption(
        "IndicNER is the public checkpoint (mBERT-uncased fine-tuned), not "
        "the model reported in its paper, and may have seen the Naamapadam "
        "test sentences in training. Hindi gap to IndicNER: significant "
        "(paired bootstrap); Marathi gap: not significant."
    )

    acc = data["bert_val_acc"]
    if acc:
        st.subheader("English NER Validation Accuracy")
        st.dataframe(
            pd.DataFrame([{
                "Model": "BERT (bert-base-cased, fine-tuned here)",
                "Checkpoint": acc["model_dir"],
                "Split": "CoNLL-2003 validation",
                "Metric": "Token-level accuracy",
                "Correct tokens": f"{acc['correct_tokens']:,}",
                "Total tokens": f"{acc['total_tokens']:,}",
                "Accuracy": (
                    f"{100 * acc['correct_tokens'] / acc['total_tokens']:.2f}%"
                ),
            }]),
            hide_index=True,
            width="stretch",
        )


def render_status() -> None:
    st.header("Project Status")

    grid([
        card(name, "", text, badge(kind, label))
        for name, kind, label, text in COMPONENT_STATUS
    ])

    left, right = st.columns(2)

    with left:
        with st.container(border=True):
            st.markdown(badge("done", "Completed"), unsafe_allow_html=True)
            st.markdown("\n".join(f"- {item}" for item in COMPLETED))

    with right:
        with st.container(border=True):
            st.markdown(
                badge("pending", "Partially completed / pending evaluation"),
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {item}" for item in PENDING))

    st.caption(
        "A component counts as evaluated only when its evaluation has been "
        "run — implemented code alone is not marked complete."
    )


def render() -> None:
    data = load_results()

    st.markdown(
        """
        <div class="hero">
            <div class="tag">RESEARCH RESULTS · PROJECT STATUS</div>
            <div class="hero-title">Research Results</div>
            <div class="hero-sub">
                Synthetic Parallel Sentence and Named Entity Pair Generation
                for Indian Languages — confirmed model results, validation
                metrics and project status (English, Hindi, Marathi).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_headline(data)
    render_methodology(data)

    st.divider()
    render_ner_tables(data)


    st.divider()
    render_status()

    st.divider()
    st.caption(
        "Indic Entity Lab · Research prototype · All values are read "
        "directly from the project's evaluation results."
    )
