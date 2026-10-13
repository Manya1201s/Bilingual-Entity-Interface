"""Train the English BiLSTM-CRF from scratch -> models/english_ner_scratch/.

    python training/train_english_scratch.py   (extra options: see train_bilstm_crf.py)

Same architecture, hyper-parameters, CRF, training loop and checkpoint
format as the Hindi and Marathi models. One difference: the English
training set (CoNLL-2003, ~201k tokens) is ~10x smaller than the Hindi /
Marathi ones (~2.2M / ~1.8M tokens), so one epoch is ~10x fewer updates.
The epoch limit is therefore raised to 30 with early stopping (patience 5)
on validation F1, instead of 8 epochs / patience 2.

The fine-tuned BERT experiment stays in models/english_ner/ and is not
touched by this script.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import MODELS_DIR  # noqa: E402
from train_bilstm_crf import main  # noqa: E402

if __name__ == "__main__":
    main(defaults={"language": "english", "epochs": 30, "patience": 5,
                   "out": os.path.join(MODELS_DIR, "english_ner_scratch")})
