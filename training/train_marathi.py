"""Train the Marathi BiLSTM-CRF from scratch -> models/marathi_ner/.

    python training/train_marathi.py          (extra options: see train_bilstm_crf.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bilstm_crf import main  # noqa: E402

if __name__ == "__main__":
    main(defaults={"language": "marathi"})
