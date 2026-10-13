| Model | Language | Training method | Test set | Precision | Recall | F1 | PER F1 | ORG F1 | LOC F1 |
|---|---|---|---|---|---|---|---|---|---|
| Hindi BiLSTM-CRF | Hindi | BiLSTM-CRF trained from scratch | Naamapadam hi test (human-annotated) (864 sent.) | 0.789 | 0.805 | **0.797** | 0.842 | 0.729 | 0.798 |
| Baseline XLM-R | Hindi | pretrained Davlan/xlm-roberta-base-ner-hrl (no training here) | Naamapadam hi test (human-annotated) (864 sent.) | 0.821 | 0.778 | **0.799** | 0.894 | 0.643 | 0.795 |
| Marathi BiLSTM-CRF | Marathi | BiLSTM-CRF trained from scratch | Naamapadam mr test (human-annotated) (1060 sent.) | 0.826 | 0.764 | **0.794** | 0.866 | 0.659 | 0.794 |
| Baseline XLM-R | Marathi | pretrained Davlan/xlm-roberta-base-ner-hrl (no training here) | Naamapadam mr test (human-annotated) (1060 sent.) | 0.843 | 0.748 | **0.792** | 0.882 | 0.581 | 0.814 |
| English BERT | English | bert-base-cased fine-tuned (experiment) | CoNLL-2003 test (3453 sent.) | 0.917 | 0.929 | **0.923** | 0.957 | 0.883 | 0.931 |
| English BiLSTM-CRF | English | BiLSTM-CRF trained from scratch | CoNLL-2003 test (3453 sent.) | 0.865 | 0.856 | **0.861** | 0.893 | 0.794 | 0.898 |
| Baseline XLM-R | English | pretrained Davlan/xlm-roberta-base-ner-hrl (no training here) | CoNLL-2003 test (3453 sent.) | 0.903 | 0.926 | **0.914** | 0.951 | 0.867 | 0.927 |
