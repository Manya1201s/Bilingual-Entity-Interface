"""
Trained NER models used by app.py and by the training scripts.

* BiLSTM-CRF (Hindi, Marathi): trained FROM SCRATCH. Word and character
  embeddings are randomly initialised (nn.Embedding default init); no
  pretrained embeddings or language models are loaded anywhere.
* BERT (English): bert-base-cased fine-tuned for token classification.

Both predictors expose the same interface:

    predict(text) -> list of {"entity_group", "score", "start", "end", "word"}

which is the format of the Hugging Face "simple" aggregation pipeline, so
app.py post-processes trained-model output exactly like baseline output.
`score` is a real model probability: for the CRF the forward-backward
marginal probability of the predicted tag, for BERT the softmax probability
of the predicted label, averaged over the tokens of the entity.
"""

import json
import os
import unicodedata
from typing import Dict, List, Sequence, Tuple

import torch
from torch import nn

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

PAD, UNK = "<pad>", "<unk>"


# ----------------------------------------------------------------------
# Tokenisation (shared by training and inference)
# ----------------------------------------------------------------------

def _is_word_char(char: str) -> bool:
    # Letters/digits plus combining marks (Devanagari vowel signs, virama,
    # anusvara, nukta) and ZWJ/ZWNJ belong to the same word.
    return (char.isalnum() or unicodedata.category(char).startswith("M")
            or char in "‌‍")


def tokenize_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into words and single punctuation marks, keeping character
    offsets into the original text. Matches the training data, where
    punctuation (। . , ? etc.) is a separate token and Marathi suffixes stay
    attached to their word.
    """
    tokens, i, n = [], 0, len(text)
    while i < n:
        char = text[i]
        if char.isspace():
            i += 1
        elif _is_word_char(char):
            j = i + 1
            while j < n and _is_word_char(text[j]):
                j += 1
            tokens.append((text[i:j], i, j))
            i = j
        else:
            tokens.append((char, i, i + 1))
            i += 1
    return tokens


def normalize_word(word: str) -> str:
    """Digits -> 0 (any script) so numbers share embeddings."""
    return "".join("0" if c.isdigit() else c for c in word)


def bio_to_entities(tags: Sequence[str], probs: Sequence[float]):
    """(type, first_token, last_token_exclusive, mean_prob) with conlleval
    semantics: an I-X that cannot continue an entity starts a new one."""
    entities, current = [], None
    for i, tag in enumerate(list(tags) + ["O"]):
        prefix, etype = tag.split("-", 1) if tag != "O" else ("O", "")
        if current and prefix == "I" and etype == current[0]:
            continue
        if current:
            span = probs[current[1]:i]
            entities.append((current[0], current[1], i, sum(span) / len(span)))
            current = None
        if prefix in ("B", "I"):
            current = (etype, i)
    return entities


def _entities_to_output(text, tokens, tags, probs):
    output = []
    for etype, first, last, prob in bio_to_entities(tags, probs):
        start, end = tokens[first][1], tokens[last - 1][2]
        output.append({"entity_group": etype, "score": float(prob),
                       "start": start, "end": end, "word": text[start:end]})
    return output


# ----------------------------------------------------------------------
# CRF
# ----------------------------------------------------------------------

class CRF(nn.Module):
    """Linear-chain CRF with start/end transitions (batch-first)."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags).uniform_(-0.1, 0.1))
        self.start = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))
        self.end = nn.Parameter(torch.empty(num_tags).uniform_(-0.1, 0.1))

    def _forward_alphas(self, emissions, mask):
        alpha = self.start + emissions[:, 0]
        alphas = [alpha]
        for t in range(1, emissions.size(1)):
            scores = alpha.unsqueeze(2) + self.transitions + emissions[:, t].unsqueeze(1)
            alpha = torch.where(mask[:, t:t + 1], torch.logsumexp(scores, dim=1), alpha)
            alphas.append(alpha)
        return alphas

    def neg_log_likelihood(self, emissions, tags, mask):
        """Mean negative log-likelihood of the gold tag sequences."""
        batch = torch.arange(emissions.size(0))
        score = self.start[tags[:, 0]] + emissions[batch, 0, tags[:, 0]]
        for t in range(1, emissions.size(1)):
            step = self.transitions[tags[:, t - 1], tags[:, t]] + emissions[batch, t, tags[:, t]]
            score = score + step * mask[:, t]
        last = tags.gather(1, (mask.sum(1) - 1).unsqueeze(1)).squeeze(1)
        score = score + self.end[last]
        log_z = torch.logsumexp(self._forward_alphas(emissions, mask)[-1] + self.end, dim=1)
        return (log_z - score).mean()

    @torch.no_grad()
    def viterbi_decode(self, emissions, mask) -> List[List[int]]:
        score = self.start + emissions[:, 0]
        history = []
        for t in range(1, emissions.size(1)):
            scores = score.unsqueeze(2) + self.transitions + emissions[:, t].unsqueeze(1)
            best, idx = scores.max(dim=1)
            score = torch.where(mask[:, t:t + 1], best, score)
            history.append(idx)
        score = score + self.end
        lengths = mask.sum(1).tolist()
        paths = []
        for b, length in enumerate(lengths):
            tag = int(score[b].argmax())
            path = [tag]
            for t in range(length - 1, 0, -1):
                tag = int(history[t - 1][b, tag])
                path.append(tag)
            paths.append(path[::-1])
        return paths

    @torch.no_grad()
    def marginals(self, emissions, mask):
        """P(tag_t = k | sentence) for every position (forward-backward)."""
        alphas = self._forward_alphas(emissions, mask)
        steps = emissions.size(1)
        betas = [None] * steps
        beta = self.end.expand(emissions.size(0), -1)
        betas[-1] = beta
        for t in range(steps - 2, -1, -1):
            nxt = torch.logsumexp(
                self.transitions.unsqueeze(0) + (emissions[:, t + 1] + beta).unsqueeze(1), dim=2)
            beta = torch.where(mask[:, t + 1:t + 2], nxt, self.end.expand_as(nxt))
            betas[t] = beta
        log_z = torch.logsumexp(alphas[-1] + self.end, dim=1, keepdim=True)
        return torch.stack([torch.exp(a + b - log_z) for a, b in zip(alphas, betas)], dim=1)


# ----------------------------------------------------------------------
# BiLSTM-CRF
# ----------------------------------------------------------------------

class BiLSTMCRF(nn.Module):
    """
    words  -> random word Embedding ----------------------\
    chars  -> random char Embedding -> CNN -> max-pool ---+-> BiLSTM -> Linear -> CRF
    (the character branch is optional, see config["use_chars"])
    """

    def __init__(self, vocab_size, char_vocab_size, num_tags, config):
        super().__init__()
        self.config = config
        self.word_embedding = nn.Embedding(vocab_size, config["embedding_dim"], padding_idx=0)
        input_dim = config["embedding_dim"]
        if config["use_chars"]:
            self.char_embedding = nn.Embedding(char_vocab_size, config["char_dim"], padding_idx=0)
            self.char_cnn = nn.Conv1d(config["char_dim"], config["char_filters"], kernel_size=3, padding=1)
            input_dim += config["char_filters"]
        self.dropout = nn.Dropout(config["dropout"])
        self.lstm = nn.LSTM(input_dim, config["hidden_dim"], num_layers=config["num_layers"],
                            bidirectional=True, batch_first=True,
                            dropout=config["dropout"] if config["num_layers"] > 1 else 0.0)
        self.hidden2tag = nn.Linear(2 * config["hidden_dim"], num_tags)
        self.crf = CRF(num_tags)

    def emissions(self, words, chars, mask):
        x = self.word_embedding(words)
        if self.config["use_chars"]:
            b, t, w = chars.shape
            c = self.char_embedding(chars.view(b * t, w)).transpose(1, 2)
            c = torch.relu(self.char_cnn(c))
            c = c.masked_fill((chars.view(b * t, w) == 0).unsqueeze(1), -1e4).max(dim=2).values
            c = torch.where(c < -1e3, torch.zeros_like(c), c)
            x = torch.cat([x, c.view(b, t, -1)], dim=2)
        x = self.dropout(x)
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=words.size(1))
        return self.hidden2tag(self.dropout(out))

    def loss(self, words, chars, tags, mask):
        return self.crf.neg_log_likelihood(self.emissions(words, chars, mask), tags, mask)


class Vocab:
    def __init__(self, words: List[str], chars: List[str]):
        self.words = words
        self.chars = chars
        self.word2id = {w: i for i, w in enumerate(words)}
        self.char2id = {c: i for i, c in enumerate(chars)}

    def encode(self, sentences: List[List[str]], max_word_len: int = 20):
        """Pad a batch -> (words [B,T], chars [B,T,W], mask [B,T])."""
        steps = max(len(s) for s in sentences)
        word_len = min(max_word_len, max(len(w) for s in sentences for w in s))
        words = torch.zeros(len(sentences), steps, dtype=torch.long)
        chars = torch.zeros(len(sentences), steps, word_len, dtype=torch.long)
        mask = torch.zeros(len(sentences), steps, dtype=torch.bool)
        unk, cunk = self.word2id[UNK], self.char2id[UNK]
        for b, sentence in enumerate(sentences):
            for t, word in enumerate(sentence):
                norm = normalize_word(word)
                words[b, t] = self.word2id.get(norm, unk)
                for k, c in enumerate(norm[:word_len]):
                    chars[b, t, k] = self.char2id.get(c, cunk)
                mask[b, t] = True
        return words, chars, mask


class BiLSTMCRFPredictor:
    """Loads models/<language>_ner/ (config.json, vocab.json, model.pt)."""

    kind = "bilstm_crf"

    def __init__(self, model: BiLSTMCRF, vocab: Vocab, labels: List[str], info: Dict):
        self.model = model.eval()
        self.vocab = vocab
        self.labels = labels
        self.info = info

    @classmethod
    def load(cls, model_dir: str):
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            info = json.load(f)
        with open(os.path.join(model_dir, "vocab.json"), encoding="utf-8") as f:
            vocab_data = json.load(f)
        vocab = Vocab(vocab_data["words"], vocab_data["chars"])
        model = BiLSTMCRF(len(vocab.words), len(vocab.chars), len(info["labels"]), info["model"])
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return cls(model, vocab, info["labels"], info)

    @torch.no_grad()
    def predict_tokens(self, sentences: List[List[str]]):
        """Gold-tokenised input -> (tags, per-token probabilities) per sentence."""
        results = [([], []) for _ in sentences]
        order = [i for i, s in enumerate(sentences) if s]
        for k in range(0, len(order), 64):
            idx = order[k:k + 64]
            words, chars, mask = self.vocab.encode([sentences[i] for i in idx])
            emissions = self.model.emissions(words, chars, mask)
            paths = self.model.crf.viterbi_decode(emissions, mask)
            marg = self.model.crf.marginals(emissions, mask)
            for row, (i, path) in enumerate(zip(idx, paths)):
                probs = [float(marg[row, t, tag]) for t, tag in enumerate(path)]
                results[i] = ([self.labels[t] for t in path], probs)
        return results

    def predict(self, text: str) -> List[Dict]:
        tokens = tokenize_with_offsets(text)
        if not tokens:
            return []
        tags, probs = self.predict_tokens([[t[0] for t in tokens]])[0]
        return _entities_to_output(text, tokens, tags, probs)


# ----------------------------------------------------------------------
# English BERT
# ----------------------------------------------------------------------

class BertNERPredictor:
    """Loads models/english_ner/ (Hugging Face tokenizer + model)."""

    kind = "bert"
    MAX_WORDS = 200  # long inputs are processed in windows of words

    def __init__(self, tokenizer, model, info: Dict):
        self.tokenizer = tokenizer
        self.model = model.eval()
        self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        self.info = info

    @classmethod
    def load(cls, model_dir: str):
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        info_path = os.path.join(model_dir, "training_info.json")
        info = {}
        if os.path.exists(info_path):
            with open(info_path, encoding="utf-8") as f:
                info = json.load(f)
        return cls(AutoTokenizer.from_pretrained(model_dir),
                   AutoModelForTokenClassification.from_pretrained(model_dir), info)

    @torch.no_grad()
    def predict_tokens(self, sentences: List[List[str]], batch_size: int = 32):
        results = [([], []) for _ in sentences]
        jobs = []  # (sentence index, word offset, words)
        for i, words in enumerate(sentences):
            for k in range(0, len(words), self.MAX_WORDS):
                jobs.append((i, words[k:k + self.MAX_WORDS]))
        for k in range(0, len(jobs), batch_size):
            batch = jobs[k:k + batch_size]
            enc = self.tokenizer([w for _, w in batch], is_split_into_words=True,
                                 truncation=True, max_length=512, padding=True,
                                 return_tensors="pt")
            probs = torch.softmax(self.model(**{k2: v for k2, v in enc.items()
                                                if k2 in ("input_ids", "attention_mask", "token_type_ids")}).logits, dim=-1)
            for row, (i, words) in enumerate(batch):
                word_ids = enc.word_ids(row)
                tags = ["O"] * len(words)
                scores = [1.0] * len(words)
                seen = set()
                for pos, wid in enumerate(word_ids):
                    if wid is None or wid in seen:
                        continue  # label only the first sub-word of a word
                    seen.add(wid)
                    p, label = probs[row, pos].max(dim=0)
                    tags[wid] = self.labels[int(label)]
                    scores[wid] = float(p)
                results[i][0].extend(tags)
                results[i][1].extend(scores)
        return results

    def predict(self, text: str) -> List[Dict]:
        tokens = tokenize_with_offsets(text)
        if not tokens:
            return []
        tags, probs = self.predict_tokens([[t[0] for t in tokens]])[0]
        return _entities_to_output(text, tokens, tags, probs)


def load_predictor(model_dir: str):
    """Pick the right predictor class from the files in model_dir."""
    if os.path.exists(os.path.join(model_dir, "vocab.json")):
        return BiLSTMCRFPredictor.load(model_dir)
    return BertNERPredictor.load(model_dir)
