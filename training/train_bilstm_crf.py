"""
Train a BiLSTM-CRF NER model FROM SCRATCH (shared by Hindi and Marathi).

    python training/train_bilstm_crf.py --language hindi
    (or the wrappers training/train_hindi.py, training/train_marathi.py)

Nothing pretrained is used: the vocabulary is built from the training split
only and the word/character embeddings start from PyTorch's random
nn.Embedding initialisation (N(0, 1)).

Model selection uses the validation split (micro entity F1, early stopping).
The test split is NOT read here; it is evaluated once by evaluate.py.

Saved to models/<language>_ner/:
    model.pt      weights (state_dict) of the best epoch
    vocab.json    word and character vocabularies
    config.json   architecture, labels, hyper-parameters, training log
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABEL2ID, LABELS, MODELS_DIR, entity_scores, read_jsonl, split_path  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trained_ner import PAD, UNK, BiLSTMCRF, Vocab, normalize_word  # noqa: E402


def build_vocab(rows, min_freq, max_size, char_min_freq):
    words = Counter(normalize_word(w) for r in rows for w in r["tokens"])
    chars = Counter(c for w, n in words.items() for c in w for _ in range(n))
    word_list = [w for w, n in words.most_common(max_size) if n >= min_freq]
    char_list = [c for c, n in chars.most_common() if n >= char_min_freq]
    return Vocab([PAD, UNK] + word_list, [PAD, UNK] + char_list)


def batches(rows, batch_size, rng, shuffle=True):
    """Length-bucketed batches: little padding -> much faster on CPU."""
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    pool = batch_size * 50
    result = []
    for k in range(0, len(order), pool):
        chunk = sorted(order[k:k + pool], key=lambda i: len(rows[i]["tokens"]))
        result += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    if shuffle:
        rng.shuffle(result)
    return result


def encode_batch(vocab, rows, idx, word_dropout=0.0, rng=None):
    words, chars, mask = vocab.encode([rows[i]["tokens"] for i in idx])
    tags = torch.zeros_like(words)
    for b, i in enumerate(idx):
        tags[b, :len(rows[i]["tags"])] = torch.tensor([LABEL2ID[t] for t in rows[i]["tags"]])
    if word_dropout > 0:
        drop = (torch.rand(words.shape, generator=rng) < word_dropout) & mask
        words = words.masked_fill(drop, vocab.word2id[UNK])
    return words, chars, tags, mask


@torch.no_grad()
def evaluate(model, vocab, rows, batch_size=128):
    model.eval()
    gold, pred = [], []
    for idx in batches(rows, batch_size, None, shuffle=False):
        words, chars, _, mask = encode_batch(vocab, rows, idx)
        paths = model.crf.viterbi_decode(model.emissions(words, chars, mask), mask)
        for i, path in zip(idx, paths):
            gold.append(rows[i]["tags"])
            pred.append([LABELS[t] for t in path])
    return entity_scores(gold, pred)


def save(model, vocab, info, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"words": vocab.words, "chars": vocab.chars}, f, ensure_ascii=False)
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)


def main(argv=None, defaults=None):
    p = argparse.ArgumentParser()
    p.add_argument("--language", required=defaults is None)
    p.add_argument("--out", help="output folder (default models/<language>_ner)")
    p.add_argument("--train-limit", type=int, default=0, help="use only the first N training sentences")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=3, help="early stopping on validation F1")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=128, help="per direction (BiLSTM output = 2x)")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--word-dropout", type=float, default=0.05, help="prob. of replacing a word by <unk> in training")
    p.add_argument("--no-chars", action="store_true", help="disable the character-CNN branch")
    p.add_argument("--char-dim", type=int, default=30)
    p.add_argument("--char-filters", type=int, default=50)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-vocab", type=int, default=50000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    if defaults:
        p.set_defaults(**defaults)
    args = p.parse_args(argv)

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    out_dir = args.out or os.path.join(MODELS_DIR, f"{args.language}_ner")

    train = read_jsonl(split_path(args.language, "train"))
    if args.train_limit:
        train = train[:args.train_limit]
    dev = read_jsonl(split_path(args.language, "validation"))
    vocab = build_vocab(train, args.min_freq, args.max_vocab, char_min_freq=3)

    config = {"embedding_dim": args.embedding_dim, "hidden_dim": args.hidden_dim,
              "num_layers": args.num_layers, "dropout": args.dropout,
              "use_chars": not args.no_chars, "char_dim": args.char_dim,
              "char_filters": args.char_filters}
    model = BiLSTMCRF(len(vocab.words), len(vocab.chars), len(LABELS), config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    info = {
        "model_type": "BiLSTM-CRF trained from scratch (no pretrained embeddings or language model)",
        "language": args.language, "labels": LABELS, "model": config,
        "parameters": n_params, "vocab_size": len(vocab.words), "char_vocab_size": len(vocab.chars),
        "training": {"train_sentences": len(train), "validation_sentences": len(dev),
                     "epochs_max": args.epochs, "patience": args.patience,
                     "batch_size": args.batch_size, "optimizer": "Adam", "learning_rate": args.lr,
                     "gradient_clip_norm": 5.0, "word_dropout": args.word_dropout,
                     "min_word_freq": args.min_freq, "seed": args.seed, "threads": args.threads,
                     "hardware": "CPU only", "selection": "best validation micro entity F1"},
        "log": [],
    }
    print(f"[{args.language}] train={len(train)} dev={len(dev)} vocab={len(vocab.words)} "
          f"chars={len(vocab.chars)} params={n_params:,} config={config}", flush=True)

    best_f1, best_epoch, bad_epochs, start = -1.0, 0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, steps = time.time(), 0.0, 0
        for idx in batches(train, args.batch_size, rng):
            words, chars, tags, mask = encode_batch(vocab, train, idx, args.word_dropout, gen)
            loss = model.loss(words, chars, tags, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item()
            steps += 1
        scores = evaluate(model, vocab, dev)
        f1 = scores["micro"]["f1"]
        entry = {"epoch": epoch, "train_loss": round(total / steps, 4),
                 "dev_precision": scores["micro"]["precision"], "dev_recall": scores["micro"]["recall"],
                 "dev_f1": f1, "dev_per_type_f1": {t: scores[t]["f1"] for t in ("PER", "ORG", "LOC")},
                 "epoch_seconds": round(time.time() - t0, 1)}
        info["log"].append(entry)
        improved = f1 > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = f1, epoch, 0
            info["best_epoch"], info["best_dev_f1"] = epoch, f1
            info["training"]["seconds"] = round(time.time() - start, 1)
            save(model, vocab, info, out_dir)
        else:
            bad_epochs += 1
        print(f"[{args.language}] epoch {epoch}: loss {entry['train_loss']:.4f} dev F1 {f1:.4f} "
              f"(P {entry['dev_precision']:.4f} R {entry['dev_recall']:.4f}) "
              f"{entry['epoch_seconds']:.0f}s{' *saved*' if improved else ''}", flush=True)
        if bad_epochs >= args.patience:
            print(f"[{args.language}] early stopping (no improvement for {args.patience} epochs)", flush=True)
            break

    # Record the full log + total time in the saved config (weights = best epoch).
    info["training"]["seconds"] = round(time.time() - start, 1)
    info["training"]["epochs_run"] = len(info["log"])
    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[{args.language}] done: best dev F1 {best_f1:.4f} at epoch {best_epoch}, "
          f"{info['training']['seconds'] / 60:.1f} min -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
