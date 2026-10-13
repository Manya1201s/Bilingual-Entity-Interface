"""
Fine-tune bert-base-cased for English NER (PER / ORG / LOC) on CoNLL-2003.

    python training/train_english.py

Transfer learning: the pretrained bert-base-cased encoder is loaded from the
Hugging Face Hub and a new token-classification head (7 labels) is trained
together with it. Only the FIRST sub-word of each word gets a label; the
other sub-words are ignored in the loss (label -100).

Model selection uses the validation split (micro entity F1). The test split
is NOT read here; it is evaluated once by evaluate.py.

Saved to models/english_ner/: Hugging Face model + tokenizer files
(config.json holds id2label/label2id) and training_info.json.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import warnings

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABEL2ID, LABELS, MODELS_DIR, entity_scores, read_jsonl, split_path  # noqa: E402


def make_batches(rows, batch_size, rng, shuffle=True):
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


def encode(tokenizer, rows, idx, max_length):
    enc = tokenizer([rows[i]["tokens"] for i in idx], is_split_into_words=True,
                    truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    labels = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for b, i in enumerate(idx):
        seen = set()
        for pos, wid in enumerate(enc.word_ids(b)):
            if wid is not None and wid not in seen:
                seen.add(wid)
                labels[b, pos] = LABEL2ID[rows[i]["tags"][wid]]
    return enc, labels


@torch.no_grad()
def evaluate(model, tokenizer, rows, max_length, batch_size=32):
    model.eval()
    gold, pred = [], []
    for idx in make_batches(rows, batch_size, None, shuffle=False):
        enc, labels = encode(tokenizer, rows, idx, max_length)
        logits = model(**enc).logits
        for b, i in enumerate(idx):
            tags = ["O"] * len(rows[i]["tokens"])
            seen = set()
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    tags[wid] = LABELS[int(logits[b, pos].argmax())]
            gold.append(rows[i]["tags"])
            pred.append(tags)
    return entity_scores(gold, pred)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="bert-base-cased")
    p.add_argument("--out", default=os.path.join(MODELS_DIR, "english_ner"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.1, help="fraction of steps for linear warm-up")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--dev-limit", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume-from", help="continue an interrupted run from a checkpoint folder "
                                         "written by this script (e.g. models/english_ner)")
    args = p.parse_args()

    # Resume: take the hyper-parameters from the saved run so the learning-rate
    # schedule and data order match the original run exactly.
    resume_info = None
    if args.resume_from:
        with open(os.path.join(args.resume_from, "training_info.json"), encoding="utf-8") as f:
            resume_info = json.load(f)
        saved = resume_info["training"]
        args.epochs, args.batch_size, args.lr = saved["epochs"], saved["batch_size"], saved["learning_rate"]
        args.weight_decay, args.warmup = saved["weight_decay"], saved["warmup_fraction"]
        args.max_length, args.seed = saved["max_length"], saved["seed"]
        args.base_model = resume_info["base_model"]

    from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train = read_jsonl(split_path("english", "train"))
    dev = read_jsonl(split_path("english", "validation"))
    if args.train_limit:
        train = train[:args.train_limit]
    if args.dev_limit:
        dev = dev[:args.dev_limit]

    source = args.resume_from or args.base_model
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForTokenClassification.from_pretrained(
        source, num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)), label2id=LABEL2ID)
    n_params = sum(p.numel() for p in model.parameters())

    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)

    info = {
        "model_type": "bert-base-cased fine-tuned for token classification (transfer learning)",
        "base_model": args.base_model, "language": "english", "labels": LABELS,
        "parameters": n_params,
        "training": {"dataset": "CoNLL-2003 (tner/conll2003), MISC -> O",
                     "train_sentences": len(train), "validation_sentences": len(dev),
                     "epochs": args.epochs, "batch_size": args.batch_size,
                     "optimizer": "AdamW", "learning_rate": args.lr,
                     "weight_decay": args.weight_decay, "warmup_fraction": args.warmup,
                     "schedule": "linear warm-up then linear decay", "max_length": args.max_length,
                     "label_alignment": "first sub-word of each word; others ignored (-100)",
                     "gradient_clip_norm": 1.0, "seed": args.seed, "threads": args.threads,
                     "hardware": "CPU only", "selection": "best validation micro entity F1"},
        "log": [],
    }
    print(f"[english] train={len(train)} dev={len(dev)} params={n_params:,} "
          f"steps/epoch={steps_per_epoch}", flush=True)

    best_f1, start, completed, previous_seconds = -1.0, time.time(), 0, 0.0
    if resume_info:
        completed = len(resume_info["log"])
        best_f1 = resume_info["best_dev_f1"]
        previous_seconds = resume_info["training"].get("seconds", 0.0)
        info["log"] = resume_info["log"]
        info["best_epoch"], info["best_dev_f1"] = resume_info["best_epoch"], best_f1
        info["training"]["resumed_after_epoch"] = completed
        info["training"]["resume_note"] = ("weights, learning-rate schedule and data order "
                                           "restored; AdamW moment estimates re-initialised")
        # Replay the shuffles and the LR schedule of the completed epochs.
        for _ in range(completed):
            make_batches(train, args.batch_size, rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(completed * steps_per_epoch):
                scheduler.step()
        print(f"[english] resuming after epoch {completed} (best dev F1 so far {best_f1:.4f}, "
              f"learning rate now {scheduler.get_last_lr()[0]:.2e})", flush=True)
        if completed >= args.epochs:
            print("[english] all planned epochs are already completed; nothing to do.", flush=True)
            return

    for epoch in range(completed + 1, args.epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for step, idx in enumerate(make_batches(train, args.batch_size, rng), start=1):
            enc, labels = encode(tokenizer, train, idx, args.max_length)
            loss = model(**enc, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if step % 100 == 0:
                rate = step / (time.time() - t0)
                print(f"[english] epoch {epoch} step {step}/{steps_per_epoch} loss {total / step:.4f} "
                      f"({rate:.2f} steps/s, ~{(steps_per_epoch - step) / rate / 60:.0f} min left in epoch)",
                      flush=True)
        scores = evaluate(model, tokenizer, dev, args.max_length)
        f1 = scores["micro"]["f1"]
        entry = {"epoch": epoch, "train_loss": round(total / steps_per_epoch, 4),
                 "dev_precision": scores["micro"]["precision"], "dev_recall": scores["micro"]["recall"],
                 "dev_f1": f1, "dev_per_type_f1": {t: scores[t]["f1"] for t in ("PER", "ORG", "LOC")},
                 "epoch_seconds": round(time.time() - t0, 1)}
        info["log"].append(entry)
        improved = f1 > best_f1
        if improved:
            best_f1 = f1
            info["best_epoch"], info["best_dev_f1"] = epoch, f1
            os.makedirs(args.out, exist_ok=True)
            model.save_pretrained(args.out)
            tokenizer.save_pretrained(args.out)
        print(f"[english] epoch {epoch}: loss {entry['train_loss']:.4f} dev F1 {f1:.4f} "
              f"(P {entry['dev_precision']:.4f} R {entry['dev_recall']:.4f}) "
              f"{entry['epoch_seconds'] / 60:.1f} min{' *saved*' if improved else ''}", flush=True)
        info["training"]["seconds"] = round(previous_seconds + time.time() - start, 1)
        with open(os.path.join(args.out, "training_info.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)

    print(f"[english] done: best dev F1 {best_f1:.4f}, "
          f"{info['training']['seconds'] / 60:.1f} min -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
