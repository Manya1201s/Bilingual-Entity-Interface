"""
Train an English -> Hindi / Marathi Transformer from scratch.

    python mt/train_mt.py --lang hi
    python mt/train_mt.py --lang mr
    python mt/train_mt.py --lang hi --resume      (continue an interrupted run)

* SentencePiece unigram vocabularies (English and target, 8,000 pieces each)
  are learned from the TRAINING split only.
* Model: mt/model.py (pre-LN Transformer encoder-decoder, random init).
* Loss: label-smoothed cross-entropy (0.1); Adam(0.9, 0.98) with linear
  warm-up then inverse-square-root decay; gradient clipping 1.0.
* Every epoch: validation loss (plain cross-entropy) and greedy-decoding
  BLEU / chrF on the validation split.
* The checkpoint with the lowest validation loss is saved as model.pt;
  last_checkpoint.pt (model + optimiser + schedule + RNG) allows --resume.
* The test split is NOT read here (see evaluate_mt.py).

Output: models/mt_en_<lang>/ (config.json with the full log, model.pt,
spm.en.model, spm.<lang>.model, last_checkpoint.pt).
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

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")
warnings.filterwarnings("ignore", message=".*mismatched key_padding_mask.*")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
from mt.model import BOS_ID, EOS_ID, PAD_ID, TransformerMT  # noqa: E402

DATA_DIR = os.path.join(HERE, "data")
MODELS_DIR = os.path.join(PROJECT, "models")


def read_split(lang, split):
    with open(os.path.join(DATA_DIR, lang, f"{split}.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def train_spm(texts, prefix, vocab_size, seed):
    import sentencepiece as spm
    tmp = prefix + ".input.txt"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    spm.SentencePieceTrainer.train(
        input=tmp, model_prefix=prefix, vocab_size=vocab_size, model_type="unigram",
        character_coverage=1.0, pad_id=PAD_ID, unk_id=1, bos_id=BOS_ID, eos_id=EOS_ID,
        input_sentence_size=1_000_000, shuffle_input_sentence=True, seed_sentencepiece_size=200_000,
        num_threads=4, minloglevel=2)
    os.remove(tmp)
    if os.path.exists(prefix + ".vocab"):
        os.remove(prefix + ".vocab")
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def encode(rows, lang, sp_src, sp_tgt, max_len):
    data, truncated = [], 0
    for r in rows:
        s = sp_src.encode(r["en"])
        t = sp_tgt.encode(r[lang])
        truncated += len(s) >= max_len or len(t) >= max_len
        data.append((s[:max_len - 1] + [EOS_ID], [BOS_ID] + t[:max_len - 1] + [EOS_ID]))
    return data, truncated


def batches(data, batch_size, rng, shuffle=True):
    order = list(range(len(data)))
    if shuffle:
        rng.shuffle(order)
    pool = batch_size * 50
    result = []
    for k in range(0, len(order), pool):
        chunk = sorted(order[k:k + pool], key=lambda i: (len(data[i][0]), len(data[i][1])))
        result += [chunk[j:j + batch_size] for j in range(0, len(chunk), batch_size)]
    if shuffle:
        rng.shuffle(result)
    return result


def pad(seqs):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD_ID] * (width - len(s)) for s in seqs])


def lr_at(step, peak, warmup):
    step = max(step, 1)
    return peak * min(step / warmup, math.sqrt(warmup / step))


@torch.no_grad()
def validate(model, data, rows, lang, sp_tgt, batch_size=64):
    import sacrebleu
    model.eval()
    total, tokens = 0.0, 0
    for idx in batches(data, batch_size, None, shuffle=False):
        src = pad([data[i][0] for i in idx])
        tgt = pad([data[i][1] for i in idx])
        logits = model(src, tgt[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1),
            ignore_index=PAD_ID, reduction="sum")
        total += loss.item()
        tokens += tgt[:, 1:].ne(PAD_ID).sum().item()
    # Greedy BLEU/chrF on the first 500 validation pairs (monitoring only;
    # checkpoint selection uses the validation loss on all pairs).
    sub = data[:500]
    hyps = [None] * len(sub)
    for idx in batches(sub, batch_size, None, shuffle=False):
        for i, ids in zip(idx, model.greedy(pad([sub[i][0] for i in idx]))):
            hyps[i] = sp_tgt.decode(ids)
    refs = [r[lang] for r in rows[:500]]
    return (total / tokens,
            sacrebleu.corpus_bleu(hyps, [refs]).score,
            sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", required=True, choices=["hi", "mr"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=7e-4, help="peak learning rate")
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--vocab-size", type=int, default=8000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=3, help="encoder and decoder layers")
    p.add_argument("--ffn", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    lang = args.lang
    out = os.path.join(MODELS_DIR, f"mt_en_{lang}")
    os.makedirs(out, exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    train_rows, val_rows = read_split(lang, "train"), read_split(lang, "validation")
    if args.resume and os.path.exists(os.path.join(out, f"spm.{lang}.model")):
        import sentencepiece as spm
        sp_src = spm.SentencePieceProcessor(model_file=os.path.join(out, "spm.en.model"))
        sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(out, f"spm.{lang}.model"))
    else:
        sp_src = train_spm([r["en"] for r in train_rows], os.path.join(out, "spm.en"), args.vocab_size, args.seed)
        sp_tgt = train_spm([r[lang] for r in train_rows], os.path.join(out, f"spm.{lang}"), args.vocab_size, args.seed)
    train, trunc_train = encode(train_rows, lang, sp_src, sp_tgt, args.max_len)
    val, _ = encode(val_rows, lang, sp_src, sp_tgt, args.max_len)

    config = {"d_model": args.d_model, "heads": args.heads, "encoder_layers": args.layers,
              "decoder_layers": args.layers, "ffn_dim": args.ffn, "dropout": args.dropout,
              "max_positions": 512, "max_source_tokens": args.max_len}
    model = TransformerMT(sp_src.get_piece_size(), sp_tgt.get_piece_size(), config)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)

    info = {
        "model_type": "Transformer encoder-decoder trained from scratch (no pretrained weights)",
        "source_language": "en", "target_language": lang, "model": config, "parameters": n_params,
        "vocabulary": {"type": "SentencePiece unigram, learned on the training split only",
                       "en": sp_src.get_piece_size(), lang: sp_tgt.get_piece_size()},
        "training": {"train_pairs": len(train), "validation_pairs": len(val),
                     "pairs_truncated_at_max_len": trunc_train, "epochs_max": args.epochs,
                     "patience": args.patience, "batch_size_sentences": args.batch_size,
                     "optimizer": "Adam(0.9, 0.98, eps 1e-9)", "peak_learning_rate": args.lr,
                     "schedule": f"linear warm-up {args.warmup} steps, then inverse square root",
                     "label_smoothing": args.label_smoothing, "gradient_clip_norm": 1.0,
                     "seed": args.seed, "threads": args.threads, "hardware": "CPU only",
                     "selection": "lowest validation cross-entropy"},
        "log": [],
    }
    start_epoch, step, best, bad, previous_seconds = 1, 0, float("inf"), 0, 0.0
    last_path = os.path.join(out, "last_checkpoint.pt")
    if args.resume and os.path.exists(last_path):
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        torch.set_rng_state(ckpt["torch_rng"])
        rng.setstate(ckpt["python_rng"])
        start_epoch, step, best, bad = ckpt["epoch"] + 1, ckpt["step"], ckpt["best"], ckpt["bad"]
        info = ckpt["info"]
        previous_seconds = info["training"].get("seconds", 0.0)
        print(f"[{lang}] resumed after epoch {ckpt['epoch']} (step {step})", flush=True)

    print(f"[{lang}] train={len(train)} val={len(val)} params={n_params:,} vocab={info['vocabulary']} "
          f"steps/epoch={math.ceil(len(train) / args.batch_size)}", flush=True)
    t_start = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for b, idx in enumerate(batches(train, args.batch_size, rng), start=1):
            step += 1
            for g in optimizer.param_groups:
                g["lr"] = lr_at(step, args.lr, args.warmup)
            src = pad([train[i][0] for i in idx])
            tgt = pad([train[i][1] for i in idx])
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
            n += 1
            if b % 200 == 0:
                print(f"[{lang}] epoch {epoch} batch {b} train loss {total / n:.4f} "
                      f"lr {optimizer.param_groups[0]['lr']:.2e} ({(time.time() - t0) / 60:.1f} min)", flush=True)
        val_loss, val_bleu, val_chrf = validate(model, val, val_rows, lang, sp_tgt)
        entry = {"epoch": epoch, "step": step, "train_loss": round(total / n, 4),
                 "val_loss": round(val_loss, 4), "val500_bleu_greedy": round(val_bleu, 2),
                 "val500_chrf++_greedy": round(val_chrf, 2), "epoch_minutes": round((time.time() - t0) / 60, 1)}
        info["log"].append(entry)
        improved = val_loss < best
        if improved:
            best, bad = val_loss, 0
            info["best_epoch"], info["best_val_loss"] = epoch, round(val_loss, 4)
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
        else:
            bad += 1
        info["training"]["seconds"] = round(previous_seconds + time.time() - t_start, 1)
        info["training"]["epochs_run"] = len(info["log"])
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(), "python_rng": rng.getstate(),
                    "epoch": epoch, "step": step, "best": best, "bad": bad, "info": info}, last_path)
        print(f"[{lang}] epoch {epoch}: train loss {entry['train_loss']:.4f} val loss {val_loss:.4f} "
              f"val BLEU {val_bleu:.2f} chrF++ {val_chrf:.2f} {entry['epoch_minutes']:.1f} min"
              f"{' *saved*' if improved else ''}", flush=True)
        if bad >= args.patience:
            print(f"[{lang}] early stopping", flush=True)
            break
    print(f"[{lang}] done: best val loss {best:.4f} at epoch {info.get('best_epoch')}, "
          f"{info['training']['seconds'] / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
