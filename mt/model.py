"""
Transformer encoder-decoder for English -> Hindi / Marathi translation,
trained FROM SCRATCH in this project (random initialisation, SentencePiece
vocabularies learned from the training split only, no pretrained weights).

    English SentencePiece ids -> Embedding*sqrt(d) + sinusoidal positions
        -> Transformer encoder (pre-LayerNorm)
    Target SentencePiece ids  -> Embedding*sqrt(d) + sinusoidal positions
        -> Transformer decoder (pre-LayerNorm, causal) -> Linear (tied with
           the target embedding) -> softmax over the target vocabulary

Decoding: batched greedy search and beam search with GNMT length penalty.
"""

import json
import math
import os
from typing import List

import torch
from torch import nn

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


class SinusoidalPositions(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.size(1)].unsqueeze(0)


class TransformerMT(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, config: dict):
        super().__init__()
        d = config["d_model"]
        self.config = config
        self.scale = math.sqrt(d)
        self.src_embed = nn.Embedding(src_vocab, d, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(tgt_vocab, d, padding_idx=PAD_ID)
        nn.init.normal_(self.src_embed.weight, std=d ** -0.5)
        nn.init.normal_(self.tgt_embed.weight, std=d ** -0.5)
        self.positions = SinusoidalPositions(d, config.get("max_positions", 512))
        self.dropout = nn.Dropout(config["dropout"])
        self.transformer = nn.Transformer(
            d_model=d, nhead=config["heads"],
            num_encoder_layers=config["encoder_layers"],
            num_decoder_layers=config["decoder_layers"],
            dim_feedforward=config["ffn_dim"], dropout=config["dropout"],
            batch_first=True, norm_first=True)
        self.output = nn.Linear(d, tgt_vocab, bias=False)
        self.output.weight = self.tgt_embed.weight  # tied output embedding

    def encode(self, src):
        x = self.dropout(self.positions(self.src_embed(src) * self.scale))
        return self.transformer.encoder(x, src_key_padding_mask=src.eq(PAD_ID))

    def decode(self, memory, src, tgt_in):
        steps = tgt_in.size(1)
        causal = torch.triu(torch.full((steps, steps), float("-inf")), diagonal=1)
        y = self.dropout(self.positions(self.tgt_embed(tgt_in) * self.scale))
        out = self.transformer.decoder(
            y, memory, tgt_mask=causal, tgt_is_causal=True,
            tgt_key_padding_mask=tgt_in.eq(PAD_ID),
            memory_key_padding_mask=src.eq(PAD_ID))
        return self.output(out)

    def forward(self, src, tgt_in):
        return self.decode(self.encode(src), src, tgt_in)

    @staticmethod
    def _length_cap(src, max_len):
        # Output may be at most twice the source length (+10): plenty for
        # en->hi/mr, and an untrained/looping model cannot run to max_len.
        return min(max_len, int(src.ne(PAD_ID).sum(1).max()) * 2 + 10)

    @torch.no_grad()
    def greedy(self, src, max_len: int = 128) -> List[List[int]]:
        max_len = self._length_cap(src, max_len)
        memory = self.encode(src)
        ys = torch.full((src.size(0), 1), BOS_ID, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        for _ in range(max_len):
            next_id = self.decode(memory, src, ys)[:, -1].argmax(-1)
            next_id = next_id.masked_fill(done, PAD_ID)
            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
            done |= next_id.eq(EOS_ID)
            if done.all():
                break
        return [_strip(row) for row in ys[:, 1:].tolist()]

    @torch.no_grad()
    def beam(self, src_row, beam_size: int = 4, max_len: int = 128, alpha: float = 0.6) -> List[int]:
        """Beam search for ONE sentence (src_row: [1, S])."""
        max_len = self._length_cap(src_row, max_len)
        memory = self.encode(src_row)
        beams = [([BOS_ID], 0.0)]
        finished = []
        for _ in range(max_len):
            ys = torch.tensor([b[0] for b in beams])
            logp = torch.log_softmax(self.decode(memory.expand(len(beams), -1, -1),
                                                 src_row.expand(len(beams), -1), ys)[:, -1], -1)
            candidates = []
            for i, (seq, score) in enumerate(beams):
                top = logp[i].topk(beam_size)
                for lp, tok in zip(top.values.tolist(), top.indices.tolist()):
                    candidates.append((seq + [tok], score + lp))
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in candidates:
                if seq[-1] == EOS_ID:
                    finished.append((seq, score / (((5 + len(seq) - 1) / 6) ** alpha)))
                else:
                    beams.append((seq, score))
                if len(beams) == beam_size:
                    break
            if len(finished) >= beam_size or not beams:
                break
        if not finished:
            finished = [(seq, score / (((5 + len(seq) - 1) / 6) ** alpha)) for seq, score in beams]
        return _strip(max(finished, key=lambda f: f[1])[0][1:])


def _strip(ids: List[int]) -> List[int]:
    out = []
    for i in ids:
        if i in (EOS_ID, PAD_ID):
            break
        out.append(i)
    return out


class Translator:
    """Loads models/mt_en_<lang>/ (config.json, model.pt, spm.en.model, spm.<lang>.model)."""

    def __init__(self, model_dir: str):
        import sentencepiece as spm
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.info = json.load(f)
        lang = self.info["target_language"]
        self.sp_src = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "spm.en.model"))
        self.sp_tgt = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, f"spm.{lang}.model"))
        self.model = TransformerMT(self.sp_src.get_piece_size(), self.sp_tgt.get_piece_size(),
                                   self.info["model"])
        state = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.max_src = self.info["model"].get("max_source_tokens", 128)

    def _src_tensor(self, sentences):
        ids = [self.sp_src.encode(s)[: self.max_src - 1] + [EOS_ID] for s in sentences]
        width = max(len(i) for i in ids)
        return torch.tensor([i + [PAD_ID] * (width - len(i)) for i in ids])

    def translate(self, sentences: List[str], beam_size: int = 4, batch_size: int = 32) -> List[str]:
        results = []
        if beam_size <= 1:
            for k in range(0, len(sentences), batch_size):
                for ids in self.model.greedy(self._src_tensor(sentences[k:k + batch_size])):
                    results.append(self.sp_tgt.decode(ids))
            return results
        for s in sentences:
            results.append(self.sp_tgt.decode(self.model.beam(self._src_tensor([s]), beam_size)))
        return results
