"""
IndicTrans2 reference translations + COMET, to be run in GOOGLE COLAB
(this machine cannot run IndicTrans2 under transformers 5 nor install COMET).

What it does
  1. Translates OUR exact MT test sources (English) into Hindi and Marathi with
     ai4bharat/indictrans2-en-indic-dist-200M, using the OFFICIAL
     IndicTransToolkit pre/post-processing, beam 5, max_length 256.
  2. Scores IndicTrans2 with the SAME sacrebleu version and settings as our
     models (BLEU default 13a; chrF; chrF++ word_order=2) and records signatures.
  3. Computes COMET (Unbabel/wmt22-comet-da) for OUR hypotheses and for
     IndicTrans2's, on the same (source, hypothesis, reference) triples.
  4. Writes indictrans2_comet_results.json (+ IndicTrans2 hypotheses).

Inputs - upload these files from the project (created by mt/evaluate_mt.py):
    mt_results/hi/sources.txt  references.txt  hypotheses.txt
    mt_results/mr/sources.txt  references.txt  hypotheses.txt
  (i.e. zip the folders mt/results/hi and mt/results/mr as mt_results/...)

Colab steps
  Runtime -> Change runtime type -> GPU (optional, faster; CPU also works)
  !pip -q install "transformers>=4.40,<5" sentencepiece sacrebleu==2.6.0 unbabel-comet IndicTransToolkit
  from huggingface_hub import login; login()      # account with IndicTrans2 access
  !python colab_indictrans2_comet.py

Bring back: indictrans2_comet_results.json  ->  put it in the project at
  mt/results/external/indictrans2_comet_results.json
then run:  python results/build_report.py

Comparability: same test sentences, same references, same scorer. IndicTrans2 is
a large pretrained model trained on BPCC, which includes Samanantar (the source
of our test sentences) - it is a REFERENCE, not a like-for-like competitor.
"""

import json
import os
import platform
import time

import sacrebleu
import torch
import transformers
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"
COMET_MODEL = "Unbabel/wmt22-comet-da"
TAGS = {"hi": "hin_Deva", "mr": "mar_Deva"}
IN_DIR, OUT = "mt_results", "indictrans2_comet_results.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def read(lang, name):
    with open(os.path.join(IN_DIR, lang, name), encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def translate(sentences, lang, tokenizer, model, processor, batch_size=16):
    out = []
    for k in range(0, len(sentences), batch_size):
        batch = processor.preprocess_batch(sentences[k:k + batch_size], src_lang="eng_Latn", tgt_lang=TAGS[lang])
        enc = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                        return_attention_mask=True).to(DEVICE)
        with torch.no_grad():
            gen = model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=5,
                                 num_return_sequences=1)
        decoded = tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        out += processor.postprocess_batch(decoded, lang=TAGS[lang])
    return out


def scores(hyps, refs):
    bleu, chrf, chrfpp = sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF(), sacrebleu.metrics.CHRF(word_order=2)
    return {"BLEU": round(bleu.corpus_score(hyps, [refs]).score, 2),
            "chrF": round(chrf.corpus_score(hyps, [refs]).score, 2),
            "chrF++": round(chrfpp.corpus_score(hyps, [refs]).score, 2),
            "signatures": {"BLEU": str(bleu.get_signature()), "chrF": str(chrf.get_signature()),
                           "chrF++": str(chrfpp.get_signature())}}


def main():
    from comet import download_model, load_from_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(DEVICE).eval()
    processor = IndicProcessor(inference=True)
    comet = load_from_checkpoint(download_model(COMET_MODEL))
    results = {"environment": {"python": platform.python_version(), "transformers": transformers.__version__,
                               "sacrebleu": sacrebleu.__version__, "torch": torch.__version__, "device": DEVICE,
                               "indictrans2_model": MODEL, "comet_model": COMET_MODEL,
                               "decoding": "beam 5, max_length 256, official IndicProcessor pre/post-processing"}}
    for lang in ("hi", "mr"):
        src, ref, ours = read(lang, "sources.txt"), read(lang, "references.txt"), read(lang, "hypotheses.txt")
        assert len(src) == len(ref) == len(ours), f"{lang}: file lengths differ"
        t0 = time.time()
        it2 = translate(src, lang, tokenizer, model, processor)
        seconds = time.time() - t0
        with open(f"indictrans2_{lang}_hypotheses.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(it2) + "\n")
        def comet_score(hyps):
            data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyps, ref)]
            return round(comet.predict(data, batch_size=16, gpus=1 if DEVICE == "cuda" else 0).system_score, 4)
        results[lang] = {"test_pairs": len(src),
                         "indictrans2": {**scores(it2, ref), "COMET": comet_score(it2), "translate_seconds": round(seconds, 1)},
                         "ours_rescored": {**scores(ours, ref), "COMET": comet_score(ours)}}
        print(lang, json.dumps(results[lang], ensure_ascii=False))
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
