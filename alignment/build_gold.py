"""
Build the gold evaluation sets for English -> Hindi / Marathi entity alignment.

    python alignment/build_gold.py

Source sentences: the held-out MT splits mt/data/<lang>/{validation,test}.jsonl
(never used to train the MT models; Indic sides that occur in the NER data
were removed when the MT data was built). validation = tuning, test = final.

Gold construction (independent of the aligner and of the target-side NER):
  1. English entity mentions = the project's English NER model
     (models/english_ner_scratch, no aliases) on the English sentence.
  2. Each mention is resolved with the English Wikipedia API: it is accepted
     only if it is the title of, or a redirect to, an English Wikipedia
     article (not a disambiguation page). The article's Wikidata item is used.
  3. The item's official Hindi/Marathi labels + aliases (Wikidata) are
     searched in the target sentence. A match must start at a word start;
     an attached Devanagari suffix is allowed (e.g. दिल्ली in दिल्लीत) and is
     excluded from the gold span. Every matching occurrence is a gold span.
  4. Mentions with at least one gold span become evaluation items; all
     others are counted (not an English Wikipedia article / no Hindi-Marathi
     label / label not present in the target sentence) and reported.

Lookups are batched (50 titles or ids per request, ~1 request/second) and
every response is cached in alignment/data/wikidata_cache.json, so the gold
sets can be rebuilt exactly without network access.

Output: alignment/data/<lang>/{validation,test}.jsonl + gold_info.json
"""

import json
import os
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT)
from trained_ner import load_predictor  # noqa: E402

DATA = os.path.join(HERE, "data")
CACHE_PATH = os.path.join(DATA, "wikidata_cache.json")
WIKIPEDIA = "https://en.wikipedia.org/w/api.php"
WIKIDATA = "https://www.wikidata.org/w/api.php"
UA = {"User-Agent": "IndicEntityLab-research/1.0 (academic NER/alignment project)"}
SPLITS = ("validation", "test")


class KB:
    def __init__(self):
        self.cache = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                self.cache = json.load(f)
        self.requests = 0

    def save(self):
        os.makedirs(DATA, exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False)

    def get(self, api, params):
        key = api + " " + json.dumps(params, sort_keys=True, ensure_ascii=False)
        if key in self.cache:
            return self.cache[key]
        data = urllib.parse.urlencode({**params, "format": "json", "formatversion": 2}).encode()
        last = None
        for attempt in range(8):
            try:
                req = urllib.request.Request(api, data=data, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as r:
                    self.cache[key] = json.load(r)
                break
            except urllib.error.HTTPError as exc:
                last, wait = exc, int(exc.headers.get("Retry-After") or 0) or 10 * (attempt + 1)
            except Exception as exc:  # noqa: BLE001 (connection resets, timeouts)
                last, wait = exc, 10 * (attempt + 1)
            print(f"  retry {attempt + 1}/8 in {wait}s ({type(last).__name__}: {last})", flush=True)
            time.sleep(wait)
        else:
            self.save()
            raise RuntimeError(f"request failed: {params} ({last!r})")
        self.requests += 1
        self.save()
        time.sleep(1.0)  # about 1 request per second
        return self.cache[key]

    def resolve_titles(self, mentions):
        """mention -> Wikidata QID (article title or redirect; no disambiguation pages)."""
        result = {}
        mentions = sorted(set(mentions))
        for k in range(0, len(mentions), 50):
            batch = mentions[k:k + 50]
            q = self.get(WIKIPEDIA, {"action": "query", "titles": "|".join(batch), "redirects": 1,
                                     "prop": "pageprops", "ppprop": "wikibase_item|disambiguation"})["query"]
            follow = {}
            for n in q.get("normalized", []):
                follow[n["from"]] = n["to"]
            redirects = {r["from"]: r["to"] for r in q.get("redirects", [])}
            pages = {p["title"]: p for p in q.get("pages", [])}
            for m in batch:
                title = follow.get(m, m)
                title = redirects.get(title, title)
                page = pages.get(title, {})
                props = page.get("pageprops", {})
                if not page.get("missing") and "disambiguation" not in props and props.get("wikibase_item"):
                    result[m] = props["wikibase_item"]
        return result

    def names(self, qids):
        """QID -> {"hi": [...], "mr": [...]} official labels + aliases."""
        out = {}
        qids = sorted(set(qids))
        for k in range(0, len(qids), 50):
            ents = self.get(WIKIDATA, {"action": "wbgetentities", "ids": "|".join(qids[k:k + 50]),
                                       "props": "labels|aliases", "languages": "hi|mr"})["entities"]
            for qid, e in ents.items():
                out[qid] = {}
                for lang in ("hi", "mr"):
                    found = []
                    label = e.get("labels", {}).get(lang, {}).get("value")
                    if label:
                        found.append(label)
                    found += [a["value"] for a in e.get("aliases", {}).get(lang, [])]
                    out[qid][lang] = sorted(set(n for n in found if n.strip()), key=len, reverse=True)
        return out


def is_mark(ch):
    return unicodedata.category(ch).startswith("M") or ch in "‌‍"


def joins(ch):
    return ch.isalnum() or is_mark(ch)


def find_spans(sentence, name):
    """Occurrences of `name` starting at a word start; a following Devanagari
    suffix letter is allowed, a following vowel sign/virama is not."""
    spans, start = [], 0
    while True:
        i = sentence.find(name, start)
        if i == -1:
            return spans
        j = i + len(name)
        if (i == 0 or not joins(sentence[i - 1])) and (j == len(sentence) or not is_mark(sentence[j])):
            spans.append([i, j])
        start = i + 1


def main():
    ner = load_predictor(os.path.join(PROJECT, "models", "english_ner_scratch"))
    pairs, mentions = {}, {}
    for lang in ("hi", "mr"):
        for split in SPLITS:
            with open(os.path.join(PROJECT, "mt", "data", lang, f"{split}.jsonl"), encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            pairs[(lang, split)] = rows
            mentions[(lang, split)] = [ner.predict(r["en"]) for r in rows]
    all_mentions = {e["word"] for ms in mentions.values() for m in ms for e in m}
    kb = KB()
    qid_of = kb.resolve_titles(all_mentions)
    names = kb.names(qid_of.values())
    print(f"unique English mentions {len(all_mentions)} | resolved to Wikipedia/Wikidata {len(qid_of)} "
          f"| requests made {kb.requests}", flush=True)

    info = {"method": __doc__.strip().split("\n\n")[2],
            "ner_model": "models/english_ner_scratch (no aliases)",
            "knowledge_base": "English Wikipedia titles/redirects -> Wikidata (CC0) Hindi/Marathi labels + aliases",
            "splits": {}}
    for (lang, split), rows in pairs.items():
        stats = {"pairs": len(rows), "english_mentions": 0, "not_a_wikipedia_article": 0,
                 f"no_{lang}_label": 0, "label_not_in_target": 0, "gold_items": 0,
                 "items_with_multiple_gold_spans": 0, "by_type": {}}
        items = []
        for p, ents in zip(rows, mentions[(lang, split)]):
            for ent in ents:
                stats["english_mentions"] += 1
                qid = qid_of.get(ent["word"])
                if not qid:
                    stats["not_a_wikipedia_article"] += 1
                    continue
                candidates = names.get(qid, {}).get(lang, [])
                if not candidates:
                    stats[f"no_{lang}_label"] += 1
                    continue
                gold, used = [], []
                for name in candidates:
                    for span in find_spans(p[lang], name):
                        if not any(s[0] < span[1] and span[0] < s[1] for s in gold):
                            gold.append(span)
                            used.append(name)
                if not gold:
                    stats["label_not_in_target"] += 1
                    continue
                stats["gold_items"] += 1
                stats["items_with_multiple_gold_spans"] += len(gold) > 1
                stats["by_type"][ent["entity_group"]] = stats["by_type"].get(ent["entity_group"], 0) + 1
                items.append({"pair_id": p["id"], "en": p["en"], "target": p[lang],
                              "entity": {"text": ent["word"], "type": ent["entity_group"],
                                         "start": ent["start"], "end": ent["end"]},
                              "wikidata": qid, "kb_names": candidates, "matched_names": used,
                              "gold_spans": sorted(gold)})
        folder = os.path.join(DATA, lang)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"{split}.jsonl"), "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        info["splits"][f"{lang}/{split}"] = stats
        print(f"[{lang}/{split}] {stats}", flush=True)
    with open(os.path.join(DATA, "gold_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"done; total requests {kb.requests}; cache {CACHE_PATH}")


if __name__ == "__main__":
    main()
