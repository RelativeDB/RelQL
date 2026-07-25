"""Semantic article -> event links, in the model's own embedding space.

Token overlap misses everything a paraphrase hides: "Maduro" never appears in
"Will the Venezuelan government fall by December 31?", and "Fed holds rates
steady" shares no distinctive word with "Will there be a rate cut in
December?". Cosine similarity between MiniLM embeddings catches both.

Two things make this legitimate rather than a leak:

  * The encoder is pretrained and frozen. It reads one headline at a time and
    knows nothing about markets, outcomes or dates.
  * A link inherits the *article's* publication time, so the engine's temporal
    bound drops it for any anchor earlier than the headline. Similarity
    decides *whether* two things are about the same subject; the timestamp
    still decides *when* you were allowed to know it.

Embeddings are cached on disk keyed by text hash — the expensive part is paid
once, not once per experiment.

    python -m scale.semantic --top-k 40 --min-similarity 0.42
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCALE = Path(__file__).resolve().parent.parent / "data" / "scale"
CACHE = SCALE / "embeddings"


def _key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def embed(texts: list[str], *, batch: int = 512, tag: str = "texts"):
    """Embed with the same MiniLM RT-J uses, caching by text hash."""
    CACHE.mkdir(parents=True, exist_ok=True)
    store = CACHE / f"{tag}.npz"
    cached: dict[str, np.ndarray] = {}
    if store.exists():
        with np.load(store) as z:
            cached = {k: z[k] for k in z.files}
    todo = [t for t in dict.fromkeys(texts) if _key(t) not in cached]
    if todo:
        from relativedb.rt_native import TextEmbedder
        encoder = TextEmbedder()
        encoder._load()
        print(f"   embedding {len(todo):,} new {tag} "
              f"({len(cached):,} cached)", flush=True)
        for i in range(0, len(todo), batch):
            chunk = todo[i:i + batch]
            vectors = encoder._model.encode(chunk, batch_size=64,
                                            show_progress_bar=False,
                                            convert_to_numpy=True)
            for text, vector in zip(chunk, vectors):
                cached[_key(text)] = vector.astype(np.float32)
            if (i // batch) % 20 == 0:
                print(f"     {min(i + batch, len(todo)):>7,}/{len(todo):,}",
                      flush=True)
                # Checkpoint. A long encode that dies at 40% and saves nothing
                # is the same as never having run it.
                np.savez(store, **cached)
        np.savez_compressed(store, **cached)
    matrix = np.stack([cached[_key(t)] for t in texts])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def link(events: dict[str, str], articles: list[dict], *, top_k: int,
         min_similarity: float, candidates_per_event: int = 4000):
    """Top-k most similar headlines per event, above a similarity floor."""
    event_ids = list(events)
    event_vectors = embed([events[e] for e in event_ids], tag="events")
    headlines = [a["headline"] for a in articles]
    article_vectors = embed(headlines, tag="articles")

    mentions = []
    # Chunked matmul: 1k events x 200k articles at once would be 800 MB of
    # float32 similarity, and nothing needs all of it in memory.
    for start in range(0, len(event_ids), 64):
        block = event_vectors[start:start + 64]
        sims = block @ article_vectors.T
        for row, event_id in enumerate(event_ids[start:start + 64]):
            scores = sims[row]
            if top_k < len(scores):
                top = np.argpartition(-scores, top_k)[:top_k]
            else:
                top = np.arange(len(scores))
            for j in top:
                if scores[j] < min_similarity:
                    continue
                mentions.append((event_id, articles[j]["article_id"],
                                 float(scores[j])))
    return mentions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--min-similarity", type=float, default=0.42)
    ap.add_argument("--prefilter-overlap", type=int, default=1,
                    help="shared distinctive words needed to be a candidate")
    ap.add_argument("--max-candidates", type=int, default=400_000)
    args = ap.parse_args()
    out_name = 'semantic_mentions.parquet'
    if args.wide:
        from scale.build import WIDE, use_sources
        use_sources(**WIDE)
        out_name = WIDE['semantic']

    from scale.build import build_events, candidate_articles
    events = build_events()
    articles = candidate_articles(events, min_overlap=args.prefilter_overlap,
                                  limit=args.max_candidates)
    print(f">> {len(events):,} events, {len(articles):,} candidate articles")
    mentions = link(events, articles, top_k=args.top_k,
                    min_similarity=args.min_similarity)
    by_article = {a["article_id"]: a for a in articles}
    table = pa.table({
        "event_id": [m[0] for m in mentions],
        "article_id": [m[1] for m in mentions],
        "similarity": [m[2] for m in mentions],
        "published_at": [by_article[m[1]]["published_at"] for m in mentions],
    })
    pq.write_table(table, SCALE / out_name)
    print(f">> wrote {table.num_rows:,} semantic mentions across "
          f"{len(set(t for t in table['event_id'].to_pylist())):,} events")


if __name__ == "__main__":
    main()
