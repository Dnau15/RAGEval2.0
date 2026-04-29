"""Efficiency measurements for Table 10.

Indexing time, per-query latency, throughput, and index memory for each
of the six retrievers on the 3633-document NFCorpus corpus.

Notes
-----
- All measurements are single-threaded CPU.  GPU is used only for the
  initial encoding pass of dense methods.
- Latency depends on the host machine; the numbers in the report were
  measured on a Colab T4 + Intel Xeon CPU and may differ on other hardware.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_beir
from rageval.models import (get_bge, get_e5, get_medcpt, get_minilm,
                              get_splade, unload)
from rageval.utils import (FEEDBACK2_DIR, bm25_run, dense_run, get_device,
                            medcpt_run, prep_corpus, splade_run)


def _time_run(fn, repeats: int = 3) -> float:
    """Median wall-clock time of ``fn()`` over ``repeats`` calls."""
    times = []
    for _ in range(repeats):
        t0 = time.time()
        fn()
        times.append(time.time() - t0)
    return float(np.median(times))


def _memsize_dense(n_docs: int, dim: int) -> float:
    """In-memory size of an n_docs x dim float32 matrix in MB."""
    return n_docs * dim * 4 / 1024 / 1024


def main() -> None:
    corpus, queries, _ = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    n_docs, n_q = len(doc_ids), len(queries)
    print(f"Corpus: {n_docs:,} docs; queries: {n_q}")

    device = get_device()
    rows = []

    # ---- BM25 ------------------------------------------------------------
    print("\n[BM25]")
    t0 = time.time()
    bm25_run(doc_ids, doc_texts, {list(queries.keys())[0]: list(queries.values())[0]},
              top_k=100)
    t_index = time.time() - t0
    # latency over 50 queries
    sub = dict(list(queries.items())[:50])
    elapsed = _time_run(lambda: bm25_run(doc_ids, doc_texts, sub, top_k=100))
    latency_ms = elapsed * 1000 / len(sub)
    rows.append({
        "Retriever": "BM25",
        "Index build (s)": round(t_index, 1),
        "Latency (ms/q)": round(latency_ms, 1),
        "Throughput (q/s)": round(1000 / latency_ms, 0),
        "Index memory": "inverted list (sparse)",
    })

    # ---- Dense (MiniLM, BGE, E5) ----------------------------------------
    for name, model_fn, qpfx, dpfx, dim in [
        ("Dense (MiniLM)", get_minilm, "", "", 384),
        ("BGE-small", get_bge, "", "", 384),
        ("E5-small", get_e5, "query: ", "passage: ", 384),
    ]:
        print(f"\n[{name}]")
        t0 = time.time()
        model = model_fn()
        _ = model.encode(doc_texts, batch_size=16, show_progress_bar=False,
                          normalize_embeddings=True, convert_to_numpy=True)
        t_index = time.time() - t0

        sub = dict(list(queries.items())[:50])
        elapsed = _time_run(
            lambda: dense_run(doc_ids, doc_texts, sub, model_fn(),
                               qpfx=qpfx, dpfx=dpfx, top_k=100))
        latency_ms = elapsed * 1000 / len(sub)
        rows.append({
            "Retriever": name,
            "Index build (s)": round(t_index, 1),
            "Latency (ms/q)": round(latency_ms, 1),
            "Throughput (q/s)": round(1000 / latency_ms, 0),
            "Index memory": f"{_memsize_dense(n_docs, dim):.0f} MB ({n_docs} x {dim} f32)",
        })

    unload("minilm", "bge", "e5")

    # ---- SPLADE ----------------------------------------------------------
    print("\n[SPLADE]")
    stok, smod = get_splade()
    t0 = time.time()
    splade_run(doc_ids, doc_texts, {list(queries.keys())[0]:
                                      list(queries.values())[0]}, stok, smod,
                top_k=100)
    t_index = time.time() - t0
    sub = dict(list(queries.items())[:50])
    elapsed = _time_run(lambda: splade_run(doc_ids, doc_texts, sub,
                                              stok, smod, top_k=100))
    latency_ms = elapsed * 1000 / len(sub)
    rows.append({
        "Retriever": "SPLADE",
        "Index build (s)": round(t_index, 1),
        "Latency (ms/q)": round(latency_ms, 1),
        "Throughput (q/s)": round(1000 / latency_ms, 0),
        "Index memory": "sparse CSR, ~151 nnz/doc",
    })
    unload("splade")

    # ---- MedCPT ----------------------------------------------------------
    print("\n[MedCPT]")
    qt, qm, at, am = get_medcpt()
    t0 = time.time()
    medcpt_run(doc_ids, doc_texts,
                {list(queries.keys())[0]: list(queries.values())[0]},
                qt, qm, at, am, top_k=100)
    t_index = time.time() - t0
    sub = dict(list(queries.items())[:50])
    elapsed = _time_run(lambda: medcpt_run(doc_ids, doc_texts, sub,
                                              qt, qm, at, am, top_k=100))
    latency_ms = elapsed * 1000 / len(sub)
    rows.append({
        "Retriever": "MedCPT",
        "Index build (s)": round(t_index, 1),
        "Latency (ms/q)": round(latency_ms, 1),
        "Throughput (q/s)": round(1000 / latency_ms, 0),
        "Index memory": f"{_memsize_dense(n_docs, 768):.0f} MB ({n_docs} x 768 f32)",
    })
    unload("medcpt")

    df = pd.DataFrame(rows)
    out = FEEDBACK2_DIR / "efficiency.csv"
    df.to_csv(out, index=False)
    print("\n", df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
