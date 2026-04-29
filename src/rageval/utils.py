"""Shared retrieval and evaluation helpers.

Used by every script under ``scripts/`` so that the metric definitions,
the four retrieval implementations (BM25, dense, SPLADE, MedCPT) and the
common utilities (cuda_gc, paths) live in one place.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Project paths.  Everything is anchored at the repo root regardless of where
# a script is invoked from.
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "notebooks" / "results"
FEEDBACK2_DIR = RESULTS_DIR / "feedback2" / "tables"
NEXT_STAGE_DIR = RESULTS_DIR / "next_stage" / "tables"
DATASETS_DIR = ROOT / "notebooks" / "datasets"
CACHE_DIR = RESULTS_DIR / "feedback2" / "cache"

for d in (FEEDBACK2_DIR, NEXT_STAGE_DIR, DATASETS_DIR, CACHE_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Lazy CUDA helpers.  Avoid the import unless torch is actually available;
# the analysis scripts (regression, bootstrap) do not need GPU.
# ---------------------------------------------------------------------------


def cuda_gc() -> None:
    """Free Python references and reclaim the CUDA cache fragment."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except ImportError:
        pass


def get_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_ndcg(ranked_docs: List[str], qrel: Dict[str, int], k: int) -> float:
    dcg = 0.0
    for i, did in enumerate(ranked_docs[:k]):
        rel = qrel.get(did, 0)
        dcg += (2 ** rel - 1) / np.log2(i + 2)
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def compute_recall(ranked_docs: List[str], qrel: Dict[str, int], k: int,
                    threshold: int = 1) -> float:
    relevant = {d for d, r in qrel.items() if r >= threshold}
    if not relevant:
        return 0.0
    found = sum(1 for d in ranked_docs[:k] if d in relevant)
    return found / len(relevant)


def compute_precision(ranked_docs: List[str], qrel: Dict[str, int], k: int,
                       threshold: int = 1) -> float:
    if k == 0:
        return 0.0
    relevant = {d for d, r in qrel.items() if r >= threshold}
    found = sum(1 for d in ranked_docs[:k] if d in relevant)
    return found / k


def compute_map(ranked_docs: List[str], qrel: Dict[str, int], k: int,
                 threshold: int = 1) -> float:
    relevant = {d for d, r in qrel.items() if r >= threshold}
    if not relevant:
        return 0.0
    score = 0.0
    hits = 0
    for i, did in enumerate(ranked_docs[:k]):
        if did in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(relevant), k)


def evaluate_retriever(results: Dict[str, Dict[str, float]],
                        qrels: Dict[str, Dict[str, int]],
                        k_values: Tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """Aggregate per-query nDCG@k, Recall@k, P@k, MAP@k.

    Returns a dict with keys ``aggregate`` (mean over queries),
    ``per_query`` (per-qid metric dict) and ``loss_array`` (1 - nDCG@10).
    """
    per_query: Dict[str, Dict[str, float]] = {}
    for qid in qrels:
        if qid not in results:
            continue
        ranking = [d for d, _ in sorted(results[qid].items(), key=lambda x: -x[1])]
        m: Dict[str, float] = {}
        for k in k_values:
            m[f"nDCG@{k}"] = compute_ndcg(ranking, qrels[qid], k)
            m[f"Recall@{k}"] = compute_recall(ranking, qrels[qid], k)
            m[f"P@{k}"] = compute_precision(ranking, qrels[qid], k)
            m[f"MAP@{k}"] = compute_map(ranking, qrels[qid], k)
        per_query[qid] = m
    if not per_query:
        return {"aggregate": {}, "per_query": {}, "loss_array": np.array([])}
    keys = list(next(iter(per_query.values())).keys())
    agg = {k: float(np.mean([m[k] for m in per_query.values()])) for k in keys}
    loss = np.array([1.0 - m["nDCG@10"] for m in per_query.values()])
    return {"aggregate": agg, "per_query": per_query, "loss_array": loss}


# ---------------------------------------------------------------------------
# Corpus and retrieval helpers.  Each ``_*_run`` function returns a
# dict-of-dicts in the BEIR format (qid -> {doc_id -> score}).
# ---------------------------------------------------------------------------


def prep_corpus(corpus: Dict[str, Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """Return (doc_ids, doc_texts) where text = title + " " + body, stripped."""
    ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
             for d in ids]
    return ids, texts


def bm25_run(doc_ids: List[str], doc_texts: List[str],
              queries: Dict[str, str], top_k: int = 100) -> Dict[str, Dict[str, float]]:
    """Single-threaded BM25 retrieval using the bm25s package (avoids the
    multiprocessing bug on Colab kernels)."""
    import bm25s

    if not queries:
        return {}
    corpus_tok = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
    bm = bm25s.BM25(k1=1.5, b=0.75)
    bm.index(corpus_tok)

    qids = list(queries.keys())
    qtok = bm25s.tokenize([queries[q] for q in qids], stopwords="en",
                           show_progress=False)
    k = min(top_k, len(doc_ids))
    try:
        res, sc = bm.retrieve(qtok, k=k, n_threads=1, show_progress=False)
    except TypeError:
        res, sc = bm.retrieve(qtok, k=k)

    return {
        qid: {doc_ids[res[i][j]]: float(sc[i][j]) for j in range(k)}
        for i, qid in enumerate(qids)
    }


def dense_run(doc_ids: List[str], doc_texts: List[str],
               queries: Dict[str, str], model,
               qpfx: str = "", dpfx: str = "",
               top_k: int = 100, batch: int = 16) -> Dict[str, Dict[str, float]]:
    """Bi-encoder dense retrieval. Pass any sentence-transformers model."""
    if not doc_ids or not queries:
        return {qid: {} for qid in queries}
    corpus_texts = [dpfx + t for t in doc_texts] if dpfx else doc_texts
    doc_emb = model.encode(corpus_texts, batch_size=batch,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")
    qids = list(queries.keys())
    qtxt = [(qpfx + queries[q]) if qpfx else queries[q] for q in qids]
    q_emb = model.encode(qtxt, batch_size=32,
                          normalize_embeddings=True,
                          convert_to_numpy=True).astype("float32")
    if doc_emb.ndim != 2 or q_emb.ndim != 2:
        raise RuntimeError(
            f"Dense encoder produced non-2D embeddings: doc={doc_emb.shape}, q={q_emb.shape}")
    scores = q_emb @ doc_emb.T
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    out: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(scores[i], -k)[-k:]
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_emb, q_emb, scores
    cuda_gc()
    return out


def splade_run(doc_ids: List[str], doc_texts: List[str],
                queries: Dict[str, str], tok, mod,
                top_k: int = 100, bs: int = 4,
                max_len: int = 128) -> Dict[str, Dict[str, float]]:
    """SPLADE: log-ReLU on MLM logits, then sparse matmul."""
    import torch

    device = get_device()
    if not doc_ids or not queries:
        return {qid: {} for qid in queries}

    def _enc(texts: Iterable[str]):
        rows = []
        texts = list(texts)
        for i in range(0, len(texts), bs):
            inp = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                vecs = torch.log(1 + torch.relu(mod(**inp).logits))
            vecs = (vecs * inp["attention_mask"].unsqueeze(-1)).max(dim=1).values
            rows.append(sp.csr_matrix(vecs.cpu().numpy()))
        return sp.vstack(rows)

    doc_vecs = _enc(doc_texts)
    qids = list(queries.keys())
    q_vecs = _enc([queries[q] for q in qids])
    scores = (q_vecs @ doc_vecs.T).toarray()

    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    out: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        if k >= n_docs:
            top = np.argsort(-scores[i])
        else:
            top = np.argpartition(-scores[i], k)[:k]
            top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_vecs, q_vecs, scores
    cuda_gc()
    return out


def medcpt_run(doc_ids: List[str], doc_texts: List[str],
                queries: Dict[str, str],
                qry_tok, qry_mod, art_tok, art_mod,
                top_k: int = 100, bs: int = 8) -> Dict[str, Dict[str, float]]:
    """Dual-encoder MedCPT retrieval (separate query and article encoders)."""
    import torch
    import torch.nn.functional as F

    device = get_device()
    if not doc_ids or not queries:
        return {qid: {} for qid in queries}

    def _enc(texts, tok, mod, max_len):
        embs = []
        texts = list(texts)
        for i in range(0, len(texts), bs):
            inp = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                       truncation=True, max_length=max_len)
            inp = {k: v.to(device) for k, v in inp.items()}
            with torch.no_grad():
                e = F.normalize(mod(**inp).last_hidden_state[:, 0, :], dim=-1)
            embs.append(e.cpu().numpy())
        return np.concatenate(embs, axis=0).astype("float32")

    doc_emb = _enc(doc_texts, art_tok, art_mod, 192)
    qids = list(queries.keys())
    q_emb = _enc([queries[q] for q in qids], qry_tok, qry_mod, 64)

    scores = q_emb @ doc_emb.T
    n_docs = len(doc_ids)
    k = min(top_k, n_docs)
    out: Dict[str, Dict[str, float]] = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(scores[i], -k)[-k:]
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_emb, q_emb, scores
    cuda_gc()
    return out


# ---------------------------------------------------------------------------
# Hybrid score fusion (used by the router and Section 4.5 of the report)
# ---------------------------------------------------------------------------


def _minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def hybrid_run(bm25_res: Dict[str, Dict[str, float]],
                dense_res: Dict[str, Dict[str, float]],
                queries: Dict[str, str],
                alpha: float = 0.40) -> Dict[str, Dict[str, float]]:
    """Linear score fusion of BM25 and any dense retriever after per-method
    min-max normalisation. ``alpha`` is the BM25 weight.
    """
    out: Dict[str, Dict[str, float]] = {}
    for qid in queries:
        sb = bm25_res.get(qid, {})
        sd = dense_res.get(qid, {})
        cand = sorted(set(sb) | set(sd))
        if not cand:
            out[qid] = {}
            continue
        sb_arr = np.array([sb.get(c, 0.0) for c in cand])
        sd_arr = np.array([sd.get(c, 0.0) for c in cand])
        hyb = alpha * _minmax(sb_arr) + (1 - alpha) * _minmax(sd_arr)
        out[qid] = {c: float(s) for c, s in zip(cand, hyb)}
    return out


# ---------------------------------------------------------------------------
# Bootstrap CIs
# ---------------------------------------------------------------------------


def cross_encoder_rerank(queries: Dict[str, str],
                           corpus: Dict[str, Dict[str, str]],
                           candidate_ids_per_query: Dict[str, List[str]],
                           model, batch_size: int = 4
                           ) -> Dict[str, Dict[str, float]]:
    """Rerank top-k candidates per query with a sentence-transformers
    CrossEncoder.  Returns dict-of-dicts in the same shape as the
    first-stage retrievers."""
    out: Dict[str, Dict[str, float]] = {}
    for qid in queries:
        cand_ids = candidate_ids_per_query.get(qid, [])
        if not cand_ids:
            out[qid] = {}
            continue
        cand_texts = [(corpus[d].get("title", "") + " "
                        + corpus[d].get("text", "")).strip() for d in cand_ids]
        pairs = [(queries[qid], t) for t in cand_texts]
        scores = model.predict(pairs, batch_size=batch_size,
                                show_progress_bar=False, convert_to_numpy=True)
        order = np.argsort(-np.asarray(scores, dtype=float))
        out[qid] = {cand_ids[i]: float(scores[i]) for i in order}
    return out


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, B: int = 10_000,
                          alpha: float = 0.05, seed: int = 42) -> dict:
    """Paired bootstrap CI for the mean difference a - b.

    Returns dict with keys ``mean_diff``, ``ci_lo``, ``ci_hi`` and
    ``p_a_gt_b`` (share of resamples where a's mean > b's).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    diffs = a - b
    rng = np.random.default_rng(seed)
    n = len(diffs)
    boot = np.array([diffs[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p_a_gt_b": float((boot > 0).mean()),
    }
