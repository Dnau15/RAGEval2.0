"""Retrieval methods, metrics and model loading.

The four retrieval implementations (BM25, dense bi-encoder, SPLADE,
MedCPT) all return BEIR-style ``{qid: {doc_id: score}}`` so they slot
into the same evaluation harness.
"""

from __future__ import annotations

import gc
import math
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------
# Paths.  Anchored at repo root so scripts work no matter where they're
# launched from.
# ----------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "notebooks" / "results"
FEEDBACK2 = RESULTS / "feedback2" / "tables"
NEXT_STAGE = RESULTS / "next_stage" / "tables"
DATASETS = ROOT / "notebooks" / "datasets"
CACHE = RESULTS / "feedback2" / "cache"

for d in (FEEDBACK2, NEXT_STAGE, DATASETS, CACHE):
    d.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Misc helpers
# ----------------------------------------------------------------------

def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def free_cuda():
    """Drop python refs and reclaim the CUDA cache fragment."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def ndcg(ranked, qrel, k):
    dcg = sum((2 ** qrel.get(d, 0) - 1) / math.log2(i + 2)
               for i, d in enumerate(ranked[:k]))
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall(ranked, qrel, k, threshold=1):
    rel = {d for d, r in qrel.items() if r >= threshold}
    if not rel:
        return 0.0
    return sum(1 for d in ranked[:k] if d in rel) / len(rel)


def precision(ranked, qrel, k, threshold=1):
    if k == 0:
        return 0.0
    rel = {d for d, r in qrel.items() if r >= threshold}
    return sum(1 for d in ranked[:k] if d in rel) / k


def average_precision(ranked, qrel, k, threshold=1):
    rel = {d for d, r in qrel.items() if r >= threshold}
    if not rel:
        return 0.0
    score = hits = 0
    for i, d in enumerate(ranked[:k]):
        if d in rel:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(rel), k)


def evaluate(results, qrels, k_values=(1, 3, 5, 10)):
    """Aggregate per-query metrics into an ``aggregate``/``per_query``
    dict plus a 1-nDCG@10 loss array."""
    per_q = {}
    for qid in qrels:
        if qid not in results:
            continue
        ranked = sorted(results[qid], key=results[qid].get, reverse=True)
        m = {}
        for k in k_values:
            m[f"nDCG@{k}"] = ndcg(ranked, qrels[qid], k)
            m[f"Recall@{k}"] = recall(ranked, qrels[qid], k)
            m[f"P@{k}"] = precision(ranked, qrels[qid], k)
            m[f"MAP@{k}"] = average_precision(ranked, qrels[qid], k)
        per_q[qid] = m
    if not per_q:
        return {"aggregate": {}, "per_query": {}, "loss": np.array([])}
    keys = next(iter(per_q.values())).keys()
    agg = {k: float(np.mean([m[k] for m in per_q.values()])) for k in keys}
    loss = np.array([1.0 - m["nDCG@10"] for m in per_q.values()])
    return {"aggregate": agg, "per_query": per_q, "loss": loss}


# ----------------------------------------------------------------------
# Corpus prep
# ----------------------------------------------------------------------

def prep_corpus(corpus):
    """Return parallel (ids, texts) lists.  Title and body are joined."""
    ids = list(corpus.keys())
    texts = [(corpus[d].get("title", "") + " " + corpus[d].get("text", "")).strip()
             for d in ids]
    return ids, texts


# ----------------------------------------------------------------------
# Retrieval methods
# ----------------------------------------------------------------------

def bm25(doc_ids, doc_texts, queries, top_k=100):
    """BM25 via the bm25s package, single-threaded to dodge a known
    multiprocessing bug on Colab kernels."""
    import bm25s
    if not queries:
        return {}

    corpus_tok = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
    bm = bm25s.BM25(k1=1.5, b=0.75)
    bm.index(corpus_tok)

    qids = list(queries)
    qtok = bm25s.tokenize([queries[q] for q in qids], stopwords="en",
                           show_progress=False)
    k = min(top_k, len(doc_ids))
    try:
        idx, scores = bm.retrieve(qtok, k=k, n_threads=1, show_progress=False)
    except TypeError:  # older bm25s without n_threads
        idx, scores = bm.retrieve(qtok, k=k)

    return {
        qid: {doc_ids[idx[i][j]]: float(scores[i][j]) for j in range(k)}
        for i, qid in enumerate(qids)
    }


def dense(doc_ids, doc_texts, queries, model, qpfx="", dpfx="",
          top_k=100, batch=16):
    """Bi-encoder retrieval with cosine similarity (inner product of
    L2-normalised embeddings)."""
    if not doc_ids or not queries:
        return {q: {} for q in queries}

    doc_texts_in = [dpfx + t for t in doc_texts] if dpfx else doc_texts
    doc_emb = model.encode(doc_texts_in, batch_size=batch,
                            show_progress_bar=False,
                            normalize_embeddings=True,
                            convert_to_numpy=True).astype("float32")

    qids = list(queries)
    q_in = [(qpfx + queries[q]) if qpfx else queries[q] for q in qids]
    q_emb = model.encode(q_in, batch_size=32,
                          normalize_embeddings=True,
                          convert_to_numpy=True).astype("float32")

    scores = q_emb @ doc_emb.T
    k = min(top_k, len(doc_ids))
    out = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(scores[i], -k)[-k:]
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}

    del doc_emb, q_emb, scores
    free_cuda()
    return out


def splade(doc_ids, doc_texts, queries, tok, model,
           top_k=100, batch=4, max_len=128):
    """SPLADE: log-ReLU on MLM logits, max-pool, sparse matmul."""
    import torch
    if not doc_ids or not queries:
        return {q: {} for q in queries}

    dev = device()

    def encode(texts):
        rows = []
        for i in range(0, len(texts), batch):
            inp = {k: v.to(dev) for k, v in tok(
                texts[i:i + batch], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len).items()}
            with torch.no_grad():
                v = torch.log(1 + torch.relu(model(**inp).logits))
            v = (v * inp["attention_mask"].unsqueeze(-1)).max(dim=1).values
            # scipy.sparse needs float32+ so cast back from fp16 if needed
            rows.append(sp.csr_matrix(v.cpu().float().numpy()))
        return sp.vstack(rows)

    doc_vecs = encode(doc_texts)
    qids = list(queries)
    q_vecs = encode([queries[q] for q in qids])
    scores = (q_vecs @ doc_vecs.T).toarray()

    k = min(top_k, len(doc_ids))
    out = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(-scores[i], k)[:k] if k < len(doc_ids) else np.argsort(-scores[i])
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_vecs, q_vecs, scores
    free_cuda()
    return out


def medcpt(doc_ids, doc_texts, queries, q_tok, q_mod, a_tok, a_mod,
            top_k=100, batch=8):
    """MedCPT dual-encoder: separate query / article encoders, CLS pooling."""
    import torch
    import torch.nn.functional as F
    if not doc_ids or not queries:
        return {q: {} for q in queries}

    dev = device()

    def encode(texts, tok, mod, max_len):
        embs = []
        for i in range(0, len(texts), batch):
            inp = {k: v.to(dev) for k, v in tok(
                texts[i:i + batch], return_tensors="pt", padding=True,
                truncation=True, max_length=max_len).items()}
            with torch.no_grad():
                e = F.normalize(mod(**inp).last_hidden_state[:, 0, :], dim=-1)
            embs.append(e.cpu().numpy())
        return np.concatenate(embs).astype("float32")

    doc_emb = encode(doc_texts, a_tok, a_mod, 192)
    qids = list(queries)
    q_emb = encode([queries[q] for q in qids], q_tok, q_mod, 64)

    scores = q_emb @ doc_emb.T
    k = min(top_k, len(doc_ids))
    out = {}
    for i, qid in enumerate(qids):
        top = np.argpartition(scores[i], -k)[-k:]
        top = top[np.argsort(-scores[i][top])]
        out[qid] = {doc_ids[j]: float(scores[i][j]) for j in top}
    del doc_emb, q_emb, scores
    free_cuda()
    return out


# ----------------------------------------------------------------------
# Hybrid score fusion (BM25 + dense, with min-max normalisation)
# ----------------------------------------------------------------------

def hybrid(bm_res, dn_res, queries, alpha=0.40):
    out = {}
    for qid in queries:
        sb = bm_res.get(qid, {})
        sd = dn_res.get(qid, {})
        cand = sorted(set(sb) | set(sd))
        if not cand:
            out[qid] = {}
            continue
        sba = np.array([sb.get(c, 0.0) for c in cand])
        sda = np.array([sd.get(c, 0.0) for c in cand])

        def mm(a):
            lo, hi = float(a.min()), float(a.max())
            return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)

        h = alpha * mm(sba) + (1 - alpha) * mm(sda)
        out[qid] = {c: float(s) for c, s in zip(cand, h)}
    return out


def rrf(bm_res, dn_res, queries, k=60):
    out = {}
    for qid in queries:
        scores = {}
        for src in (bm_res.get(qid, {}), dn_res.get(qid, {})):
            for rank, d in enumerate(sorted(src, key=src.get, reverse=True)):
                scores[d] = scores.get(d, 0.0) + 1.0 / (k + rank + 1)
        out[qid] = scores
    return out


# ----------------------------------------------------------------------
# Cross-encoder reranking
# ----------------------------------------------------------------------

def cross_encoder_rerank(queries, corpus, candidates, model, batch=4):
    """Score every (query, candidate) pair with a sentence-transformers
    CrossEncoder; output mirrors the first-stage retriever shape."""
    out = {}
    for qid in queries:
        cand = candidates.get(qid, [])
        if not cand:
            out[qid] = {}
            continue
        texts = [(corpus[d].get("title", "") + " "
                   + corpus[d].get("text", "")).strip() for d in cand]
        scores = model.predict([(queries[qid], t) for t in texts],
                                 batch_size=batch, show_progress_bar=False,
                                 convert_to_numpy=True)
        order = np.argsort(-np.asarray(scores, dtype=float))
        out[qid] = {cand[i]: float(scores[i]) for i in order}
    return out


def medcpt_ce_rerank(queries, corpus, candidates, tok, model, batch=8):
    import torch
    dev = device()
    out = {}
    for qid in queries:
        cand = candidates.get(qid, [])
        if not cand:
            out[qid] = {}
            continue
        texts = [(corpus[d].get("title", "") + " "
                   + corpus[d].get("text", "")).strip() for d in cand]
        scores = []
        for i in range(0, len(cand), batch):
            inp = tok([(queries[qid], t) for t in texts[i:i + batch]],
                       return_tensors="pt", truncation=True, padding=True,
                       max_length=512).to(dev)
            with torch.no_grad():
                lg = model(**inp).logits.squeeze(-1).cpu().float().numpy()
            scores.extend(lg.tolist())
        order = np.argsort(-np.array(scores))
        out[qid] = {cand[i]: float(scores[i]) for i in order}
    return out


# ----------------------------------------------------------------------
# Bootstrap CIs
# ----------------------------------------------------------------------

def paired_bootstrap(a, b, B=10_000, alpha=0.05, seed=42):
    """Paired bootstrap CI for the mean of a-b."""
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


# ----------------------------------------------------------------------
# Model loaders.  No fancy registry -- just call the function and get
# the model.  When you're done with a model, do `del model; free_cuda()`.
# ----------------------------------------------------------------------

def load_minilm():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2", device=device())


def load_bge():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("BAAI/bge-small-en-v1.5", device=device())


def load_e5():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("intfloat/e5-small-v2", device=device())


def load_splade():
    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    name = "naver/splade-cocondenser-ensembledistil"
    dt = torch.float16 if device() == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForMaskedLM.from_pretrained(name, torch_dtype=dt).to(device()).eval()
    return tok, mod


def load_medcpt():
    import torch
    from transformers import AutoTokenizer, AutoModel
    dt = torch.float16 if device() == "cuda" else torch.float32
    qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    qm = AutoModel.from_pretrained(
        "ncbi/MedCPT-Query-Encoder", torch_dtype=dt).to(device()).eval()
    at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
    am = AutoModel.from_pretrained(
        "ncbi/MedCPT-Article-Encoder", torch_dtype=dt).to(device()).eval()
    return qt, qm, at, am


def load_bge_reranker():
    import torch
    from sentence_transformers import CrossEncoder
    kw = {}
    if device() == "cuda":
        kw["model_kwargs"] = {"torch_dtype": torch.float16}
    return CrossEncoder("BAAI/bge-reranker-base", device=device(),
                         max_length=512, **kw)


def load_medcpt_ce():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dt = torch.float16 if device() == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    mod = AutoModelForSequenceClassification.from_pretrained(
        "ncbi/MedCPT-Cross-Encoder", torch_dtype=dt).to(device()).eval()
    return tok, mod
