"""Phase B.2: cross-encoder reranking.

BGE cross-encoder over the strongest first-stage retriever per dataset,
plus the optional in-domain MedCPT cross-encoder on the two biomedical
datasets (NFCorpus, BioASQ-subset).

Produces ``feedback2/tables/reranker_ndcg10.csv``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_beir
from rageval.models import (get_bge, get_bge_reranker, get_e5, get_medcpt_ce,
                              get_minilm, unload)
from rageval.utils import (CACHE_DIR, FEEDBACK2_DIR, bm25_run, compute_ndcg,
                            cross_encoder_rerank, cuda_gc, dense_run,
                            evaluate_retriever, get_device, prep_corpus)


RERANK_BATCH = 4


def medcpt_ce_rerank(queries, corpus, candidates, batch_size: int = 8):
    tok, mod = get_medcpt_ce()
    device = get_device()
    out = {}
    for qid in tqdm(queries, desc="MedCPT-CE rerank"):
        cand_ids = candidates.get(qid, [])
        if not cand_ids:
            out[qid] = {}
            continue
        cand_texts = [(corpus[d].get("title", "") + " "
                        + corpus[d].get("text", "")).strip() for d in cand_ids]
        scores: List[float] = []
        q = queries[qid]
        for i in range(0, len(cand_ids), batch_size):
            inp = tok([(q, t) for t in cand_texts[i:i + batch_size]],
                       return_tensors="pt", truncation=True, padding=True,
                       max_length=512).to(device)
            with torch.no_grad():
                lg = mod(**inp).logits.squeeze(-1).cpu().numpy()
            scores.extend(lg.tolist())
        order = np.argsort(-np.array(scores))
        out[qid] = {cand_ids[i]: float(scores[i]) for i in order}
    return out


def _agg_ndcg10(res, qrels):
    return evaluate_retriever(res, qrels)["aggregate"]["nDCG@10"]


def main() -> None:
    reranker = get_bge_reranker()
    cuda_gc()
    rows = []

    # ---- NFCorpus: BGE-small -> BGE-rerank, BM25 -> BGE-rerank ----------
    print("\n[NFCorpus]")
    corpus, queries, qrels = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    fs_bge = dense_run(doc_ids, doc_texts, queries, get_bge(), top_k=100)
    fs_bm25 = bm25_run(doc_ids, doc_texts, queries, top_k=100)
    fs_bge_n = _agg_ndcg10(fs_bge, qrels)
    fs_bm25_n = _agg_ndcg10(fs_bm25, qrels)

    cand = {q: list(fs_bge[q].keys()) for q in queries}
    rer = cross_encoder_rerank(queries, corpus, cand, reranker)
    cuda_gc()
    rer_n = _agg_ndcg10(rer, qrels)
    rows.append({
        "Dataset": "NFCorpus", "FirstStage": "BGE-small",
        "FirstStage_nDCG10": round(fs_bge_n, 4),
        "RerankedSystem": "BGE-small+BGE-reranker@100",
        "Reranked_nDCG10": round(rer_n, 4),
        "Delta": round(rer_n - fs_bge_n, 4), "Status": "done",
    })
    nf_fs_bge = fs_bge  # save for MedCPT-CE

    cand = {q: list(fs_bm25[q].keys()) for q in queries}
    rer = cross_encoder_rerank(queries, corpus, cand, reranker)
    cuda_gc()
    rer_n = _agg_ndcg10(rer, qrels)
    rows.append({
        "Dataset": "NFCorpus", "FirstStage": "BM25",
        "FirstStage_nDCG10": round(fs_bm25_n, 4),
        "RerankedSystem": "BM25+BGE-reranker@100",
        "Reranked_nDCG10": round(rer_n, 4),
        "Delta": round(rer_n - fs_bm25_n, 4), "Status": "done",
    })
    nf_corpus, nf_qrels, nf_queries = corpus, qrels, queries

    # ---- BioASQ-subset (load from B.1 cache) ----------------------------
    print("\n[BioASQ-subset]")
    cache = pickle.load(open(CACHE_DIR / "bioasq_results.pkl", "rb"))
    ba_corpus = cache["corpus"]
    ba_queries = cache["queries"]
    ba_qrels = cache["qrels"]
    ba_results = cache["results"]
    fs_bm25_ba = ba_results["BM25"]
    fs_bm25_ba_n = _agg_ndcg10(fs_bm25_ba, ba_qrels)
    cand = {q: list(fs_bm25_ba[q].keys()) for q in ba_queries}
    rer = cross_encoder_rerank(ba_queries, ba_corpus, cand, reranker)
    cuda_gc()
    rer_n = _agg_ndcg10(rer, ba_qrels)
    rows.append({
        "Dataset": "BioASQ-subset", "FirstStage": "BM25",
        "FirstStage_nDCG10": round(fs_bm25_ba_n, 4),
        "RerankedSystem": "BM25+BGE-reranker@100",
        "Reranked_nDCG10": round(rer_n, 4),
        "Delta": round(rer_n - fs_bm25_ba_n, 4), "Status": "done",
    })

    # ---- TREC-COVID: E5 -> BGE-rerank -----------------------------------
    print("\n[TREC-COVID]")
    corpus, queries, qrels = load_beir("trec-covid", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    fs = dense_run(doc_ids, doc_texts, queries, get_e5(),
                    qpfx="query: ", dpfx="passage: ", top_k=100)
    fs_n = _agg_ndcg10(fs, qrels)
    cand = {q: list(fs[q].keys()) for q in queries}
    rer = cross_encoder_rerank(queries, corpus, cand, reranker)
    cuda_gc()
    rer_n = _agg_ndcg10(rer, qrels)
    rows.append({
        "Dataset": "TREC-COVID", "FirstStage": "E5-small",
        "FirstStage_nDCG10": round(fs_n, 4),
        "RerankedSystem": "E5-small+BGE-reranker@100",
        "Reranked_nDCG10": round(rer_n, 4),
        "Delta": round(rer_n - fs_n, 4), "Status": "done",
    })
    unload("e5")

    # ---- SciFact: BGE-small -> BGE-rerank -------------------------------
    for ds_name, beir_name in [("SciFact", "scifact"), ("ArguAna", "arguana")]:
        print(f"\n[{ds_name}]")
        corpus, queries, qrels = load_beir(beir_name, split="test")
        doc_ids, doc_texts = prep_corpus(corpus)
        fs = dense_run(doc_ids, doc_texts, queries, get_bge(), top_k=100)
        fs_n = _agg_ndcg10(fs, qrels)
        cand = {q: list(fs[q].keys()) for q in queries}
        rer = cross_encoder_rerank(queries, corpus, cand, reranker)
        cuda_gc()
        rer_n = _agg_ndcg10(rer, qrels)
        rows.append({
            "Dataset": ds_name, "FirstStage": "BGE-small",
            "FirstStage_nDCG10": round(fs_n, 4),
            "RerankedSystem": "BGE-small+BGE-reranker@100",
            "Reranked_nDCG10": round(rer_n, 4),
            "Delta": round(rer_n - fs_n, 4), "Status": "done",
        })

    # ---- MedCPT-CE on biomedical datasets -------------------------------
    print("\n[MedCPT-CE]")
    try:
        get_medcpt_ce()
    except Exception as exc:
        print(f"  MedCPT-CE unavailable: {exc}")
    else:
        # NFCorpus
        cand = {q: list(nf_fs_bge[q].keys()) for q in nf_queries}
        rer = medcpt_ce_rerank(nf_queries, nf_corpus, cand)
        cuda_gc()
        n = _agg_ndcg10(rer, nf_qrels)
        nf_fs_bge_n = _agg_ndcg10(nf_fs_bge, nf_qrels)
        rows.append({
            "Dataset": "NFCorpus", "FirstStage": "BGE-small",
            "FirstStage_nDCG10": round(nf_fs_bge_n, 4),
            "RerankedSystem": "BGE-small+MedCPT-CE@100",
            "Reranked_nDCG10": round(n, 4),
            "Delta": round(n - nf_fs_bge_n, 4), "Status": "done",
        })
        # BioASQ
        cand = {q: list(fs_bm25_ba[q].keys()) for q in ba_queries}
        rer = medcpt_ce_rerank(ba_queries, ba_corpus, cand)
        n = _agg_ndcg10(rer, ba_qrels)
        rows.append({
            "Dataset": "BioASQ-subset", "FirstStage": "BM25",
            "FirstStage_nDCG10": round(fs_bm25_ba_n, 4),
            "RerankedSystem": "BM25+MedCPT-CE@100",
            "Reranked_nDCG10": round(n, 4),
            "Delta": round(n - fs_bm25_ba_n, 4), "Status": "done",
        })
        unload("medcpt_ce")

    df = pd.DataFrame(rows)
    out = FEEDBACK2_DIR / "reranker_ndcg10.csv"
    df.to_csv(out, index=False)
    print("\n", df.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
