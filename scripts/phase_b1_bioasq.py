"""Phase B.1: BioASQ-subset retrieval pipeline.

Loads BioASQ via the three-source fallback (local / BeIR HF mirror /
mini-BioASQ), builds a deterministic subset (qrel-union + up to 10000
distractors), runs all six retrievers, backfills the BioASQ-subset row
in ``multi_dataset_ndcg10_v2.csv``, and computes the
``bioasq_paired_bootstrap.csv`` MedCPT - BM25 contrast.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_bioasq
from rageval.models import (get_bge, get_e5, get_medcpt, get_minilm,
                              get_splade, unload)
from rageval.utils import (CACHE_DIR, FEEDBACK2_DIR, RESULTS_DIR,
                            bm25_run, dense_run, evaluate_retriever,
                            medcpt_run, paired_bootstrap_ci, prep_corpus,
                            splade_run)


SEED = 42
TARGET_SIZE = 50_000
SAMPLE_SIZE = 500


def _build_subset(corpus: dict, queries: dict, qrels: dict, source: str
                    ) -> tuple:
    """Build the deterministic subset described in Section 2 of the report."""
    rng = np.random.default_rng(SEED)
    qrel_union = set()
    for q, rels in qrels.items():
        qrel_union.update(rels.keys())
    qrel_union &= set(corpus.keys())

    remaining = list(set(corpus.keys()) - qrel_union)
    target = min(TARGET_SIZE, len(corpus))
    n_dist = max(0, min(target - len(qrel_union), 10_000, len(remaining)))
    distractors = (rng.choice(remaining, size=n_dist, replace=False).tolist()
                    if n_dist > 0 else [])
    subset_ids = sorted(qrel_union | set(distractors))
    print(f"  qrel_union={len(qrel_union):,}  distractors={n_dist:,}  "
           f"total={len(subset_ids):,}")

    # Persist manifest
    out_dir = RESULTS_DIR / "feedback2" / "datasets"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": SEED, "source": source,
        "protocol": "union_of_qrel_relevant_docs_plus_distractors",
        "target_size": target, "n_qrel_union": len(qrel_union),
        "n_distractors": n_dist, "n_total": len(subset_ids),
        "doc_ids": subset_ids,
    }
    (out_dir / "bioasq_subset_doc_ids.json").write_text(json.dumps(manifest))
    return subset_ids, qrel_union, n_dist


def main() -> None:
    print("Loading BioASQ ...")
    (corpus_full, queries_full, qrels_full), source = load_bioasq()
    print(f"  source: {source}")
    print(f"  full corpus = {len(corpus_full):,}  queries = {len(queries_full):,}")

    subset_ids, qrel_union, n_dist = _build_subset(
        corpus_full, queries_full, qrels_full, source)

    subset_set = set(subset_ids)
    bioasq_corpus = {d: corpus_full[d] for d in subset_ids}

    bioasq_qrels = {}
    for qid, rels in qrels_full.items():
        inner = {d: r for d, r in rels.items() if d in subset_set}
        if inner:
            bioasq_qrels[qid] = inner

    eligible = sorted(bioasq_qrels.keys())
    rng = np.random.default_rng(SEED)
    sample_n = min(SAMPLE_SIZE, len(eligible))
    sampled = sorted(rng.choice(eligible, size=sample_n, replace=False).tolist())
    bioasq_queries = {q: queries_full[q] for q in sampled if q in queries_full}
    bioasq_qrels = {q: bioasq_qrels[q] for q in bioasq_queries}
    print(f"  sampled {len(bioasq_queries)} queries")

    doc_ids, doc_texts = prep_corpus(bioasq_corpus)

    # ---- Six-retriever sweep --------------------------------------------
    results = {}
    print("\nBM25 ..."); results["BM25"] = bm25_run(doc_ids, doc_texts, bioasq_queries)
    print("Dense (MiniLM) ...");
    results["Dense"] = dense_run(doc_ids, doc_texts, bioasq_queries, get_minilm())
    print("BGE-small ...")
    results["BGE-small"] = dense_run(doc_ids, doc_texts, bioasq_queries, get_bge())
    print("E5-small ...")
    results["E5-small"] = dense_run(doc_ids, doc_texts, bioasq_queries, get_e5(),
                                       qpfx="query: ", dpfx="passage: ")
    print("SPLADE ...")
    stok, smod = get_splade()
    results["SPLADE"] = splade_run(doc_ids, doc_texts, bioasq_queries, stok, smod)
    print("MedCPT ...")
    qt, qm, at, am = get_medcpt()
    results["MedCPT"] = medcpt_run(doc_ids, doc_texts, bioasq_queries,
                                      qt, qm, at, am)

    # cache for B.2 reranker reuse
    import pickle
    with open(CACHE_DIR / "bioasq_results.pkl", "wb") as fh:
        pickle.dump({"results": results, "qrels": bioasq_qrels,
                      "queries": bioasq_queries, "corpus": bioasq_corpus,
                      "doc_ids": doc_ids, "doc_texts": doc_texts}, fh)

    # ---- Aggregate metrics + backfill multi_dataset_ndcg10_v2.csv -------
    eval_per = {n: evaluate_retriever(r, bioasq_qrels) for n, r in results.items()}
    md_path = FEEDBACK2_DIR / "multi_dataset_ndcg10_v2.csv"
    md = pd.read_csv(md_path)
    mask = md["Dataset"] == "BioASQ-subset"
    for col, name in [("BM25 nDCG@10", "BM25"), ("Dense nDCG@10", "Dense"),
                       ("BGE-small nDCG@10", "BGE-small"),
                       ("E5-small nDCG@10", "E5-small"),
                       ("SPLADE nDCG@10", "SPLADE"),
                       ("MedCPT nDCG@10", "MedCPT")]:
        md.loc[mask, col] = round(eval_per[name]["aggregate"]["nDCG@10"], 4)
    md.loc[mask, "# Docs"] = len(doc_ids)
    md.loc[mask, "# Queries"] = len(bioasq_queries)
    md.loc[mask, "Notes"] = (
        f"Subset of qrel-union ({len(qrel_union):,}) + {n_dist:,} distractors, "
        f"seed=42, {len(bioasq_queries)}-query sample")
    md.to_csv(md_path, index=False)
    print(f"\nUpdated {md_path}")

    # ---- Per-query nDCG@10 parquet (used by router) --------------------
    qids = sorted(bioasq_queries.keys())
    perquery = {"qid": qids}
    for n, ev in eval_per.items():
        perquery[n] = [ev["per_query"][q]["nDCG@10"] for q in qids]
    pd.DataFrame(perquery).to_parquet(
        CACHE_DIR / "bioasq_perquery.parquet", index=False)

    # ---- Paired bootstrap MedCPT - BM25 --------------------------------
    a = np.array([eval_per["MedCPT"]["per_query"][q]["nDCG@10"] for q in qids])
    b = np.array([eval_per["BM25"]["per_query"][q]["nDCG@10"] for q in qids])
    ci = paired_bootstrap_ci(a, b)
    pd.DataFrame([{
        "contrast": "MedCPT - BM25",
        "dataset": "BioASQ-subset",
        "n_queries": len(qids),
        "mean_diff": round(ci["mean_diff"], 4),
        "ci_lo": round(ci["ci_lo"], 4),
        "ci_hi": round(ci["ci_hi"], 4),
        "p_a_gt_b": round(ci["p_a_gt_b"], 3),
    }]).to_csv(FEEDBACK2_DIR / "bioasq_paired_bootstrap.csv", index=False)
    print(f"\nMedCPT - BM25 (BioASQ): mean={ci['mean_diff']:+.4f} "
           f"CI=[{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] "
           f"P(MedCPT>BM25)={ci['p_a_gt_b']:.3f}")

    unload("minilm", "e5", "splade", "medcpt")


if __name__ == "__main__":
    main()
