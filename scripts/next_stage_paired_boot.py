"""Paired bootstrap CIs and per-split metrics on NFCorpus.

Produces
--------
- ``next_stage/tables/split_metrics_baselines.csv`` (BM25, Dense per split)
- ``next_stage/tables/split_metrics_with_hybrid.csv`` (adds Hybrid)
- ``next_stage/tables/hybrid_alpha_sweep_dev.csv``
- ``next_stage/tables/hybrid_test_comparison.csv`` (BM25, Dense, Hybrid, RRF)
- ``next_stage/tables/query_subset_resampling.csv`` (per-trial values)
- ``next_stage/tables/query_subset_resampling_summary.csv`` (Table 12)
- ``next_stage/tables/wilcoxon_dense_bm25.csv``

Tables in the report
--------------------
Table 6 (paired bootstrap), Table 12 (subsampling), Table 13 (gap baselines).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_beir
from rageval.models import get_minilm
from rageval.utils import (NEXT_STAGE_DIR, bm25_run, dense_run,
                            evaluate_retriever, hybrid_run,
                            paired_bootstrap_ci, prep_corpus)


def _per_split_metrics(corpus, splits: Dict[str, dict]) -> pd.DataFrame:
    """Run BM25 and Dense (MiniLM) on every split and aggregate."""
    doc_ids, doc_texts = prep_corpus(corpus)
    rows = []
    minilm = get_minilm()
    for split_name, payload in splits.items():
        queries = payload["queries"]
        qrels = payload["qrels"]
        bm = bm25_run(doc_ids, doc_texts, queries, top_k=100)
        dn = dense_run(doc_ids, doc_texts, queries, minilm, top_k=100)
        ev_bm = evaluate_retriever(bm, qrels)
        ev_dn = evaluate_retriever(dn, qrels)
        for name, ev in [("BM25", ev_bm), ("Dense", ev_dn)]:
            a = ev["aggregate"]
            loss = ev["loss_array"]
            rows.append({
                "split": split_name, "method": name,
                "n_queries": len(queries),
                "nDCG@10": a["nDCG@10"], "Recall@10": a["Recall@10"],
                "P@10": a["P@10"],
                "empirical_risk": float(loss.mean()),
                "loss_variance": float(loss.var(ddof=1)),
            })
    return pd.DataFrame(rows)


def _alpha_sweep(corpus, dev_queries, dev_qrels) -> tuple:
    """Tune alpha in [0, 1] on the dev split.  Returns (df, best_alpha)."""
    doc_ids, doc_texts = prep_corpus(corpus)
    bm = bm25_run(doc_ids, doc_texts, dev_queries, top_k=100)
    dn = dense_run(doc_ids, doc_texts, dev_queries, get_minilm(), top_k=100)
    rows = []
    best = (0.0, -1.0)
    for a in np.arange(0.0, 1.01, 0.1):
        hyb = hybrid_run(bm, dn, dev_queries, alpha=float(a))
        ev = evaluate_retriever(hyb, dev_qrels)
        agg = ev["aggregate"]
        loss = ev["loss_array"]
        rows.append({
            "alpha": round(float(a), 1),
            "nDCG@10": agg["nDCG@10"],
            "Recall@10": agg["Recall@10"],
            "P@10": agg["P@10"],
            "empirical_risk": float(loss.mean()),
        })
        if agg["nDCG@10"] > best[1]:
            best = (round(float(a), 1), agg["nDCG@10"])
    return pd.DataFrame(rows), best[0]


def _rrf(bm, dn, queries, k: int = 60) -> Dict[str, Dict[str, float]]:
    """Reciprocal Rank Fusion."""
    out: Dict[str, Dict[str, float]] = {}
    for qid in queries:
        scores: Dict[str, float] = {}
        for src in (bm.get(qid, {}), dn.get(qid, {})):
            for rank, did in enumerate(sorted(src, key=src.get, reverse=True)):
                scores[did] = scores.get(did, 0.0) + 1.0 / (k + rank + 1)
        out[qid] = scores
    return out


def main() -> None:
    splits: Dict[str, dict] = {}
    corpus = None
    for split in ("train", "dev", "test"):
        c, q, r = load_beir("nfcorpus", split=split)
        if corpus is None:
            corpus = c
        splits[split] = {"queries": q, "qrels": r}

    # ---- per-split BM25 + Dense ----------------------------------------
    base = _per_split_metrics(corpus, splits)
    base.to_csv(NEXT_STAGE_DIR / "split_metrics_baselines.csv", index=False)
    print(f"wrote split_metrics_baselines.csv ({len(base)} rows)")

    # ---- alpha sweep on dev --------------------------------------------
    sweep, best_alpha = _alpha_sweep(
        corpus, splits["dev"]["queries"], splits["dev"]["qrels"])
    sweep.to_csv(NEXT_STAGE_DIR / "hybrid_alpha_sweep_dev.csv", index=False)
    print(f"wrote hybrid_alpha_sweep_dev.csv (alpha* = {best_alpha})")

    # ---- per-split + Hybrid (with best alpha) --------------------------
    doc_ids, doc_texts = prep_corpus(corpus)
    minilm = get_minilm()
    rows_h = base.to_dict(orient="records")
    for split_name, payload in splits.items():
        bm = bm25_run(doc_ids, doc_texts, payload["queries"], top_k=100)
        dn = dense_run(doc_ids, doc_texts, payload["queries"], minilm, top_k=100)
        hyb = hybrid_run(bm, dn, payload["queries"], alpha=best_alpha)
        ev = evaluate_retriever(hyb, payload["qrels"])
        a = ev["aggregate"]; loss = ev["loss_array"]
        rows_h.append({
            "split": split_name, "method": "Hybrid",
            "n_queries": len(payload["queries"]),
            "nDCG@10": a["nDCG@10"], "Recall@10": a["Recall@10"],
            "P@10": a["P@10"],
            "empirical_risk": float(loss.mean()),
            "loss_variance": float(loss.var(ddof=1)),
        })
    pd.DataFrame(rows_h).to_csv(
        NEXT_STAGE_DIR / "split_metrics_with_hybrid.csv", index=False)
    print("wrote split_metrics_with_hybrid.csv")

    # ---- hybrid_test_comparison.csv (BM25, Dense, Hybrid, RRF on test) -
    bm = bm25_run(doc_ids, doc_texts, splits["test"]["queries"], top_k=100)
    dn = dense_run(doc_ids, doc_texts, splits["test"]["queries"], minilm,
                    top_k=100)
    hyb = hybrid_run(bm, dn, splits["test"]["queries"], alpha=best_alpha)
    rrf_res = _rrf(bm, dn, splits["test"]["queries"])
    methods = {"BM25": bm, "Dense": dn, "Hybrid": hyb,
                f"RRF(k=60)": rrf_res}
    test_rows = []
    for name, res in methods.items():
        ev = evaluate_retriever(res, splits["test"]["qrels"])
        a = ev["aggregate"]; loss = ev["loss_array"]
        test_rows.append({
            "method": name,
            "nDCG@10": a["nDCG@10"], "Recall@10": a["Recall@10"],
            "P@10": a["P@10"],
            "empirical_risk": float(loss.mean()),
            "loss_variance": float(loss.var(ddof=1)),
        })
    pd.DataFrame(test_rows).to_csv(
        NEXT_STAGE_DIR / "hybrid_test_comparison.csv", index=False)
    print("wrote hybrid_test_comparison.csv")

    # ---- paired bootstrap + Wilcoxon (Dense - BM25) --------------------
    test_qrels = splits["test"]["qrels"]
    qids = sorted(set(test_qrels.keys()) & set(bm.keys()))

    def _per_q(res):
        from rageval.utils import compute_ndcg
        per = {}
        for qid in qids:
            ranking = [d for d, _ in sorted(res[qid].items(), key=lambda x: -x[1])]
            per[qid] = compute_ndcg(ranking, test_qrels[qid], 10)
        return np.array([per[q] for q in qids])

    bm_per = _per_q(bm)
    dn_per = _per_q(dn)
    pb = paired_bootstrap_ci(dn_per, bm_per, B=10_000, seed=42)
    print(f"\nDense - BM25 paired boot: mean={pb['mean_diff']:+.4f} "
           f"CI=[{pb['ci_lo']:+.4f}, {pb['ci_hi']:+.4f}]")

    from scipy.stats import wilcoxon
    w = wilcoxon(dn_per, bm_per, zero_method="wilcox", alternative="two-sided")
    pd.DataFrame([{
        "contrast": "Dense - BM25",
        "wilcoxon_W": float(w.statistic),
        "wilcoxon_p": float(w.pvalue),
        "n_queries": len(qids),
        "bm25_wins": int(np.sum(bm_per > dn_per)),
        "dense_wins": int(np.sum(dn_per > bm_per)),
        "ties": int(np.sum(bm_per == dn_per)),
    }]).to_csv(NEXT_STAGE_DIR / "wilcoxon_dense_bm25.csv", index=False)
    print(f"Wilcoxon W={w.statistic:.1f}, p={w.pvalue:.4f}")

    # ---- query-subset resampling (Table 12) ----------------------------
    print("\nQuery-subset resampling: 2000 trials of n=323 from 3237 queries ...")
    union_q = {}
    union_qrels = {}
    for s in ("train", "dev", "test"):
        union_q.update(splits[s]["queries"])
        union_qrels.update(splits[s]["qrels"])
    all_qids = sorted(union_q.keys())
    target_n = len(splits["test"]["queries"])

    bm_full = bm25_run(doc_ids, doc_texts, union_q, top_k=100)
    dn_full = dense_run(doc_ids, doc_texts, union_q, minilm, top_k=100)
    hy_full = hybrid_run(bm_full, dn_full, union_q, alpha=best_alpha)

    from rageval.utils import compute_ndcg
    bm_perq = {q: compute_ndcg(
        [d for d, _ in sorted(bm_full[q].items(), key=lambda x: -x[1])],
        union_qrels.get(q, {}), 10) for q in all_qids}
    dn_perq = {q: compute_ndcg(
        [d for d, _ in sorted(dn_full[q].items(), key=lambda x: -x[1])],
        union_qrels.get(q, {}), 10) for q in all_qids}
    hy_perq = {q: compute_ndcg(
        [d for d, _ in sorted(hy_full[q].items(), key=lambda x: -x[1])],
        union_qrels.get(q, {}), 10) for q in all_qids}

    rng = np.random.default_rng(42)
    trial_records: List[dict] = []
    for trial in range(2000):
        idx = rng.choice(len(all_qids), size=target_n, replace=True)
        qs = [all_qids[i] for i in idx]
        bm_v = float(np.mean([bm_perq[q] for q in qs]))
        dn_v = float(np.mean([dn_perq[q] for q in qs]))
        hy_v = float(np.mean([hy_perq[q] for q in qs]))
        trial_records.append({
            "trial": trial, "BM25": bm_v, "Dense": dn_v, "Hybrid": hy_v,
            "Dense_minus_BM25": dn_v - bm_v,
            "Hybrid_minus_Dense": hy_v - dn_v,
        })
    trials_df = pd.DataFrame(trial_records)
    trials_df.to_csv(NEXT_STAGE_DIR / "query_subset_resampling.csv", index=False)

    summ_rows = []
    for col in ("BM25", "Dense", "Hybrid", "Dense_minus_BM25", "Hybrid_minus_Dense"):
        v = trials_df[col].values
        summ_rows.append({
            "method": col,
            "subset_size": target_n,
            "n_trials": 2000,
            f"mean_subset_nDCG@10": float(v.mean()),
            "q025": float(np.quantile(v, 0.025)),
            "q975": float(np.quantile(v, 0.975)),
            f"std_subset_nDCG@10": float(v.std(ddof=1)),
        })
    pd.DataFrame(summ_rows).to_csv(
        NEXT_STAGE_DIR / "query_subset_resampling_summary.csv", index=False)
    print(f"wrote query_subset_resampling[_summary].csv")


if __name__ == "__main__":
    main()
