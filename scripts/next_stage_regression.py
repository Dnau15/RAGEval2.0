"""Δ(q) regression and stratification analyses on NFCorpus.

Produces
--------
- ``next_stage/tables/vocabulary_gap_features.csv`` (per-query features)
- ``next_stage/tables/vocabulary_gap_correlations.csv`` (Spearman ρ)
- ``next_stage/tables/vocabulary_gap_stratification_ndcg10.csv`` (Table 14)
- ``next_stage/tables/technicality_stratification_ndcg10.csv`` (Table 15)
- ``next_stage/tables/delta_regression_coefficients.csv`` (Table 7)

All three features are z-scored before regression.  CIs are bootstrap
percentile intervals (B = 10000).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_beir
from rageval.models import get_minilm
from rageval.utils import (NEXT_STAGE_DIR, bm25_run, dense_run, compute_ndcg,
                            hybrid_run, prep_corpus)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
ALPHA_HYBRID = 0.40


def _toks(s: str) -> list:
    return [t.lower() for t in WORD_RE.findall(s)]


def _vocab_gap(qtxt: str, rels: Dict[str, int],
                doc_text_by_id: Dict[str, str]) -> float:
    qset = set(_toks(qtxt))
    if not qset:
        return 0.0
    best = 0.0
    for d in rels:
        if d not in doc_text_by_id:
            continue
        dset = set(_toks(doc_text_by_id[d]))
        ov = len(qset & dset) / len(qset)
        best = max(best, ov)
    return 1.0 - best


def _idf_table(doc_texts) -> Dict[str, float]:
    from collections import Counter
    df = Counter()
    for t in doc_texts:
        df.update(set(_toks(t)))
    n = len(doc_texts)
    return {tok: math.log((n + 1) / (df[tok] + 1)) + 1.0 for tok in df}


def _per_query_ndcg(res, qrels):
    return {qid: compute_ndcg(
        [d for d, _ in sorted(res[qid].items(), key=lambda x: -x[1])],
        qrels.get(qid, {}), 10) for qid in res}


def main() -> None:
    corpus, queries, qrels = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    doc_text_by_id = dict(zip(doc_ids, doc_texts))
    idf = _idf_table(doc_texts)

    # ---- Per-query features ---------------------------------------------
    feat_rows = []
    for qid, qtxt in queries.items():
        toks = _toks(qtxt)
        rels = qrels.get(qid, {})
        feat_rows.append({
            "qid": qid,
            "gap": _vocab_gap(qtxt, rels, doc_text_by_id),
            "len": len(toks),
            "tech": float(np.mean([idf.get(t, 1.0) for t in toks])) if toks else 0.0,
        })
    feats = pd.DataFrame(feat_rows)
    feats.to_csv(NEXT_STAGE_DIR / "vocabulary_gap_features.csv", index=False)
    print(f"wrote vocabulary_gap_features.csv ({len(feats)} rows)")

    # ---- Compute per-query Δ(q) for Dense - BM25 ------------------------
    bm = bm25_run(doc_ids, doc_texts, queries, top_k=100)
    dn = dense_run(doc_ids, doc_texts, queries, get_minilm(), top_k=100)
    hyb = hybrid_run(bm, dn, queries, alpha=ALPHA_HYBRID)
    bm_q = _per_query_ndcg(bm, qrels)
    dn_q = _per_query_ndcg(dn, qrels)
    hy_q = _per_query_ndcg(hyb, qrels)

    feats["BM25"] = feats["qid"].map(bm_q)
    feats["Dense"] = feats["qid"].map(dn_q)
    feats["Hybrid"] = feats["qid"].map(hy_q)
    feats["delta"] = feats["Dense"] - feats["BM25"]

    # ---- Spearman correlations (Table at end of Sec 5.4) ----------------
    rho_rows = []
    rho_rows.append({
        "relationship": "BM25: retrieved lexical overlap vs nDCG@10",
        "spearman_rho": float(spearmanr(feats["gap"], feats["BM25"]).statistic),
        "p_value": float(spearmanr(feats["gap"], feats["BM25"]).pvalue),
    })
    rho_rows.append({
        "relationship": "Dense: retrieved lexical overlap vs nDCG@10",
        "spearman_rho": float(spearmanr(feats["gap"], feats["Dense"]).statistic),
        "p_value": float(spearmanr(feats["gap"], feats["Dense"]).pvalue),
    })
    rho_rows.append({
        "relationship": "Hybrid: retrieved lexical overlap vs nDCG@10",
        "spearman_rho": float(spearmanr(feats["gap"], feats["Hybrid"]).statistic),
        "p_value": float(spearmanr(feats["gap"], feats["Hybrid"]).pvalue),
    })
    overlap = 1.0 - feats["gap"]
    rho_overlap_delta = spearmanr(overlap, feats["delta"])
    rho_rows.append({
        "relationship": "oracle lexical overlap vs (Dense - BM25) nDCG@10",
        "spearman_rho": float(rho_overlap_delta.statistic),
        "p_value": float(rho_overlap_delta.pvalue),
    })
    pd.DataFrame(rho_rows).to_csv(
        NEXT_STAGE_DIR / "vocabulary_gap_correlations.csv", index=False)
    print("wrote vocabulary_gap_correlations.csv")

    # ---- Stratifications (Tables 14, 15) --------------------------------
    def _strat(col: str, labels):
        bins = pd.qcut(feats[col], 3, labels=labels)
        out = []
        for lab in labels:
            sub = feats[bins == lab]
            out.append({
                "stratum": lab, "n_queries": int(len(sub)),
                "BM25_nDCG@10": float(sub["BM25"].mean()),
                "Dense_nDCG@10": float(sub["Dense"].mean()),
                "Hybrid_nDCG@10": float(sub["Hybrid"].mean()),
                "Dense_minus_BM25": float((sub["Dense"] - sub["BM25"]).mean()),
                "Hybrid_minus_Dense": float((sub["Hybrid"] - sub["Dense"]).mean()),
            })
        return pd.DataFrame(out)

    _strat("gap", ["Low gap", "Medium gap", "High gap"]).to_csv(
        NEXT_STAGE_DIR / "vocabulary_gap_stratification_ndcg10.csv", index=False)
    print("wrote vocabulary_gap_stratification_ndcg10.csv")

    _strat("tech", ["Low IDF / plainer", "Mid IDF",
                       "High IDF / technical"]).to_csv(
        NEXT_STAGE_DIR / "technicality_stratification_ndcg10.csv", index=False)
    print("wrote technicality_stratification_ndcg10.csv")

    # ---- OLS regression with bootstrap CIs (Table 7) --------------------
    feature_cols = ["gap", "len", "tech"]
    X = StandardScaler().fit_transform(feats[feature_cols].values)
    y = feats["delta"].values

    lr = LinearRegression().fit(X, y)
    coefs = [lr.intercept_, *lr.coef_]
    names = ["Intercept", "Vocab gap", "Query length", "Technicality (mean IDF)"]

    # Bootstrap CIs
    rng = np.random.default_rng(42)
    B = 10_000
    boot = np.zeros((B, len(coefs)))
    n = len(y)
    for b in range(B):
        idx = rng.integers(0, n, n)
        Xb, yb = X[idx], y[idx]
        lrb = LinearRegression().fit(Xb, yb)
        boot[b, 0] = lrb.intercept_
        boot[b, 1:] = lrb.coef_

    rows = []
    for i, name in enumerate(names):
        c = coefs[i]
        bs = boot[:, i]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        # Two-sided bootstrap p-value: share of resamples with opposite sign
        p = float((bs * np.sign(c) <= 0).mean()) if c != 0 else 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        rows.append({
            "Feature": name, "Coef": float(c),
            "CI_lo": float(lo), "CI_hi": float(hi),
            "p_boot": round(p, 4), "Sig": sig,
        })
    pd.DataFrame(rows).to_csv(
        NEXT_STAGE_DIR / "delta_regression_coefficients.csv", index=False)
    print(f"wrote delta_regression_coefficients.csv  (R^2 = {lr.score(X, y):.3f})")


if __name__ == "__main__":
    main()
