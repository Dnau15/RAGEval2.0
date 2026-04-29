"""Statistical analyses on NFCorpus.

Per-split metrics, hybrid-alpha sweep on dev, paired bootstrap CI,
Wilcoxon comparison, query-subset resampling, Delta(q) regression with
bootstrap CIs, and the vocabulary-gap / technicality stratifications.

Outputs (under notebooks/results/next_stage/tables/):
    split_metrics_baselines.csv
    split_metrics_with_hybrid.csv
    hybrid_alpha_sweep_dev.csv
    hybrid_test_comparison.csv
    query_subset_resampling.csv
    query_subset_resampling_summary.csv
    wilcoxon_dense_bm25.csv
    vocabulary_gap_features.csv
    vocabulary_gap_correlations.csv
    vocabulary_gap_stratification_ndcg10.csv
    technicality_stratification_ndcg10.csv
    delta_regression_coefficients.csv

Run as:  python scripts/analysis.py
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.data import load_beir
from rageval.retrieval import (
    NEXT_STAGE,
    bm25, dense, hybrid, rrf, evaluate, prep_corpus, free_cuda,
    paired_bootstrap, ndcg, load_minilm,
)


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def toks(s):
    return [t.lower() for t in WORD_RE.findall(s)]


def per_query_ndcg(res, qrels):
    return {qid: ndcg(sorted(r, key=r.get, reverse=True),
                       qrels.get(qid, {}), 10)
            for qid, r in res.items()}


# ======================================================================
# Split metrics + hybrid alpha sweep
# ======================================================================

def per_split_metrics(corpus, splits, model):
    """BM25 + Dense (MiniLM) on every split, aggregated."""
    doc_ids, doc_texts = prep_corpus(corpus)
    rows = []
    for s_name, payload in splits.items():
        bm = bm25(doc_ids, doc_texts, payload["queries"])
        dn = dense(doc_ids, doc_texts, payload["queries"], model)
        for label, res in [("BM25", bm), ("Dense", dn)]:
            ev = evaluate(res, payload["qrels"])
            a, loss = ev["aggregate"], ev["loss"]
            rows.append({
                "split": s_name, "method": label,
                "n_queries": len(payload["queries"]),
                "nDCG@10": a["nDCG@10"], "Recall@10": a["Recall@10"],
                "P@10": a["P@10"],
                "empirical_risk": float(loss.mean()),
                "loss_variance": float(loss.var(ddof=1)),
            })
    return pd.DataFrame(rows)


def alpha_sweep_on_dev(corpus, dev_queries, dev_qrels, model):
    """Tune the BM25-MiniLM hybrid weight on the dev split.  Returns the
    full sweep DataFrame and the best alpha (rounded to 1 dp)."""
    doc_ids, doc_texts = prep_corpus(corpus)
    bm = bm25(doc_ids, doc_texts, dev_queries)
    dn = dense(doc_ids, doc_texts, dev_queries, model)
    rows = []
    best = (0.0, -1.0)
    for a in np.arange(0.0, 1.01, 0.1):
        ev = evaluate(hybrid(bm, dn, dev_queries, alpha=float(a)), dev_qrels)
        agg, loss = ev["aggregate"], ev["loss"]
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


def split_metrics_and_alpha():
    """Section 4.5 of the report: per-split BM25/Dense/Hybrid + the
    BM25 vs Dense vs Hybrid vs RRF table on test."""
    splits = {}
    corpus = None
    for s in ("train", "dev", "test"):
        c, q, r = load_beir("nfcorpus", split=s)
        if corpus is None:
            corpus = c
        splits[s] = {"queries": q, "qrels": r}

    minilm = load_minilm()

    base = per_split_metrics(corpus, splits, minilm)
    base.to_csv(NEXT_STAGE / "split_metrics_baselines.csv", index=False)
    print(f"wrote split_metrics_baselines.csv ({len(base)} rows)")

    sweep, alpha = alpha_sweep_on_dev(
        corpus, splits["dev"]["queries"], splits["dev"]["qrels"], minilm)
    sweep.to_csv(NEXT_STAGE / "hybrid_alpha_sweep_dev.csv", index=False)
    print(f"alpha* on dev = {alpha}")

    # Add Hybrid rows to the per-split table
    doc_ids, doc_texts = prep_corpus(corpus)
    rows_h = base.to_dict(orient="records")
    for s_name, payload in splits.items():
        bm = bm25(doc_ids, doc_texts, payload["queries"])
        dn = dense(doc_ids, doc_texts, payload["queries"], minilm)
        ev = evaluate(hybrid(bm, dn, payload["queries"], alpha=alpha),
                       payload["qrels"])
        a, loss = ev["aggregate"], ev["loss"]
        rows_h.append({
            "split": s_name, "method": "Hybrid",
            "n_queries": len(payload["queries"]),
            "nDCG@10": a["nDCG@10"], "Recall@10": a["Recall@10"],
            "P@10": a["P@10"],
            "empirical_risk": float(loss.mean()),
            "loss_variance": float(loss.var(ddof=1)),
        })
    pd.DataFrame(rows_h).to_csv(
        NEXT_STAGE / "split_metrics_with_hybrid.csv", index=False)

    # Test-split comparison: BM25 / Dense / Hybrid / RRF
    bm = bm25(doc_ids, doc_texts, splits["test"]["queries"])
    dn = dense(doc_ids, doc_texts, splits["test"]["queries"], minilm)
    test_rows = []
    for name, res in [
        ("BM25", bm), ("Dense", dn),
        ("Hybrid", hybrid(bm, dn, splits["test"]["queries"], alpha=alpha)),
        ("RRF(k=60)", rrf(bm, dn, splits["test"]["queries"])),
    ]:
        ev = evaluate(res, splits["test"]["qrels"])
        a, loss = ev["aggregate"], ev["loss"]
        test_rows.append({
            "method": name,
            "nDCG@10": a["nDCG@10"], "Recall@10": a["Recall@10"],
            "P@10": a["P@10"],
            "empirical_risk": float(loss.mean()),
            "loss_variance": float(loss.var(ddof=1)),
        })
    pd.DataFrame(test_rows).to_csv(
        NEXT_STAGE / "hybrid_test_comparison.csv", index=False)
    print("wrote hybrid_test_comparison.csv")

    del minilm; free_cuda()
    return alpha


# ======================================================================
# Paired bootstrap + Wilcoxon + query-subset resampling
# ======================================================================

def boot_and_subsampling(alpha):
    splits = {}
    corpus = None
    for s in ("train", "dev", "test"):
        c, q, r = load_beir("nfcorpus", split=s)
        if corpus is None:
            corpus = c
        splits[s] = {"queries": q, "qrels": r}
    doc_ids, doc_texts = prep_corpus(corpus)
    minilm = load_minilm()

    bm = bm25(doc_ids, doc_texts, splits["test"]["queries"])
    dn = dense(doc_ids, doc_texts, splits["test"]["queries"], minilm)

    qids = sorted(set(splits["test"]["qrels"]) & set(bm))
    bm_p = np.array([per_query_ndcg(bm, splits["test"]["qrels"])[q] for q in qids])
    dn_p = np.array([per_query_ndcg(dn, splits["test"]["qrels"])[q] for q in qids])

    pb = paired_bootstrap(dn_p, bm_p, B=10_000, seed=42)
    print(f"\nDense-BM25 bootstrap: mean={pb['mean_diff']:+.4f}  "
           f"CI=[{pb['ci_lo']:+.4f}, {pb['ci_hi']:+.4f}]")

    # Wilcoxon as a pedagogical contrast (the report explains why
    # paired bootstrap is preferred for an effect-size claim)
    from scipy.stats import wilcoxon
    w = wilcoxon(dn_p, bm_p, zero_method="wilcox", alternative="two-sided")
    pd.DataFrame([{
        "contrast": "Dense - BM25",
        "wilcoxon_W": float(w.statistic), "wilcoxon_p": float(w.pvalue),
        "n_queries": len(qids),
        "bm25_wins": int(np.sum(bm_p > dn_p)),
        "dense_wins": int(np.sum(dn_p > bm_p)),
        "ties": int(np.sum(bm_p == dn_p)),
    }]).to_csv(NEXT_STAGE / "wilcoxon_dense_bm25.csv", index=False)
    print(f"Wilcoxon W={w.statistic:.1f}  p={w.pvalue:.4f}")

    # Query-subset resampling: 2000 trials of n=323 from the 3237-query union
    print("\nQuery-subset resampling: 2000 trials of n=323 from 3237 ...")
    union_q, union_qrels = {}, {}
    for s in ("train", "dev", "test"):
        union_q.update(splits[s]["queries"])
        union_qrels.update(splits[s]["qrels"])
    all_qids = sorted(union_q)
    target_n = len(splits["test"]["queries"])

    bm_full = bm25(doc_ids, doc_texts, union_q)
    dn_full = dense(doc_ids, doc_texts, union_q, minilm)
    hy_full = hybrid(bm_full, dn_full, union_q, alpha=alpha)

    bm_pq = per_query_ndcg(bm_full, union_qrels)
    dn_pq = per_query_ndcg(dn_full, union_qrels)
    hy_pq = per_query_ndcg(hy_full, union_qrels)

    rng = np.random.default_rng(42)
    rows = []
    for trial in range(2000):
        idx = rng.choice(len(all_qids), size=target_n, replace=True)
        qs = [all_qids[i] for i in idx]
        bm_v = float(np.mean([bm_pq[q] for q in qs]))
        dn_v = float(np.mean([dn_pq[q] for q in qs]))
        hy_v = float(np.mean([hy_pq[q] for q in qs]))
        rows.append({"trial": trial, "BM25": bm_v, "Dense": dn_v, "Hybrid": hy_v,
                      "Dense_minus_BM25": dn_v - bm_v,
                      "Hybrid_minus_Dense": hy_v - dn_v})
    trials = pd.DataFrame(rows)
    trials.to_csv(NEXT_STAGE / "query_subset_resampling.csv", index=False)

    summ = []
    for col in ("BM25", "Dense", "Hybrid", "Dense_minus_BM25", "Hybrid_minus_Dense"):
        v = trials[col].values
        summ.append({
            "method": col, "subset_size": target_n, "n_trials": 2000,
            "mean_subset_nDCG@10": float(v.mean()),
            "q025": float(np.quantile(v, 0.025)),
            "q975": float(np.quantile(v, 0.975)),
            "std_subset_nDCG@10": float(v.std(ddof=1)),
        })
    pd.DataFrame(summ).to_csv(
        NEXT_STAGE / "query_subset_resampling_summary.csv", index=False)
    print("wrote query_subset_resampling[_summary].csv")

    del minilm; free_cuda()


# ======================================================================
# Per-query features, stratifications, OLS regression with bootstrap CIs
# ======================================================================

def vocab_gap(qtxt, rels, doc_text_by_id):
    qset = set(toks(qtxt))
    if not qset:
        return 0.0
    best = 0.0
    for d in rels:
        if d in doc_text_by_id:
            ov = len(qset & set(toks(doc_text_by_id[d]))) / len(qset)
            best = max(best, ov)
    return 1.0 - best


def regression_and_stratifications():
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    corpus, queries, qrels = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    doc_text_by_id = dict(zip(doc_ids, doc_texts))

    # IDF table for technicality
    df = Counter()
    for t in doc_texts:
        df.update(set(toks(t)))
    n = len(doc_texts)
    idf = {tok: math.log((n + 1) / (df[tok] + 1)) + 1.0 for tok in df}

    # Per-query features
    rows = []
    for qid, qtxt in queries.items():
        t = toks(qtxt)
        rels = qrels.get(qid, {})
        rows.append({
            "qid": qid,
            "gap": vocab_gap(qtxt, rels, doc_text_by_id),
            "len": len(t),
            "tech": float(np.mean([idf.get(w, 1.0) for w in t])) if t else 0.0,
        })
    feats = pd.DataFrame(rows)
    feats.to_csv(NEXT_STAGE / "vocabulary_gap_features.csv", index=False)

    # Per-query nDCG@10 for BM25, Dense, Hybrid -> Δ(q)
    minilm = load_minilm()
    bm = bm25(doc_ids, doc_texts, queries)
    dn = dense(doc_ids, doc_texts, queries, minilm)
    hy = hybrid(bm, dn, queries, alpha=0.40)
    del minilm; free_cuda()

    feats["BM25"] = feats["qid"].map(per_query_ndcg(bm, qrels))
    feats["Dense"] = feats["qid"].map(per_query_ndcg(dn, qrels))
    feats["Hybrid"] = feats["qid"].map(per_query_ndcg(hy, qrels))
    feats["delta"] = feats["Dense"] - feats["BM25"]

    # Spearman correlations
    rho_rows = []
    for label, x, y in [
        ("BM25: retrieved lexical overlap vs nDCG@10", feats["gap"], feats["BM25"]),
        ("Dense: retrieved lexical overlap vs nDCG@10", feats["gap"], feats["Dense"]),
        ("Hybrid: retrieved lexical overlap vs nDCG@10", feats["gap"], feats["Hybrid"]),
        ("oracle lexical overlap vs (Dense - BM25) nDCG@10",
         1.0 - feats["gap"], feats["delta"]),
    ]:
        s = spearmanr(x, y)
        rho_rows.append({"relationship": label,
                          "spearman_rho": float(s.statistic),
                          "p_value": float(s.pvalue)})
    pd.DataFrame(rho_rows).to_csv(
        NEXT_STAGE / "vocabulary_gap_correlations.csv", index=False)

    # Stratifications (rank-based qcut so ties don't collapse the bins)
    def stratify(col, labels):
        bins = pd.qcut(feats[col].rank(method="first"), 3, labels=labels)
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

    stratify("gap", ["Low gap", "Medium gap", "High gap"]).to_csv(
        NEXT_STAGE / "vocabulary_gap_stratification_ndcg10.csv", index=False)
    stratify("tech", ["Low IDF / plainer", "Mid IDF",
                        "High IDF / technical"]).to_csv(
        NEXT_STAGE / "technicality_stratification_ndcg10.csv", index=False)

    # OLS regression with bootstrap CIs
    cols = ["gap", "len", "tech"]
    X = StandardScaler().fit_transform(feats[cols].values)
    y = feats["delta"].values

    lr = LinearRegression().fit(X, y)
    coefs = [lr.intercept_, *lr.coef_]
    names = ["Intercept", "Vocab gap", "Query length",
             "Technicality (mean IDF)"]

    rng = np.random.default_rng(42)
    B = 10_000
    boot = np.zeros((B, len(coefs)))
    for b in range(B):
        idx = rng.integers(0, len(y), len(y))
        lrb = LinearRegression().fit(X[idx], y[idx])
        boot[b, 0] = lrb.intercept_
        boot[b, 1:] = lrb.coef_

    rows = []
    for i, name in enumerate(names):
        c = coefs[i]
        bs = boot[:, i]
        lo, hi = np.quantile(bs, [0.025, 0.975])
        # Two-sided bootstrap p-value
        p = float((bs * np.sign(c) <= 0).mean()) if c != 0 else 1.0
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        rows.append({"Feature": name, "Coef": float(c),
                      "CI_lo": float(lo), "CI_hi": float(hi),
                      "p_boot": round(p, 4), "Sig": sig})
    pd.DataFrame(rows).to_csv(
        NEXT_STAGE / "delta_regression_coefficients.csv", index=False)
    print(f"wrote delta_regression_coefficients.csv (R^2 = {lr.score(X, y):.3f})")


# ======================================================================

if __name__ == "__main__":
    alpha = split_metrics_and_alpha()
    boot_and_subsampling(alpha)
    regression_and_stratifications()
