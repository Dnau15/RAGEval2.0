"""Phase B.3: per-query router on NFCorpus.

Builds six per-query features, computes per-strategy nDCG@10 across
train/dev/test for {BM25, BGE-small, Hybrid(alpha=0.40, BM25+BGE),
BGE-small+BGE-reranker}, trains logistic and LightGBM classifiers,
reports oracle upper bound and train-test generalisation gap.

Produces
--------
- ``feedback2/tables/router_test.csv``
- ``feedback2/tables/router_feature_importance.csv``
- ``feedback2/tables/router_train_test_gap.csv``
"""

from __future__ import annotations

import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_beir
from rageval.models import get_bge, get_bge_reranker, get_minilm
from rageval.utils import (FEEDBACK2_DIR, bm25_run, compute_ndcg,
                            cross_encoder_rerank, cuda_gc, dense_run,
                            hybrid_run, prep_corpus)

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")
ALPHA = 0.40
QUESTION_RE = re.compile(
    r"^\s*(what|who|how|why|when|where|is|are|does|do|can|should)\b", re.I)
MED_AFFIX = ("mab", "tinib", "vir", "olol", "azepam", "azole", "statin",
              "cycline", "itis", "osis", "emia", "oma", "pathy", "cardio",
              "neuro", "hepato", "renal", "gastro", "pulmonary")


def _toks(s):
    return [t.lower() for t in WORD_RE.findall(s)]


def _med_share(toks):
    if not toks:
        return 0.0
    return sum(any(t.endswith(x) or x in t for x in MED_AFFIX) for t in toks) / len(toks)


def _vocab_gap(qtxt, rels, doc_text_by_id):
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


def _per_q_ndcg(res, qrels):
    out = {}
    for qid, scores in res.items():
        ranking = [d for d, _ in sorted(scores.items(), key=lambda x: -x[1])]
        out[qid] = compute_ndcg(ranking, qrels.get(qid, {}), 10)
    return out


def _bm25_top1_scores(queries, doc_ids, doc_texts):
    import bm25s
    corpus_tok = bm25s.tokenize(doc_texts, stopwords="en", show_progress=False)
    bm = bm25s.BM25(k1=1.5, b=0.75)
    bm.index(corpus_tok)
    out = {}
    for qid, q in queries.items():
        qtok = bm25s.tokenize([q], stopwords="en", show_progress=False)
        try:
            _, sc = bm.retrieve(qtok, k=1, n_threads=1, show_progress=False)
        except TypeError:
            _, sc = bm.retrieve(qtok, k=1)
        out[qid] = float(sc[0][0]) if len(sc) and len(sc[0]) else 0.0
    return out


def main() -> None:
    splits = {}
    corpus = None
    for split in ("train", "dev", "test"):
        c, q, r = load_beir("nfcorpus", split=split)
        if corpus is None:
            corpus = c
        splits[split] = {"queries": q, "qrels": r}
    doc_ids, doc_texts = prep_corpus(corpus)
    doc_text_by_id = dict(zip(doc_ids, doc_texts))

    df_count = Counter()
    for t in doc_texts:
        df_count.update(set(_toks(t)))
    n = len(doc_ids)
    idf = {tok: math.log((n + 1) / (df_count[tok] + 1)) + 1.0 for tok in df_count}

    bm25_top1_all = {}
    for sp in ("train", "dev", "test"):
        bm25_top1_all.update(_bm25_top1_scores(splits[sp]["queries"],
                                                  doc_ids, doc_texts))

    feature_rows = []
    for sp in ("train", "dev", "test"):
        for qid, qtxt in splits[sp]["queries"].items():
            toks = _toks(qtxt)
            rels = splits[sp]["qrels"].get(qid, {})
            feature_rows.append({
                "split": sp, "qid": qid,
                "gap": _vocab_gap(qtxt, rels, doc_text_by_id),
                "len": len(toks),
                "tech": float(np.mean([idf.get(t, 1.0) for t in toks])) if toks else 0.0,
                "is_question": int(bool(QUESTION_RE.match(qtxt))),
                "med_share": _med_share(toks),
                "bm25_top1": bm25_top1_all.get(qid, 0.0),
            })
    feats = pd.DataFrame(feature_rows)
    print(f"features: {feats.shape}")

    # ---- Per-strategy nDCG@10 across all splits -------------------------
    bge = get_bge()
    reranker = get_bge_reranker()
    cuda_gc()

    rows = []
    for sp_name, payload in splits.items():
        bm = bm25_run(doc_ids, doc_texts, payload["queries"], top_k=100)
        bg = dense_run(doc_ids, doc_texts, payload["queries"], bge, top_k=100)
        hyb = hybrid_run(bm, bg, payload["queries"], alpha=ALPHA)
        cand = {q: list(bg[q].keys()) for q in payload["queries"]}
        cuda_gc()
        rer = cross_encoder_rerank(payload["queries"], corpus, cand, reranker)
        cuda_gc()
        bm_q = _per_q_ndcg(bm, payload["qrels"])
        bg_q = _per_q_ndcg(bg, payload["qrels"])
        hy_q = _per_q_ndcg(hyb, payload["qrels"])
        rer_q = _per_q_ndcg(rer, payload["qrels"])
        for qid in payload["queries"]:
            rows.append({
                "split": sp_name, "qid": qid,
                "BM25": bm_q.get(qid, 0.0),
                "BGE-small": bg_q.get(qid, 0.0),
                "Hybrid": hy_q.get(qid, 0.0),
                "BGE+BGE-reranker": rer_q.get(qid, 0.0),
            })
    perq = pd.DataFrame(rows)

    # ---- Train logistic + LightGBM router -------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    import lightgbm as lgb

    STRATS = ["BM25", "BGE-small", "Hybrid", "BGE+BGE-reranker"]
    FEATURES = ["gap", "len", "tech", "is_question", "med_share", "bm25_top1"]

    df_all = feats.merge(perq, on=["split", "qid"], how="inner")
    labels = df_all[STRATS].values.argmax(axis=1)
    df_all["label"] = labels

    train_mask = df_all["split"].isin(["train", "dev"])
    test_mask = df_all["split"] == "test"
    Xtr = df_all.loc[train_mask, FEATURES].values
    ytr = df_all.loc[train_mask, "label"].values
    Xte = df_all.loc[test_mask, FEATURES].values
    yte = df_all.loc[test_mask, "label"].values

    log_pipe = make_pipeline(StandardScaler(),
                              LogisticRegression(max_iter=2000))
    log_pipe.fit(Xtr, ytr)

    gbm = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05,
                              n_estimators=300, objective="multiclass",
                              num_class=4, random_state=42, verbose=-1)
    gbm.fit(Xtr, ytr)

    def _route(model, X, df_split):
        preds = model.predict(X)
        chosen = df_split[STRATS].values[np.arange(len(df_split)), preds]
        return float(np.mean(chosen)), preds

    test_df = df_all.loc[test_mask].reset_index(drop=True)
    log_test, _ = _route(log_pipe, Xte, test_df)
    gbm_test, _ = _route(gbm, Xte, test_df)
    oracle = float(np.mean(test_df[STRATS].values.max(axis=1)))
    always_bm = float(np.mean(test_df["BM25"].values))
    always_bge = float(np.mean(test_df["BGE-small"].values))
    always_hyb = float(np.mean(test_df["Hybrid"].values))
    always_rer = float(np.mean(test_df["BGE+BGE-reranker"].values))

    pd.DataFrame([
        {"Method": "Always_BM25", "nDCG@10": round(always_bm, 4), "Status": "baseline"},
        {"Method": "Always_BGE_small", "nDCG@10": round(always_bge, 4), "Status": "baseline"},
        {"Method": "Static_Hybrid_alpha_0.40", "nDCG@10": round(always_hyb, 4), "Status": "baseline"},
        {"Method": "Always_BGE_small+BGE_reranker", "nDCG@10": round(always_rer, 4), "Status": "baseline"},
        {"Method": "Logistic_router", "nDCG@10": round(log_test, 4), "Status": "learned"},
        {"Method": "LightGBM_router", "nDCG@10": round(gbm_test, 4), "Status": "learned"},
        {"Method": "Oracle_router (upper bound)", "nDCG@10": round(oracle, 4), "Status": "oracle"},
    ]).to_csv(FEEDBACK2_DIR / "router_test.csv", index=False)
    print(f"wrote router_test.csv (LightGBM = {gbm_test:.4f}, Oracle = {oracle:.4f})")

    # Feature importance
    fi = pd.DataFrame({
        "feature": FEATURES,
        "gain": gbm.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(FEEDBACK2_DIR / "router_feature_importance.csv", index=False)
    print(f"wrote router_feature_importance.csv")
    print(fi)

    # Train-test gap
    train_df = df_all.loc[train_mask].reset_index(drop=True)
    log_train, _ = _route(log_pipe, Xtr, train_df)
    gbm_train, _ = _route(gbm, Xtr, train_df)
    pd.DataFrame([
        {"Router": "Logistic_router",
         "R_train": round(1 - log_train, 4),
         "R_test": round(1 - log_test, 4),
         "Gap": round((1 - log_test) - (1 - log_train), 4)},
        {"Router": "LightGBM_router",
         "R_train": round(1 - gbm_train, 4),
         "R_test": round(1 - gbm_test, 4),
         "Gap": round((1 - gbm_test) - (1 - gbm_train), 4)},
    ]).to_csv(FEEDBACK2_DIR / "router_train_test_gap.csv", index=False)
    print(f"wrote router_train_test_gap.csv")


if __name__ == "__main__":
    main()
