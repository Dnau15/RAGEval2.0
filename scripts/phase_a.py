"""Phase A — first-stage retrieval, multi-dataset sweep, n_required, efficiency.

Outputs (under notebooks/results/feedback2/tables/):
    nfcorpus_canonical.csv      -- one row per retriever, mean nDCG@10 on NFCorpus
    nfcorpus_full_metrics.csv   -- the full Table 3 (nDCG@1/3/5/10, MAP, R, P, ...)
    multi_dataset_ndcg10_v2.csv -- 6 retrievers x 5 datasets (BioASQ filled by phase_b.py)
    n_required.csv              -- Hoeffding sample-size requirements (Table 11)
    efficiency.csv              -- index time, latency, throughput, memory (Table 10)

Run as:
    python scripts/phase_a.py
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.data import load_beir
from rageval.retrieval import (
    FEEDBACK2,
    NEXT_STAGE,
    bm25,
    dense,
    evaluate,
    free_cuda,
    load_bge,
    load_e5,
    load_medcpt,
    load_minilm,
    load_splade,
    medcpt,
    prep_corpus,
    splade,
)


RETRIEVERS = ("BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT")

DATASETS = {
    "NFCorpus": {
        "beir_id": "nfcorpus", "n_docs": 3633, "n_queries": 323,
        "domain": "biomedical", "skip_heavy": False,
    },
    "TREC-COVID": {
        "beir_id": "trec-covid", "n_docs": 171332, "n_queries": 50,
        "domain": "biomedical", "skip_heavy": True,  # corpus too big for SPLADE/MedCPT
    },
    "SciFact": {
        "beir_id": "scifact", "n_docs": 5183, "n_queries": 300,
        "domain": "scientific", "skip_heavy": False,
    },
    "ArguAna": {
        "beir_id": "arguana", "n_docs": 8674, "n_queries": 1406,
        "domain": "argumentation", "skip_heavy": False,
    },
}


# ----------------------------------------------------------------------
# First-stage sweep
# ----------------------------------------------------------------------

def run_all_retrievers(beir_id, with_splade_medcpt=True):
    print(f"\n[{beir_id}]")
    corpus, queries, qrels = load_beir(beir_id, split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    print(f"  {len(doc_ids):,} docs / {len(queries):,} queries")

    out = {}

    print("  BM25 ...")
    out["BM25"] = evaluate(bm25(doc_ids, doc_texts, queries), qrels)

    print("  MiniLM ...")
    minilm = load_minilm()
    out["Dense"] = evaluate(dense(doc_ids, doc_texts, queries, minilm), qrels)
    del minilm
    free_cuda()

    print("  BGE-small ...")
    bge = load_bge()
    out["BGE-small"] = evaluate(dense(doc_ids, doc_texts, queries, bge), qrels)
    del bge
    free_cuda()

    print("  E5-small ...")
    e5 = load_e5()
    out["E5-small"] = evaluate(
        dense(doc_ids, doc_texts, queries, e5,
              qpfx="query: ", dpfx="passage: "),
        qrels,
    )
    del e5
    free_cuda()

    if with_splade_medcpt:
        print("  SPLADE ...")
        stok, smod = load_splade()
        out["SPLADE"] = evaluate(
            splade(doc_ids, doc_texts, queries, stok, smod), qrels)
        del stok, smod
        free_cuda()

        print("  MedCPT ...")
        qt, qm, at, am = load_medcpt()
        out["MedCPT"] = evaluate(
            medcpt(doc_ids, doc_texts, queries, qt, qm, at, am), qrels)
        del qt, qm, at, am
        free_cuda()

    for name, ev in out.items():
        print(f"    {name:<10} nDCG@10 = {ev['aggregate']['nDCG@10']:.4f}")
    return out


def first_stage_and_multi_dataset():
    nf = run_all_retrievers("nfcorpus", with_splade_medcpt=True)

    # nfcorpus_canonical.csv -- single column of nDCG@10
    pd.DataFrame([
        {"Method": m, "nDCG@10": round(nf[m]["aggregate"]["nDCG@10"], 4)}
        for m in RETRIEVERS
    ]).to_csv(FEEDBACK2 / "nfcorpus_canonical.csv", index=False)
    print(f"\nwrote {FEEDBACK2 / 'nfcorpus_canonical.csv'}")

    # nfcorpus_full_metrics.csv -- the full Table 3
    full_rows = []
    for m, ev in nf.items():
        agg = ev["aggregate"]
        loss = ev["loss"]
        full_rows.append({
            "Method": m,
            "nDCG@1": round(agg.get("nDCG@1", float("nan")), 4),
            "nDCG@3": round(agg.get("nDCG@3", float("nan")), 4),
            "nDCG@5": round(agg.get("nDCG@5", float("nan")), 4),
            "nDCG@10": round(agg["nDCG@10"], 4),
            "MAP@10": round(agg["MAP@10"], 4),
            "Recall@10": round(agg["Recall@10"], 4),
            "P@10": round(agg["P@10"], 4),
            "Empirical_risk": round(float(loss.mean()), 4),
            "Loss_variance": round(float(loss.var(ddof=1)), 4),
        })
    pd.DataFrame(full_rows).to_csv(
        FEEDBACK2 / "nfcorpus_full_metrics.csv", index=False)
    print(f"wrote {FEEDBACK2 / 'nfcorpus_full_metrics.csv'}")

    # multi_dataset_ndcg10_v2.csv -- 6 retrievers x 5 datasets.
    # The BioASQ row is created here as NaN and filled in by phase_b.py.
    multi = [_row_for("NFCorpus", nf), _bioasq_placeholder()]
    for ds_name, info in DATASETS.items():
        if ds_name == "NFCorpus":
            continue
        ev = run_all_retrievers(info["beir_id"],
                                  with_splade_medcpt=not info["skip_heavy"])
        multi.append(_row_for(ds_name, ev))

    pd.DataFrame(multi).to_csv(
        FEEDBACK2 / "multi_dataset_ndcg10_v2.csv", index=False)
    print(f"\nwrote {FEEDBACK2 / 'multi_dataset_ndcg10_v2.csv'}")


def _row_for(ds_name, ev):
    info = DATASETS[ds_name]
    row = {
        "Dataset": ds_name,
        "Domain": info["domain"],
        "# Docs": info["n_docs"],
        "# Queries": info["n_queries"],
    }
    for m in RETRIEVERS:
        row[f"{m} nDCG@10"] = (
            round(ev[m]["aggregate"]["nDCG@10"], 4) if m in ev else np.nan)

    if ds_name == "NFCorpus":
        row["Notes"] = "Canonical NFCorpus test run"
    elif info["skip_heavy"]:
        row["Notes"] = "SPLADE and MedCPT skipped due compute budget"
    else:
        row["Notes"] = ""
    return row


def _bioasq_placeholder():
    return {
        "Dataset": "BioASQ-subset",
        "Domain": "biomedical",
        "# Docs": np.nan,
        "# Queries": np.nan,
        **{f"{m} nDCG@10": np.nan for m in RETRIEVERS},
        "Notes": "Filled by scripts/phase_b.py",
    }


# ----------------------------------------------------------------------
# Hoeffding n_min (Table 11). Reads canonical + hybrid CSVs.
# ----------------------------------------------------------------------

CONFIDENCE = 0.95
DELTA = 1.0 - CONFIDENCE


def hoeffding_n(gap):
    if abs(gap) < 1e-12:
        return 10 ** 9
    return math.ceil(math.log(2 / DELTA) / (2 * gap ** 2))


def write_n_required():
    canonical = FEEDBACK2 / "nfcorpus_canonical.csv"
    htc = NEXT_STAGE / "hybrid_test_comparison.csv"
    if not canonical.exists() or not htc.exists():
        print("  skip n_required (need canonical + hybrid CSVs first)")
        return

    nf = {
        r["Method"]: float(r["nDCG@10"])
        for _, r in pd.read_csv(canonical).iterrows()
    }
    hyb = pd.read_csv(htc).set_index("method")["nDCG@10"]

    contrasts = [
        ("BGE-small", "BM25",     nf["BGE-small"] - nf["BM25"]),
        ("BGE-small", "E5-small", nf["BGE-small"] - nf["E5-small"]),
        ("Hybrid",    "Dense",    float(hyb["Hybrid"] - hyb["Dense"])),
        ("Dense",     "BM25",     float(hyb["Dense"] - hyb["BM25"])),
    ]
    df = pd.DataFrame([
        {
            "A": a,
            "B": b,
            "observed_gap": round(g, 4),
            "hoeffding_n_required_95pct": hoeffding_n(g),
        }
        for a, b, g in contrasts
    ])
    df.to_csv(FEEDBACK2 / "n_required.csv", index=False)
    print(df.to_string(index=False))
    print(f"wrote {FEEDBACK2 / 'n_required.csv'}")


# ----------------------------------------------------------------------
# Efficiency (Table 10). Reloads each model -- numbers depend on host hardware.
# ----------------------------------------------------------------------

LATENCY_QUERIES = 50
LATENCY_REPEATS = 3


def median_time(fn, n=LATENCY_REPEATS):
    times = []
    for _ in range(n):
        t0 = time.time()
        fn()
        times.append(time.time() - t0)
    return float(np.median(times))


def time_once(fn):
    t0 = time.time()
    fn()
    return time.time() - t0


def write_efficiency():
    corpus, queries, _ = load_beir("nfcorpus", split="test")
    doc_ids, doc_texts = prep_corpus(corpus)

    sub = dict(list(queries.items())[:LATENCY_QUERIES])
    one_qid = next(iter(queries))
    one = {one_qid: queries[one_qid]}
    n_docs = len(doc_ids)

    rows = []

    # ---- BM25 ----
    print("\n[BM25]")
    index_t = time_once(lambda: bm25(doc_ids, doc_texts, one))
    latency = median_time(
        lambda: bm25(doc_ids, doc_texts, sub)) * 1000 / len(sub)
    rows.append(("BM25", index_t, latency, "inverted list (sparse)"))

    # ---- Dense bi-encoders ----
    dense_models = [
        ("Dense (MiniLM)", load_minilm, "", "", 384),
        ("BGE-small", load_bge, "", "", 384),
        ("E5-small", load_e5, "query: ", "passage: ", 384),
    ]
    for label, loader, qpfx, dpfx, dim in dense_models:
        print(f"\n[{label}]")
        model = loader()
        index_t = time_once(lambda: model.encode(
            doc_texts, batch_size=16, show_progress_bar=False,
            normalize_embeddings=True, convert_to_numpy=True,
        ))
        # Capture loop variables explicitly so the lambda is independent
        # of subsequent iterations.
        latency = median_time(
            lambda mdl=model, qp=qpfx, dp=dpfx: dense(
                doc_ids, doc_texts, sub, mdl, qpfx=qp, dpfx=dp,
            )
        ) * 1000 / len(sub)
        memory = f"{n_docs * dim * 4 // (1024 ** 2)} MB ({n_docs} x {dim} f32)"
        rows.append((label, index_t, latency, memory))
        del model
        free_cuda()

    # ---- SPLADE ----
    print("\n[SPLADE]")
    stok, smod = load_splade()
    index_t = time_once(lambda: splade(doc_ids, doc_texts, one, stok, smod))
    latency = median_time(
        lambda: splade(doc_ids, doc_texts, sub, stok, smod)) * 1000 / len(sub)
    rows.append(("SPLADE", index_t, latency, "sparse CSR, ~151 nnz/doc"))
    del stok, smod
    free_cuda()

    # ---- MedCPT ----
    print("\n[MedCPT]")
    qt, qm, at, am = load_medcpt()
    index_t = time_once(
        lambda: medcpt(doc_ids, doc_texts, one, qt, qm, at, am))
    latency = median_time(
        lambda: medcpt(doc_ids, doc_texts, sub, qt, qm, at, am)
    ) * 1000 / len(sub)
    memory = f"{n_docs * 768 * 4 // (1024 ** 2)} MB ({n_docs} x 768 f32)"
    rows.append(("MedCPT", index_t, latency, memory))
    del qt, qm, at, am
    free_cuda()

    df = pd.DataFrame([
        {
            "Retriever": label,
            "Index build (s)": round(idx_t, 1),
            "Latency (ms/q)": round(lat, 1),
            "Throughput (q/s)": round(1000 / lat, 0),
            "Index memory": memory,
        }
        for label, idx_t, lat, memory in rows
    ])
    df.to_csv(FEEDBACK2 / "efficiency.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {FEEDBACK2 / 'efficiency.csv'}")


# ----------------------------------------------------------------------

if __name__ == "__main__":
    first_stage_and_multi_dataset()
    write_n_required()
    write_efficiency()
