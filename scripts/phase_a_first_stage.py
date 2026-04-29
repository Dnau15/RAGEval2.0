"""Phase A first-stage: NFCorpus 6-way comparison + multi-dataset sweep.

Produces
--------
- ``notebooks/results/feedback2/tables/nfcorpus_canonical.csv``
  one row per retriever, mean nDCG@10 on the 323-query NFCorpus test split.
- ``notebooks/results/feedback2/tables/multi_dataset_ndcg10_v2.csv``
  six retrievers x five datasets nDCG@10 (BioASQ row left blank; filled by
  ``phase_b1_bioasq.py``).  TREC-COVID skips SPLADE and MedCPT.
- ``notebooks/results/feedback2/tables/nfcorpus_full_metrics.csv``
  the full Table 3: nDCG@{1,3,5,10}, MAP@10, Recall@10, P@10, empirical
  risk and per-query loss variance.

Tables in the report
--------------------
Table 3 (NFCorpus 6-way), Table 4 (multi-dataset).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.datasets import load_beir
from rageval.models import (get_bge, get_e5, get_medcpt, get_minilm,
                              get_splade, unload)
from rageval.utils import (FEEDBACK2_DIR, bm25_run, dense_run, evaluate_retriever,
                            medcpt_run, prep_corpus, splade_run)


def _eval_all_retrievers_on(name: str, *, with_splade_medcpt: bool = True
                              ) -> dict:
    """Run all six retrievers on a single BEIR dataset and return their
    aggregate metrics."""
    print(f"\n[{name}] loading ...")
    corpus, queries, qrels = load_beir(name, split="test")
    doc_ids, doc_texts = prep_corpus(corpus)
    print(f"  docs={len(doc_ids):,}  queries={len(queries):,}")

    out = {}

    print("  BM25 ...")
    out["BM25"] = evaluate_retriever(
        bm25_run(doc_ids, doc_texts, queries, top_k=100), qrels)

    print("  Dense (MiniLM) ...")
    out["Dense"] = evaluate_retriever(
        dense_run(doc_ids, doc_texts, queries, get_minilm(), top_k=100), qrels)

    print("  BGE-small ...")
    out["BGE-small"] = evaluate_retriever(
        dense_run(doc_ids, doc_texts, queries, get_bge(), top_k=100), qrels)

    print("  E5-small ...")
    out["E5-small"] = evaluate_retriever(
        dense_run(doc_ids, doc_texts, queries, get_e5(),
                   qpfx="query: ", dpfx="passage: ", top_k=100), qrels)

    if with_splade_medcpt:
        print("  SPLADE ...")
        stok, smod = get_splade()
        out["SPLADE"] = evaluate_retriever(
            splade_run(doc_ids, doc_texts, queries, stok, smod, top_k=100), qrels)

        print("  MedCPT ...")
        qt, qm, at, am = get_medcpt()
        out["MedCPT"] = evaluate_retriever(
            medcpt_run(doc_ids, doc_texts, queries, qt, qm, at, am, top_k=100),
            qrels)

    for k, v in out.items():
        print(f"    {k:<10} nDCG@10 = {v['aggregate'].get('nDCG@10', 0):.4f}")

    return out


def main() -> None:
    # ---- NFCorpus full table 3 -------------------------------------------
    nf = _eval_all_retrievers_on("nfcorpus", with_splade_medcpt=True)

    # nfcorpus_canonical.csv: just method + nDCG@10
    pd.DataFrame([
        {"Method": m, "nDCG@10": round(nf[m]["aggregate"]["nDCG@10"], 4)}
        for m in ("BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT")
    ]).to_csv(FEEDBACK2_DIR / "nfcorpus_canonical.csv", index=False)
    print(f"\nwrote {FEEDBACK2_DIR / 'nfcorpus_canonical.csv'}")

    # nfcorpus_full_metrics.csv: every metric Table 3 reports
    rows = []
    for m, ev in nf.items():
        a = ev["aggregate"]
        loss = ev["loss_array"]
        rows.append({
            "Method": m,
            "nDCG@1": round(a.get("nDCG@1", float("nan")), 4),
            "nDCG@3": round(a.get("nDCG@3", float("nan")), 4),
            "nDCG@5": round(a.get("nDCG@5", float("nan")), 4),
            "nDCG@10": round(a["nDCG@10"], 4),
            "MAP@10": round(a["MAP@10"], 4),
            "Recall@10": round(a["Recall@10"], 4),
            "P@10": round(a["P@10"], 4),
            "Empirical_risk": round(float(loss.mean()), 4),
            "Loss_variance": round(float(loss.var(ddof=1)), 4),
        })
    pd.DataFrame(rows).to_csv(FEEDBACK2_DIR / "nfcorpus_full_metrics.csv",
                               index=False)
    print(f"wrote {FEEDBACK2_DIR / 'nfcorpus_full_metrics.csv'}")

    # Free heavy first-stage retrievers we no longer need.
    unload("minilm", "e5", "splade", "medcpt")

    # ---- Multi-dataset sweep ---------------------------------------------
    multi_rows = [
        # NFCorpus row reuses the canonical numbers above
        {"Dataset": "NFCorpus", "Domain": "biomedical", "# Docs": 3633,
         "# Queries": 323,
         **{f"{m} nDCG@10": round(nf[m]["aggregate"]["nDCG@10"], 4)
            for m in ("BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT")},
         "Notes": "Canonical NFCorpus test run"},

        # BioASQ-subset is filled by phase_b1_bioasq.py
        {"Dataset": "BioASQ-subset", "Domain": "biomedical",
         "# Docs": np.nan, "# Queries": np.nan,
         **{f"{m} nDCG@10": np.nan for m in
            ("BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT")},
         "Notes": "Filled by phase_b1_bioasq.py"},
    ]

    # TREC-COVID, SciFact, ArguAna  -- BM25 + 3 dense; SPLADE/MedCPT only
    # where the corpus is small enough.
    for ds_name, beir_name, run_splade in [
        ("TREC-COVID", "trec-covid", False),
        ("SciFact", "scifact", True),
        ("ArguAna", "arguana", True),
    ]:
        ev = _eval_all_retrievers_on(beir_name, with_splade_medcpt=run_splade)
        row: dict = {"Dataset": ds_name,
                      "Domain": {"trec-covid": "biomedical",
                                 "scifact": "scientific",
                                 "arguana": "argumentation"}[beir_name],
                      "# Docs": {"trec-covid": 171332, "scifact": 5183,
                                  "arguana": 8674}[beir_name],
                      "# Queries": {"trec-covid": 50, "scifact": 300,
                                     "arguana": 1406}[beir_name]}
        for m in ("BM25", "Dense", "BGE-small", "E5-small", "SPLADE", "MedCPT"):
            row[f"{m} nDCG@10"] = (
                round(ev[m]["aggregate"]["nDCG@10"], 4) if m in ev else np.nan)
        row["Notes"] = ("SPLADE and MedCPT skipped due compute budget"
                         if not run_splade else "")
        multi_rows.append(row)

        unload("minilm", "e5", "splade", "medcpt")

    pd.DataFrame(multi_rows).to_csv(
        FEEDBACK2_DIR / "multi_dataset_ndcg10_v2.csv", index=False)
    print(f"\nwrote {FEEDBACK2_DIR / 'multi_dataset_ndcg10_v2.csv'}")


if __name__ == "__main__":
    main()
