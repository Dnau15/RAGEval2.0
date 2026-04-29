"""Dataset loaders.

BEIR datasets (NFCorpus, TREC-COVID, SciFact, ArguAna) come from the
official UKP-DARMSTADT mirror.  BioASQ uses a three-source fallback
chain because the official corpus is gated.
"""

from __future__ import annotations

import ast
import json

from .retrieval import DATASETS


BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def load_beir(name, split="test"):
    """Download (if needed) and load a BEIR dataset for a given split."""
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    ds_dir = DATASETS / name
    if not ds_dir.exists():
        beir_util.download_and_unzip(f"{BEIR_BASE}/{name}.zip", str(DATASETS))
    return GenericDataLoader(str(ds_dir)).load(split=split)


# ----------------------------------------------------------------------
# BioASQ.  The official corpus is gated, so we try the local directory
# first (if the user pre-downloaded it), then the HF BeIR mirror, then
# the public mini-BioASQ mirror.  The mini mirror is small and has no
# distractors -- this is the degenerate-corpus case discussed in the
# report.
# ----------------------------------------------------------------------

def _bioasq_local():
    from beir.datasets.data_loader import GenericDataLoader
    if not (DATASETS / "bioasq" / "corpus.jsonl").exists():
        return None
    return GenericDataLoader(str(DATASETS / "bioasq")).load(split="test")


def _bioasq_hf():
    """BeIR/bioasq HuggingFace mirror.

    Note: BeIR/bioasq was withdrawn from the Hub due to BioASQ's licensing
    terms, so this loader almost always fails now.  We try a couple of
    alternative repo paths in case any of them is still up; the chain
    falls through to the mini-BioASQ mirror otherwise.
    """
    from datasets import load_dataset

    candidates = [
        ("BeIR/bioasq", "BeIR/bioasq-qrels"),
        ("mteb/bioasq", "mteb/bioasq-qrels"),
    ]
    last_err = None
    for corpus_repo, qrels_repo in candidates:
        try:
            corpus_ds = load_dataset(corpus_repo, "corpus", split="corpus")
            queries_ds = load_dataset(corpus_repo, "queries", split="queries")
            qrels_ds = load_dataset(qrels_repo, split="test")
        except Exception as e:
            last_err = e
            continue

        corpus = {str(r["_id"]): {"title": r.get("title", "") or "",
                                     "text": r.get("text", "") or ""}
                   for r in corpus_ds}
        queries = {str(r["_id"]): r["text"] for r in queries_ds}
        qrels = {}
        for r in qrels_ds:
            qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
        return corpus, queries, qrels

    raise last_err if last_err else RuntimeError("no HF BioASQ mirror responded")


def _bioasq_mini():
    """Public mini-BioASQ.  Only ~28K passages -- the qrel union exhausts
    the pool, so the resulting subset has no distractors."""
    from datasets import load_dataset

    last_err = None
    for repo in ("enelpol/rag-mini-bioasq", "rag-datasets/rag-mini-bioasq"):
        try:
            corpus_ds = load_dataset(repo, "text-corpus", split="passages")
            try:
                qa_ds = load_dataset(repo, "question-answer-passages", split="test")
            except Exception:
                qa_ds = load_dataset(repo, "question-answer-passages", split="train")
        except Exception as e:
            last_err = e
            continue

        corpus = {}
        for r in corpus_ds:
            text = r.get("passage", "") or ""
            if text and text != "nan":
                corpus[str(r.get("id"))] = {"title": "", "text": text}
        if not corpus:
            continue

        queries, qrels = {}, {}
        for i, r in enumerate(qa_ds):
            q = r.get("question", "") or ""
            rel = r.get("relevant_passage_ids", []) or []
            if isinstance(rel, str):
                # Some mirrors store the list as a JSON-ish string.
                try:
                    rel = json.loads(rel)
                except (ValueError, TypeError):
                    try:
                        rel = ast.literal_eval(rel)
                    except (ValueError, SyntaxError):
                        rel = []
            rel_ids = [str(d) for d in rel if str(d) in corpus]
            if not q or not rel_ids:
                continue
            qid = str(r.get("id", i))
            queries[qid] = q
            qrels[qid] = {d: 1 for d in rel_ids}

        if queries:
            return corpus, queries, qrels

    if last_err is not None:
        raise last_err
    raise RuntimeError("mini-BioASQ mirrors loaded but produced no QA triples")


def load_bioasq():
    """Try the three sources in order; return ((corpus, queries, qrels), label).

    Heads-up: BeIR/bioasq was withdrawn from HuggingFace, and the BEIR zip
    distribution at UKP-Darmstadt has been a placeholder HTML for years.
    In practice the chain almost always lands on rag-mini-bioasq, whose
    corpus is small enough that the qrel union exhausts it -- the
    resulting subset has zero distractors and BM25 is inflated by roughly
    +0.30 nDCG@10.  See main.tex Section 4.2 for the caveat.

    To get a non-degenerate run, register at http://bioasq.org/, download
    the BEIR-format release, and place it under notebooks/datasets/bioasq/
    (the local loader is tried first).
    """
    for label, fn in [("local BEIR directory", _bioasq_local),
                       ("HuggingFace BeIR/bioasq", _bioasq_hf),
                       ("rag-mini-bioasq fallback", _bioasq_mini)]:
        try:
            res = fn()
        except Exception as e:
            print(f"  [{label}] failed: {type(e).__name__}: {e}")
            continue
        if res is None:
            continue
        if label == "rag-mini-bioasq fallback":
            print("\n  ⚠  fell back to rag-mini-bioasq -- the resulting subset")
            print("     will have zero distractors (BM25 inflated by ~+0.30 nDCG@10).")
            print("     For a fair comparison, register at http://bioasq.org/ and")
            print("     unpack the BEIR-format release under notebooks/datasets/bioasq/.\n")
        return res, label
    raise RuntimeError(
        "All BioASQ sources failed.  Register at http://bioasq.org/ and "
        "place the BEIR-format release in datasets/bioasq/.")
