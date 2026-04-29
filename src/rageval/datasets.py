"""Dataset loaders.

Wraps BEIR's ``GenericDataLoader`` for the four BEIR-distributed
datasets (NFCorpus, TREC-COVID, SciFact, ArguAna) and provides a
multi-source loader for BioASQ.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict, Tuple

from .utils import DATASETS_DIR


_BEIR_BASE = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"


def load_beir(name: str, split: str = "test") -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[str, str],
    Dict[str, Dict[str, int]],
]:
    """Download (if needed) and load a BEIR dataset for the requested split.

    Parameters
    ----------
    name : one of ``nfcorpus``, ``trec-covid``, ``scifact``, ``arguana``.
    split : ``train``, ``dev`` or ``test``.

    Returns
    -------
    (corpus, queries, qrels) in BEIR's dict-of-dicts format.
    """
    from beir import util as beir_util
    from beir.datasets.data_loader import GenericDataLoader

    ds_dir = DATASETS_DIR / name
    if not ds_dir.exists():
        beir_util.download_and_unzip(f"{_BEIR_BASE}/{name}.zip", str(DATASETS_DIR))
    return GenericDataLoader(str(ds_dir)).load(split=split)


# ---------------------------------------------------------------------------
# BioASQ multi-source loader.  The official BioASQ-BEIR mirror is gated, so
# the notebook's strategy is to try a local directory, the BeIR HF mirror,
# and finally the public ``rag-mini-bioasq`` mirror.
# ---------------------------------------------------------------------------


def _load_bioasq_local() -> Tuple[dict, dict, dict] | None:
    """Local BEIR-format directory (only if the user pre-downloaded the
    official corpus)."""
    from beir.datasets.data_loader import GenericDataLoader

    bioasq_dir = DATASETS_DIR / "bioasq"
    if not (bioasq_dir / "corpus.jsonl").exists():
        return None
    return GenericDataLoader(str(bioasq_dir)).load(split="test")


def _load_bioasq_hf_beir() -> Tuple[dict, dict, dict] | None:
    """``BeIR/bioasq`` HuggingFace mirror."""
    from datasets import load_dataset

    corpus_ds = load_dataset("BeIR/bioasq", "corpus", split="corpus")
    queries_ds = load_dataset("BeIR/bioasq", "queries", split="queries")
    qrels_ds = load_dataset("BeIR/bioasq-qrels", split="test")

    corpus: Dict[str, Dict[str, str]] = {}
    for r in corpus_ds:
        corpus[str(r["_id"])] = {
            "title": r.get("title", "") or "",
            "text": r.get("text", "") or "",
        }
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: Dict[str, Dict[str, int]] = {}
    for r in qrels_ds:
        qid, did, score = str(r["query-id"]), str(r["corpus-id"]), int(r["score"])
        qrels.setdefault(qid, {})[did] = score
    return corpus, queries, qrels


def _load_bioasq_mini() -> Tuple[dict, dict, dict]:
    """Public ``rag-mini-bioasq`` mirror.  Used as final fallback because
    its corpus is much smaller (which causes the no-distractor caveat
    discussed in the report).
    """
    from datasets import load_dataset

    last_exc = None
    for repo in ("enelpol/rag-mini-bioasq", "rag-datasets/rag-mini-bioasq"):
        try:
            corpus_ds = load_dataset(repo, "text-corpus", split="passages")
            try:
                qa_ds = load_dataset(repo, "question-answer-passages", split="test")
            except Exception:
                qa_ds = load_dataset(repo, "question-answer-passages", split="train")
        except Exception as exc:
            last_exc = exc
            continue

        corpus: Dict[str, Dict[str, str]] = {}
        for r in corpus_ds:
            did = str(r.get("id"))
            text = r.get("passage", "") or ""
            if not text or text == "nan":
                continue
            corpus[did] = {"title": "", "text": text}
        if not corpus:
            continue

        queries: Dict[str, str] = {}
        qrels: Dict[str, Dict[str, int]] = {}
        for i, r in enumerate(qa_ds):
            qid = str(r.get("id", i))
            q = r.get("question", "") or ""
            rel = r.get("relevant_passage_ids", []) or []
            if isinstance(rel, str):
                try:
                    rel = json.loads(rel)
                except (ValueError, TypeError):
                    try:
                        rel = ast.literal_eval(rel)
                    except (ValueError, SyntaxError):
                        rel = []
            rel_ids = [str(d) for d in rel if str(d) in corpus]
            if not rel_ids or not q:
                continue
            queries[qid] = q
            qrels[qid] = {d: 1 for d in rel_ids}

        if queries:
            return corpus, queries, qrels

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("rag-mini-bioasq mirrors loaded but produced no QAP triples")


def load_bioasq() -> Tuple[
    Tuple[dict, dict, dict],  # (corpus, queries, qrels)
    str,                      # source label
]:
    """Try the three BioASQ sources in order; return the first that loads."""
    for label, fn in [
        ("local BEIR directory", _load_bioasq_local),
        ("HuggingFace BeIR/bioasq", _load_bioasq_hf_beir),
        ("rag-mini-bioasq fallback", _load_bioasq_mini),
    ]:
        try:
            res = fn()
        except Exception as exc:
            print(f"  [{label}] failed: {type(exc).__name__}: {exc}")
            continue
        if res is None:
            continue
        return res, label

    raise RuntimeError(
        "All BioASQ loaders failed.  Register at http://bioasq.org/ and "
        "place the BEIR-format release in datasets/bioasq/."
    )
