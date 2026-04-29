"""Phase B.4: downstream PubMedQA-labeled with flan-t5-base.

Builds the PubMedQA pseudo-corpus, retrieves top-5 evidence with three
retrievers (BM25, BGE-small, BGE-small+BGE-reranker), and prompts
flan-t5-base for a yes/no/maybe answer.

Llama-3.2-3B-Instruct is gated; if the user is not logged in, the
script silently falls back to flan-t5-base only and writes the rows
that succeeded.

Produces ``feedback2/tables/mirage_accuracy.csv``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rageval.models import get_bge, get_bge_reranker, unload
from rageval.utils import (FEEDBACK2_DIR, bm25_run, cross_encoder_rerank,
                            cuda_gc, dense_run, get_device)


PROMPT = ("You are a biomedical question-answering assistant.\n"
           "Answer with one of: yes, no, maybe.\n"
           "Question: {q}\n"
           "Evidence:\n{ctx}\n"
           "Final answer:")


def _normalise(text: str) -> str:
    t = text.strip().lower()
    for label in ("yes", "no", "maybe"):
        if re.search(rf"\b{label}\b", t):
            return label
    return "unknown"


def _format_prompt(q: str, passages: List[str]) -> str:
    ctx = "\n".join(f"[{i+1}] {p}" for i, p in enumerate(passages or []))
    if not ctx:
        ctx = "(no retrieved evidence)"
    return PROMPT.format(q=q, ctx=ctx)


def _build_pubmedqa() -> tuple:
    """Load PubMedQA-labeled and build a per-question pseudo-corpus."""
    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    rows = []
    docs: Dict[str, Dict[str, str]] = {}
    for r in ds:
        rows.append({
            "qid": str(r["pubid"]),
            "question": r["question"],
            "answer": r["final_decision"].lower().strip(),
            "context_passages": r["context"]["contexts"],
        })
        for ci, passage in enumerate(r["context"]["contexts"]):
            docs[f"{r['pubid']}_p{ci}"] = {
                "title": "", "text": passage,
                "owner_qid": str(r["pubid"]),
            }
    return rows, docs


def _gen_flan(tok, mod, prompt: str, max_new: int = 4) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True,
               max_length=1024).to(mod.device)
    with torch.no_grad():
        out = mod.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return _normalise(tok.decode(out[0], skip_special_tokens=True))


def _gen_llama(tok, mod, prompt: str, max_new: int = 8) -> str:
    msgs = [{"role": "user", "content": prompt}]
    inp = tok.apply_chat_template(msgs, return_tensors="pt",
                                    add_generation_prompt=True).to(mod.device)
    with torch.no_grad():
        out = mod.generate(inp, max_new_tokens=max_new, do_sample=False)
    txt = tok.decode(out[0, inp.shape[-1]:], skip_special_tokens=True)
    return _normalise(txt)


def main() -> None:
    print("Loading PubMedQA-labeled ...")
    pq, pq_corpus = _build_pubmedqa()
    print(f"  questions: {len(pq)}  passages indexed: {len(pq_corpus)}")
    pq_doc_ids = list(pq_corpus.keys())
    pq_doc_texts = [pq_corpus[d]["text"] for d in pq_doc_ids]

    queries = {r["qid"]: r["question"] for r in pq}

    # ---- Build retrieval contexts (top-5 evidence) ----------------------
    print("\nRetrieval contexts:")
    print("  BM25 ..."); top_bm = bm25_run(pq_doc_ids, pq_doc_texts, queries, top_k=5)
    print("  BGE-small ...")
    bge = get_bge()
    top_bge = dense_run(pq_doc_ids, pq_doc_texts, queries, bge, top_k=5)

    print("  BGE-small + BGE-reranker ...")
    reranker = get_bge_reranker()
    cuda_gc()
    first = dense_run(pq_doc_ids, pq_doc_texts, queries, bge, top_k=100)
    cand = {q: list(first[q].keys()) for q in queries}
    rer = cross_encoder_rerank(queries, pq_corpus, cand, reranker)
    top_rerank = {q: dict(list(rer[q].items())[:5]) for q in queries}
    cuda_gc()

    contexts = {
        "None (closed-book)": {r["qid"]: [] for r in pq},
        "BM25": {q: [pq_corpus[d]["text"] for d in top_bm[q]] for q in queries},
        "BGE-small": {q: [pq_corpus[d]["text"] for d in top_bge[q]] for q in queries},
        "BGE-small+BGE-reranker": {q: [pq_corpus[d]["text"] for d in top_rerank[q]]
                                     for q in queries},
    }

    # Free retrieval models before loading generators
    unload("bge", "bge_reranker")
    cuda_gc()

    # ---- Load generators ------------------------------------------------
    from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                                AutoModelForCausalLM, BitsAndBytesConfig)

    print("\nLoading flan-t5-base (fp16 on CUDA) ...")
    device = get_device()
    dt = torch.float16 if device == "cuda" else torch.float32
    ft_tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    ft_mod = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base", torch_dtype=dt).to(device)

    print("Loading Llama-3.2-3B-Instruct (4-bit) ...")
    llama_ok = False
    ll_tok = ll_mod = None
    try:
        bnb = BitsAndBytesConfig(load_in_4bit=True,
                                   bnb_4bit_compute_dtype=torch.float16)
        ll_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
        ll_mod = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            quantization_config=bnb, device_map="auto")
        llama_ok = True
    except Exception as exc:
        print(f"  Llama unavailable ({type(exc).__name__}): falling back to "
               f"flan-t5-base only.  Run `huggingface-cli login` and accept "
               f"the licence to enable Llama.")

    generators = [("flan-t5-base", ft_tok, ft_mod, _gen_flan)]
    if llama_ok:
        generators.append(("Llama-3.2-3B-Instruct", ll_tok, ll_mod, _gen_llama))

    # ---- Score every (retriever, generator) pair ------------------------
    rows = []
    for retr_name in ["None (closed-book)", "BM25", "BGE-small",
                       "BGE-small+BGE-reranker"]:
        for gen_name, tok, mod, gen_fn in generators:
            correct = total = 0
            for r in tqdm(pq, desc=f"{retr_name} | {gen_name}", leave=False):
                ctx = contexts[retr_name].get(r["qid"], [])
                ans = gen_fn(tok, mod, _format_prompt(r["question"], ctx))
                if ans == r["answer"]:
                    correct += 1
                total += 1
            acc = correct / total if total else 0.0
            rows.append({
                "Task": "PubMedQA-labeled", "Retriever": retr_name,
                "Generator": gen_name, "Accuracy": round(acc, 3),
                "Status": "done",
            })
            print(f"  {retr_name:<22} | {gen_name:<22} acc = {acc:.3f}")
            cuda_gc()

    pd.DataFrame(rows).to_csv(FEEDBACK2_DIR / "mirage_accuracy.csv", index=False)
    print(f"\nwrote {FEEDBACK2_DIR / 'mirage_accuracy.csv'}")


if __name__ == "__main__":
    main()
