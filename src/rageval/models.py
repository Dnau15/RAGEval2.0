"""Lazy model registry.

Loading every encoder up front exhausts a Colab T4.  These helpers cache
each model in a module-level dict and provide an ``unload`` helper that
moves the model back to CPU and frees the CUDA cache.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .utils import cuda_gc, get_device


_MODELS: Dict[str, object] = {}


def get_minilm():
    if "minilm" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["minilm"] = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2", device=get_device())
    return _MODELS["minilm"]


def get_bge():
    if "bge" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["bge"] = SentenceTransformer(
            "BAAI/bge-small-en-v1.5", device=get_device())
    return _MODELS["bge"]


def get_e5():
    if "e5" not in _MODELS:
        from sentence_transformers import SentenceTransformer
        _MODELS["e5"] = SentenceTransformer(
            "intfloat/e5-small-v2", device=get_device())
    return _MODELS["e5"]


def get_splade():
    if "splade" not in _MODELS:
        import torch
        from transformers import AutoTokenizer, AutoModelForMaskedLM

        name = "naver/splade-cocondenser-ensembledistil"
        device = get_device()
        dt = torch.float16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained(name)
        mod = AutoModelForMaskedLM.from_pretrained(
            name, torch_dtype=dt).to(device).eval()
        _MODELS["splade"] = (tok, mod)
    return _MODELS["splade"]


def get_medcpt():
    if "medcpt" not in _MODELS:
        import torch
        from transformers import AutoTokenizer, AutoModel

        device = get_device()
        dt = torch.float16 if device == "cuda" else torch.float32
        qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        qm = AutoModel.from_pretrained(
            "ncbi/MedCPT-Query-Encoder", torch_dtype=dt).to(device).eval()
        at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        am = AutoModel.from_pretrained(
            "ncbi/MedCPT-Article-Encoder", torch_dtype=dt).to(device).eval()
        _MODELS["medcpt"] = (qt, qm, at, am)
    return _MODELS["medcpt"]


def get_bge_reranker():
    if "bge_reranker" not in _MODELS:
        import torch
        from sentence_transformers import CrossEncoder

        device = get_device()
        kwargs: dict = {}
        if device == "cuda":
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        _MODELS["bge_reranker"] = CrossEncoder(
            "BAAI/bge-reranker-base", device=device, max_length=512, **kwargs)
    return _MODELS["bge_reranker"]


def get_medcpt_ce():
    if "medcpt_ce" not in _MODELS:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = get_device()
        dt = torch.float16 if device == "cuda" else torch.float32
        tok = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        mod = AutoModelForSequenceClassification.from_pretrained(
            "ncbi/MedCPT-Cross-Encoder", torch_dtype=dt).to(device).eval()
        _MODELS["medcpt_ce"] = (tok, mod)
    return _MODELS["medcpt_ce"]


def unload(*names: str) -> None:
    """Drop one or more cached models from VRAM."""
    for n in names:
        entry = _MODELS.pop(n, None)
        if entry is None:
            continue
        items = entry if isinstance(entry, tuple) else (entry,)
        for x in items:
            try:
                if hasattr(x, "to"):
                    x.to("cpu")
            except Exception:
                pass
        del entry, items
    cuda_gc()
