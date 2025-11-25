"""
rerank.py
---------
Lightweight reranker that re-sorts retrieved hits by cosine similarity
between the question embedding and chunk embeddings using the same
SentenceTransformer model as the retriever.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import numpy as np

from utils.logger import get_logger
from utils.config import load_config, get_section
from graph.nodes.query import _get_model

logger = get_logger(__name__)


def rerank_hits(
    question_vector: Optional[List[float]],
    hits: List[Dict[str, Any]],
    topk: Optional[int] = None,
    question: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if not hits or not question_vector:
        return hits

    cfg = load_config()
    qsec = get_section(cfg, "qa")
    rerank_cfg = get_section(qsec, "rerank")
    topk = min(topk or rerank_cfg.get("topk", len(hits)), len(hits))

    embedding_cfg = get_section(cfg, "embedding")
    model_name = embedding_cfg.get("model_name", "Qwen/Qwen3-Embedding-4B")
    normalize = bool(embedding_cfg.get("normalize_embeddings", False))

    model = _get_model(model_name)
    texts = [h.get("text", "") for h in hits]
    if not any(texts):
        return hits

    chunk_vectors = model.encode(
        texts,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )
    qvec = np.array(question_vector, dtype=float)
    if normalize:
        norm = np.linalg.norm(qvec)
        if norm:
            qvec = qvec / norm
    scores = chunk_vectors @ qvec

    reranked = sorted(zip(scores, hits), key=lambda x: x[0], reverse=True)
    top_hits = [hit for _, hit in reranked[:topk]]
    logger.info(f"[OK] Reranked hits (top {topk})")
    return top_hits
