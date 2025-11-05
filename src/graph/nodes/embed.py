# src/graph/nodes/embed.py
"""
embed.py
--------
This module provides functionality to generate dense embeddings for document chunks
using a Hugging Face SentenceTransformer model, as configured in default.yaml.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from torch import cuda
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)


def generate_embeddings(
    chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate dense embeddings for document chunks.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries.

    Returns:
        List[Dict[str, Any]]: Same chunks, each with an 'embedding' key added.
    """
    
    # load configuration
    cfg = load_config()
    esec = get_section(cfg, "embedding")
    
    model_name = esec.get("model_name", "Qwen/Qwen3-Embedding-4B")
    batch_size = int(esec.get("batch_size", 8))
    normalize_embeddings = bool(esec.get("normalize_embeddings", True))

    if cuda.is_available():
        device = "cuda"
        logger.info(f"[INFO] Using CUDA device for embedding generation.")
    else:
        device = "cpu"
        logger.info(f"[INFO] Using CPU device for embedding generation.")
    
    logger.info(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    logger.info(f"[INFO] Generating embeddings for {len(texts)} chunks...")

    #raw encoding (keep internal normalization off for pipeline control)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        device=device
    )

    # optional pipeline-level normalization
    if normalize_embeddings:
        embeddings = np.array([v / np.linalg.norm(v) for v in embeddings])

    # attach embeddings back to chunks
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    logger.info(f"[OK] Embeddings generated: {len(chunks)} items, dim={len(embeddings[0])}")
    return chunks
