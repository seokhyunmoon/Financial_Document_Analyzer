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
import torch
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)

_MODEL_CACHE = {
    "model": None,
    "name": None,
    "device": None,
}


def _text_for_embedding(chunk):
    """Prepare text from a chunk for embedding generation.
    Args:
        chunk: A chunk dictionary containing text and optional section title.
        
    Returns:
        A single string combining section title and text.
    """
    
    parts = []
    if chunk.get("section_title"):
        parts.append(chunk["section_title"])
    parts.append(chunk.get("text") or "")
    return "\n".join(part for part in parts if part)


def _get_model(model_name: str, device: str) -> SentenceTransformer:
    """Load or reuse a SentenceTransformer on the requested device."""
    cached = _MODEL_CACHE["model"]
    if cached is not None and _MODEL_CACHE["name"] == model_name and _MODEL_CACHE["device"] == device:
        return cached

    if cached is not None and device == "cuda":
        torch.cuda.empty_cache()

    logger.info(f"[INFO] Loading embedding model {model_name} on {device.upper()} ...")
    model = SentenceTransformer(model_name, device=device)
    _MODEL_CACHE.update({"model": model, "name": model_name, "device": device})
    return model


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate dense embeddings for document chunks.

    Args:
        chunks: List of chunk dictionaries containing text.

    Returns:
        Same list where each chunk has an added ``embedding`` vector.
    """
    
    # 1) load configuration
    cfg = load_config()
    esec = get_section(cfg, "embedding")
    
    model_name = esec.get("model_name", "Qwen/Qwen3-Embedding-4B")
    batch_size = int(esec.get("batch_size", 8))
    normalize_embeddings = bool(esec.get("normalize_embeddings", True))

    # 2) etermine device
    device = "cuda" if cuda.is_available() else "cpu"
    logger.info(f"[INFO] Using {device.upper()} device for embedding generation.")
    
    # 3) load model (reuse cached instance when possible)
    try:
        model = _get_model(model_name, device)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model {model_name}: {e}")
        raise

    # 4) generate embeddings
    texts = [_text_for_embedding(c) for c in chunks]
    logger.info(f"[INFO] Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=False,
        batch_size=batch_size,
        device=device,
    )

    # 5) normalize if required
    if normalize_embeddings:
        embeddings = np.array([v / np.linalg.norm(v) for v in embeddings])

    # 6) attach embeddings to chunks
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    dim = int(len(embeddings[0]))
    logger.info(f"[OK] Embeddings generated: {len(chunks)} items, dim={len(embeddings[0])}")
    
    vector_dim = esec.get("vector_dimension", 0)
    if vector_dim != 0 and vector_dim != dim:
        logger.warning(f"[WARN] Configured vector_dimension {vector_dim} does not match actual dimension {dim}.")
    
    return chunks
