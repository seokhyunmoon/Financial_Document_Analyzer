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

    # determine device
    device = "cuda" if cuda.is_available() else "cpu"
    logger.info(f"[INFO] Using {device.upper()} device for embedding generation.")
    
    # load model
    try:
        model = SentenceTransformer(model_name, device=device)
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model {model_name}: {e}")
        raise

    texts = [c["text"] for c in chunks]
    logger.info(f"[INFO] Generating embeddings for {len(texts)} chunks...")

    # check for prompt usage (config + model support)
    use_prompts = bool(esec.get("use_prompts", True))
    doc_key = esec.get("prompts", {}).get("doc", "passage")
    prompt_name = None
    if use_prompts and hasattr(model, "prompts"):
        if isinstance(getattr(model, "prompts"), dict) and doc_key in model.prompts:
            prompt_name = doc_key
            logger.info(f"[INFO] Using document prompt_name='{prompt_name}'")
        else:
            logger.warning(f"[WARN] Document prompt_name='{doc_key}' not found in model.prompts")
            prompt_name = None

    
    # kwargs
    encode_kwargs = dict(
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        device=device,
    )
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name
    #raw encoding (keep internal normalization off for pipeline control)    
    embeddings = model.encode(
        texts, **encode_kwargs
    )

    # optional pipeline-level normalization
    if normalize_embeddings:
        embeddings = np.array([v / np.linalg.norm(v) for v in embeddings])

    # attach embeddings back to chunks
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    dim = int(len(embeddings[0]))
    logger.info(f"[OK] Embeddings generated: {len(chunks)} items, dim={len(embeddings[0])}")
    vector_dim = esec.get("vector_dimension", 0)
    if vector_dim != 0 and vector_dim != dim:
        logger.warning(f"[WARN] Configured vector_dimension {vector_dim} does not match actual dimension {dim}.")
    
    return chunks
