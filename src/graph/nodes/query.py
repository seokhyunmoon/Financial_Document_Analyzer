# src/graph/nodes/query.py
"""
query.py
-----------
This module defines nodes for querying embeddings in the graph.
"""

from typing import List
from sentence_transformers import SentenceTransformer
from torch import cuda
import numpy as np
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)

def query_embeddings(question: str) -> List[float]:
    """Generates an embedding for a given query string.

    This function takes a user's question, processes it according to the
    configured embedding model settings (including prompt alignment and
    normalization), and returns its vector representation.

    Args:
        question: The input string (question) to embed.

    Returns:
        A list of floats representing the normalized embedding vector for the query.
    """
    
    # 1) Config
    cfg = load_config()
    esec = get_section(cfg, "embedding")
    model_name = esec.get("model_name", "Qwen/Qwen3-Embedding-4B")
    normalize_embeddings = bool(esec.get("normalize_embeddings", True))
    use_prompts = bool(esec.get("use_prompts", True))
    doc_key   = (esec.get("prompts", {}) or {}).get("doc", "passage")
    query_key = (esec.get("prompts", {}) or {}).get("query", "query")

    # 2) Device
    device = "cuda" if cuda.is_available() else "cpu"
    logger.info(f"[INFO] Using {device.upper()} device for query embedding.")

    # 3) Load model
    model = SentenceTransformer(model_name, device=device)

    # 4) Prompt alignment 
    prompt_name = None
    if use_prompts and hasattr(model, "prompts") and isinstance(model.prompts, dict):
        has_doc = doc_key in model.prompts
        has_qry = query_key in model.prompts
        if has_doc and has_qry:
            prompt_name = query_key
            logger.info(f"[INFO] Using query prompt_name='{prompt_name}'")
        else:
            logger.warning(
                f"[WARN] prompts missing (doc='{doc_key}' exist={has_doc}, "
                f"query='{query_key}' exist={has_qry}); using no prompts."
            )

    # 5) Encode 
    encode_kwargs = dict(
        device=device,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    if prompt_name:
        encode_kwargs["prompt_name"] = prompt_name

    vec = model.encode([question], **encode_kwargs)[0]

    # 6) Normalize (cosine)
    arr = np.asarray(vec, dtype=np.float32)
    if normalize_embeddings:
        n = np.linalg.norm(arr)
        if n > 0:
            arr = arr / n

    logger.info("[INFO] Generated query embedding.")
    return arr.tolist()