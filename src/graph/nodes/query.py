# src/graph/nodes/query.py
"""
query.py
-----------
This module defines nodes for querying embeddings in the graph.
"""

from typing import List
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from torch import cuda
import numpy as np
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)

@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    """Loads and caches the SentenceTransformer model.
    
    Args:
        model_name: The name of the embedding model.
        
    Returns:
        An instance of SentenceTransformer.
    """
    
    # Load configuration
    device = "cuda" if cuda.is_available() else "cpu"
    
    try:
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        raise

def query_embeddings(question: str) -> List[float]:
    """Generates an embedding for a given query string.

    Args:
        question: The input string (question) to embed.

    Returns:
        A list of floats representing the normalized embedding vector for the query.
    """
    
    # 1) Config
    cfg = load_config()
    esec = get_section(cfg, "embedding")
    model_name = esec.get("model_name", "Qwen/Qwen3-Embedding-4B")
    normalize_embeddings = bool(esec.get("normalize_embeddings", False))

    # 2) Get model
    model = _get_model(model_name)
    question_vector = model.encode(question, normalize_embeddings=False, show_progress_bar=False)
    
    if normalize_embeddings:
        question_vector = question_vector / np.linalg.norm(question_vector)
        
    logger.info(f"[INFO] Generated query embedding. normalized={normalize_embeddings}, dimension={len(question_vector)}.")
    
    return question_vector.tolist()
