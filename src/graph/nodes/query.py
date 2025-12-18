# src/graph/nodes/query.py
"""
query.py
-----------
This module defines nodes for querying embeddings in the graph.
"""

from typing import List
from functools import lru_cache
import threading
from sentence_transformers import SentenceTransformer
from torch import cuda
import numpy as np
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)
_MODEL_LOCK = threading.Lock()

@lru_cache(maxsize=1)
def _get_model(model_name: str) -> SentenceTransformer:
    """Load and cache the SentenceTransformer model.

    Args:
        model_name: Name of the embedding model.

    Returns:
        Loaded SentenceTransformer instance.
    """
    
    # Load configuration
    device = "cuda" if cuda.is_available() else "cpu"
    
    # Guard first load to avoid parallel reloads under concurrency
    with _MODEL_LOCK:
        try:
            model = SentenceTransformer(model_name, device=device)
            return model
        except Exception as e:
            raise

def query_embeddings(question: str) -> List[float]:
    """Generate an embedding for a query string.

    Args:
        question: Input question text.

    Returns:
        Normalized embedding vector as a list of floats.
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
