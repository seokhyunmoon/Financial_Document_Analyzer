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

def query_embeddings(
    question: str,
) -> List[float]: 
    """
    Description:
        Generate embedding for the input question string.

    Args:
        question (str): The input question string.

    Returns:
        List[float]: The embedding vector for the question.
    """
    
    # Load configuration
    cfg = load_config()
    esec = get_section(cfg, "embedding")
    model_name = esec.get("model_name", "Qwen/Qwen3-Embedding-4B")
    normalize_embeddings = bool(esec.get("normalize_embeddings", True))
    
    # Determine device
    if cuda.is_available():
        device = "cuda"
        logger.info(f"[INFO] Using CUDA device for embedding generation.")
    else:
        device = "cpu"
        logger.info(f"[INFO] Using CPU device for embedding generation.")
    
    # Load model
    logger.info(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embedding for the question
    question_vector = model.encode([question], 
                                   device=device, 
                                   show_progress_bar=True, 
                                   normalize_embeddings=False
                                   )
    
    # Normalize if required
    if normalize_embeddings:
        question_vector = question_vector / np.linalg.norm(question_vector, axis=1, keepdims=True)
        
    # Convert to list and return
    logger.info(f"[INFO] Generated embedding for the question.")
    return question_vector[0].tolist()
    
    