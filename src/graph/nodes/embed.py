# src/graph/nodes/index_dense.py
"""
index_dense.py
--------------
Generate dense embeddings for document chunks using Hugging Face SentenceTransformer.
Supports Qwen/Qwen3-Embedding-4B model.
"""

from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


def generate_embeddings(
    chunks: List[Dict[str, Any]],
    model_name: str = "Qwen/Qwen3-Embedding-4B",
    batch_size: int = 8,
    normalize_embeddings: bool = True,
    device: str = "cpu"
) -> List[Dict[str, Any]]:
    """
    Generate dense embeddings for document chunks.

    Args:
        chunks (List[Dict[str, Any]]): List of chunk dictionaries.
        model_name (str): HuggingFace model ID for SentenceTransformer.
        batch_size (int): Batch size for encoding.
        normalize_embeddings (bool): Whether to L2-normalize embeddings before storage.

    Returns:
        List[Dict[str, Any]]: Same chunks, each with an 'embedding' key added.
    """
    print(f"[INFO] Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    texts = [c["text"] for c in chunks]
    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")

    # Step 1 — raw encoding (keep internal normalization off for pipeline control)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=False,
        device=device
    )

    # Step 2 — optional pipeline-level normalization
    if normalize_embeddings:
        embeddings = np.array([v / np.linalg.norm(v) for v in embeddings])

    # Step 3 — attach embeddings back to chunks
    for i, emb in enumerate(embeddings):
        chunks[i]["embedding"] = emb.tolist()

    print(f"[OK] Embeddings generated: {len(chunks)} items, dim={len(embeddings[0])}")
    return chunks
