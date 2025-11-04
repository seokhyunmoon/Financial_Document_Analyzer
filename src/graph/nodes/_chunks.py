# src/graph/nodes/chunks.py
"""
chunks.py
----------
Hybrid Structural Chunking module based on §3.4 of the paper (2402.05131v3)
and Unstructured's Element object model.

This version preserves Element metadata while forming semantically coherent
chunks separated by structural boundaries (titles, tables, page changes).
"""

from typing import List
from unstructured.documents.elements import Element, CompositeElement
from unstructured.chunking.title import chunk_by_title
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)

def chunk_elements(elements: List[Element]) -> List[Element]:
    """
    Chunk Element objects following structural boundaries defined in §3.4:
      - Titles always start new chunks.
      - Tables are standalone chunks.
      - Page changes trigger chunk split.
      - Text accumulates until min_chars threshold is reached.

    Args:
        elements (List[Element]): List of Elements extracted from PDF.

    Returns:
        List[Element]: List of CompositeElements (chunked sections).
    """
    cfg = load_config()
    csec = get_section(cfg, "chunking")

    min_chars   = int(csec.get("min_chars", 2048))
    max_chars   = int(csec.get("max_chars", 4096))
    overlap     = int(csec.get("overlap", 128))
    combine_n   = min(int(csec.get("combine_under_n_chars", min_chars)), max_chars)

    logger.info(f"[INFO] Chunking {len(elements)} elements (min_chars={min_chars}, max_chars={max_chars}) ...")

    chunks: List[Element] = chunk_by_title(
        elements=elements,
        max_characters=max_chars,
        combine_text_under_n_chars=combine_n,
        new_after_n_chars=min_chars,
        overlap=overlap,
        include_orig_elements=False,    
    )

    for ch in chunks:
        md = getattr(ch, "metadata", None)
        if md is None:
            continue
        if hasattr(md, "orig_elements"):
            md.orig_elements = None
        if hasattr(md, "detection_class_prob"):
            md.detection_class_prob = None

    logger.info(f"[OK] Generated {len(chunks)} chunks from {len(elements)} elements")
    return chunks