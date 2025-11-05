# src/graph/nodes/chunks.py
"""
chunks.py
---------
This module provides functionality to merge extracted elements into structural chunks
based on predefined rules and configuration, preparing them for further processing.
"""

from typing import List, Dict, Any
from tqdm import tqdm
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)


def merge_elements_to_chunks(
    elements: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Description:
        Merge extracted elements into structural chunks following the paper's §3.4 rules (papers/2402.05131v3.pdf).

    Args:
        elements (List[Dict[str, Any]]): List of elements from Unstructured (sorted by page and order).

    Returns:
        List[Dict[str, Any]]: List of merged chunk dictionaries with metadata.
    """

    # load configuration
    cfg = load_config()
    csec = get_section(cfg, "chunking")
    min_chars = int(csec.get("min_chars", 2048))
    
    # merging logic
    boundary_types = ["title", "table"]
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_length: int = 0
    chunk_id: int = 1

    for i, el in enumerate(tqdm(elements, desc="Merging elements")):
        el_type = el.get("type", "text")

        # boundary: title or table → close previous chunk
        if el_type in boundary_types:
            if current_chunk:
                merged_text = " ".join(e["text"].strip() for e in current_chunk if e.get("text"))
                head = current_chunk[0]
                chunks.append({
                    "source_doc": head.get("source_doc"),
                    "doc_id": head.get("doc_id"),
                    "chunk_id": chunk_id,
                    "type": "text",
                    "text": merged_text.strip(),
                    "page_start": current_chunk[0]["page"],
                    "page_end": current_chunk[-1]["page"],
                    "source_elements": [j for j in range(i - len(current_chunk), i)],
                })
                chunk_id += 1
                current_chunk = []
                current_length = 0

            # Add boundary element (title/table) as its own chunk
            chunks.append({
                "source_doc": el.get("source_doc"),
                "doc_id": el.get("doc_id"),
                "chunk_id": chunk_id,
                "type": el_type,
                "text": el.get("text", "").strip(),
                "page_start": el.get("page"),
                "page_end": el.get("page"),
                "source_elements": [i],
            })
            chunk_id += 1
            continue

        # text element merging
        current_chunk.append(el)
        current_length += len(el.get("text", ""))

        # if enough length reached → finalize chunk
        if current_length >= min_chars:
            merged_text = " ".join(e["text"].strip() for e in current_chunk if e.get("text"))
            head = current_chunk[0]
            chunks.append({
                "source_doc": head.get("source_doc"),
                "doc_id": head.get("doc_id"),
                "chunk_id": chunk_id,
                "type": "text",
                "text": merged_text.strip(),
                "page_start": current_chunk[0]["page"],
                "page_end": current_chunk[-1]["page"],
                "source_elements": [j for j in range(i - len(current_chunk) + 1, i + 1)],
            })
            chunk_id += 1
            current_chunk = []
            current_length = 0

    # finalize remaining chunk
    if current_chunk:
        merged_text = " ".join(e["text"].strip() for e in current_chunk if e.get("text"))
        head = current_chunk[0]
        chunks.append({
            "source_doc": head.get("source_doc"),
            "doc_id": head.get("doc_id"),
            "chunk_id": chunk_id,
            "type": "text",
            "text": merged_text.strip(),
            "page_start": current_chunk[0]["page"],
            "page_end": current_chunk[-1]["page"],
            "source_elements": [len(elements) - len(current_chunk) + j for j in range(len(current_chunk))],
        })

    logger.info(f"[INFO] Merged into {len(chunks)} chunks from {len(elements)} elements")
    return chunks