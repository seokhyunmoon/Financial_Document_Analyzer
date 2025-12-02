"""Chunking helpers."""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
import tiktoken
from tqdm import tqdm
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)

# Types emitted by elements.py (raw Unstructured categories)
TITLE_TYPES: Set[str] = {"title"}
BODY_TYPES: Set[str] = {
    "narrativetext",
    "listitem",
    "uncategorizedtext",
    "formula",
    "codesnippet",
    "address",
    "emailaddress",
    "image",
    "figurecaption",
}
TABLE_TYPES: Set[str] = {"table"}
NOISE_TYPES: Set[str] = {"header", "footer", "pagenumber", "pagebreak"}


def _token_len(encoder, text: str) -> int:
    return len(encoder.encode(text or ""))


def merge_elements_to_chunks(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge raw elements into retrieval-sized chunks.

    Args:
        elements: Element dictionaries emitted by ``extract_elements`` (holding
            ``source_doc``, ``type``, ``text``, and ``page`` info).

    Returns:
        A list of chunk dictionaries ready for embedding / retrieval.
    """
    
    #load config
    cfg = load_config()
    chunk_cfg = get_section(cfg, "chunking")
    max_tokens = int(chunk_cfg.get("max_tokens", 512))
    encoder = tiktoken.get_encoding("cl100k_base")

    #initialize state
    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_indices: List[int] = []
    current_tokens = 0
    current_section: Optional[str] = None
    chunk_id = 1

    def flush_text_chunk() -> None:
        """Flush the current text chunk to the chunks list."""

        nonlocal current_chunk, current_indices, current_tokens, chunk_id
        if not current_chunk:
            return
        merged = " ".join(el["text"].strip() for el in current_chunk if el.get("text"))
        head = current_chunk[0]
        page = head.get("page")
        chunk = {
            "source_doc": head.get("source_doc"),
            "chunk_id": chunk_id,
            "type": "text",
            "text": merged.strip(),
            "page_start": page,
            "page_end": current_chunk[-1].get("page"),
            "source_elements": current_indices.copy(),
        }
        if current_section:
            chunk["section_title"] = current_section
        chunks.append(chunk)
        chunk_id += 1
        current_chunk = []
        current_indices = []
        current_tokens = 0

    for idx, el in enumerate(tqdm(elements, desc="Merging elements")):
        etype = (el.get("type") or "").lower()
        text = (el.get("text") or "").strip()

        if etype in NOISE_TYPES:
            flush_text_chunk()
            continue

        if etype in TITLE_TYPES:
            if text:
                current_section = text
            continue

        if etype in TABLE_TYPES:
            flush_text_chunk()
            chunk = {
                "source_doc": el.get("source_doc"),
                "chunk_id": chunk_id,
                "type": "table",
                "text": text,
                "page_start": el.get("page"),
                "page_end": el.get("page"),
                "source_elements": [idx],
            }
            if current_section:
                chunk["section_title"] = current_section
            table_html = el.get("table_as_html")
            if table_html:
                chunk["text_as_html"] = table_html
            chunks.append(chunk)
            chunk_id += 1
            continue

        if etype in BODY_TYPES or not etype:
            token_len = _token_len(encoder, text)
            if token_len > max_tokens:
                flush_text_chunk()
                chunk = {
                    "source_doc": el.get("source_doc"),
                    "chunk_id": chunk_id,
                    "type": "text",
                    "text": text,
                    "page_start": el.get("page"),
                    "page_end": el.get("page"),
                    "source_elements": [idx],
                }
                if current_section:
                    chunk["section_title"] = current_section
                chunks.append(chunk)
                chunk_id += 1
                continue

            if current_tokens + token_len > max_tokens:
                flush_text_chunk()

            current_chunk.append(el)
            current_indices.append(idx)
            current_tokens += token_len
            continue

    flush_text_chunk()
    logger.info(f"[INFO] Merged into {len(chunks)} chunks from {len(elements)} elements")
    return chunks
