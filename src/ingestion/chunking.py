from __future__ import annotations
from typing import List, Dict, Any, Optional, Set
import tiktoken
from tqdm import tqdm
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)

# Types emitted by elements.py (raw Unstructured categories)
TITLE_TYPES: Set[str] = {"title", "header"}
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
    "footer", 
    "pagenumber", 
    "pagebreak"
}
TABLE_TYPES: Set[str] = {"table"}
NOISE_TYPES: Set[str] = {}


def _token_len(encoder, text: str) -> int:
    """Return the token length of ``text`` under the provided encoder."""
    return len(encoder.encode(text or ""))


def merge_elements_to_chunks(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge raw elements into retrieval-sized chunks that honor section titles.

    Args:
        elements (List[Dict[str, Any]]): Sequence of element dictionaries emitted
            by :func:`extract_elements`.

    Returns:
        List[Dict[str, Any]]: Chunk dictionaries ready for downstream embedding
        and retrieval steps.
    """

    cfg = load_config()
    chunk_cfg = get_section(cfg, "chunking")
    chunk_mode = str(chunk_cfg.get("mode", "tokens")).lower()
    if chunk_mode == "chars":
        max_len = int(chunk_cfg.get("max_char", 2048))
        encoder = None
    else:
        chunk_mode = "tokens"
        max_len = int(chunk_cfg.get("max_tokens", 128))
        encoder = tiktoken.get_encoding("cl100k_base")

    chunks: List[Dict[str, Any]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_indices: List[int] = []
    current_len = 0
    current_titles: List[str] = []
    active_section: Optional[str] = None
    chunk_section_at_start: Optional[str] = None
    chunk_id = 1

    def _consume_titles() -> Optional[str]:
        """Convert buffered titles into a single section string.

        Returns:
            The resolved section title if any titles were buffered, otherwise
            the last active section.
        """
        nonlocal current_titles, active_section
        if current_titles:
            joined = " ".join(t.strip() for t in current_titles if t.strip())
            if joined:
                active_section = joined
            current_titles = []
        return active_section

    def flush_text_chunk() -> None:
        """Emit the accumulated text chunk (if any) to the output list."""

        nonlocal current_chunk, current_indices, current_len, chunk_id, chunk_section_at_start
        if not current_chunk:
            return
        merged = " ".join(el.get("text", "").strip() for el in current_chunk if el.get("text"))
        head = current_chunk[0]
        chunk = {
            "source_doc": head.get("source_doc"),
            "chunk_id": chunk_id,
            "type": "text",
            "text": merged.strip(),
            "page_start": head.get("page"),
            "page_end": current_chunk[-1].get("page"),
            "source_elements": current_indices.copy(),
        }
        if chunk_section_at_start:
            chunk["section_title"] = chunk_section_at_start
        chunks.append(chunk)
        chunk_id += 1
        current_chunk = []
        current_indices = []
        current_len = 0
        chunk_section_at_start = None

    for idx, el in enumerate(tqdm(elements, desc="Merging elements")):
        etype = (el.get("type") or "").lower()
        text = (el.get("text") or "").strip()

        if etype in NOISE_TYPES:
            flush_text_chunk()
            continue

        if etype in TITLE_TYPES:
            if current_chunk:
                flush_text_chunk()
            if text:
                current_titles.append(text)
            continue

        if etype in TABLE_TYPES:
            flush_text_chunk()
            section_for_table = _consume_titles() or chunk_section_at_start
            chunk = {
                "source_doc": el.get("source_doc"),
                "chunk_id": chunk_id,
                "type": "table",
                "text": text,
                "page_start": el.get("page"),
                "page_end": el.get("page"),
                "source_elements": [idx],
            }
            if section_for_table:
                chunk["section_title"] = section_for_table
            table_html = el.get("table_as_html")
            if table_html:
                chunk["text_as_html"] = table_html
            chunks.append(chunk)
            chunk_id += 1
            continue

        if etype in BODY_TYPES or not etype:
            unit_len = len(text) if chunk_mode == "chars" else _token_len(encoder, text)
            if not current_chunk:
                chunk_section_at_start = _consume_titles()

            if unit_len > max_len:
                flush_text_chunk()
                section_for_segment = _consume_titles() or chunk_section_at_start
                chunk = {
                    "source_doc": el.get("source_doc"),
                    "chunk_id": chunk_id,
                    "type": "text",
                    "text": text,
                    "page_start": el.get("page"),
                    "page_end": el.get("page"),
                    "source_elements": [idx],
                }
                if section_for_segment:
                    chunk["section_title"] = section_for_segment
                chunks.append(chunk)
                chunk_id += 1
                continue

            if current_len + unit_len > max_len:
                flush_text_chunk()
                if not chunk_section_at_start:
                    chunk_section_at_start = _consume_titles()

            current_chunk.append(el)
            current_indices.append(idx)
            current_len += unit_len
            continue

    flush_text_chunk()
    logger.info(f"[INFO] Merged into {len(chunks)} chunks from {len(elements)} elements")
    return chunks
