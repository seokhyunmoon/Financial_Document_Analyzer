# src/graph/nodes/elements.py
"""
elements.py
-----------
This module handles the extraction and normalization of structural elements from PDF documents
using the Unstructured library, applying configuration-driven cleaning and mapping
to a simplified schema for downstream processing.
"""
import re
from typing import List, Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import replace_unicode_quotes, clean 
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)


_WS_RE = re.compile(r"\s+")

def _norm_text(s: str) -> str:
    """
    Description:
        Normalize whitespace to improve downstream chunking quality.

    Args:
        s (str): Raw text.

    Returns:
        str: Trimmed text with internal spaces collapsed.
    """
    s = s.strip()
    if not s:
        return s
    return _WS_RE.sub(" ", s)

def _map_unstructured_category(category: Optional[str]) -> str:
    """
    Description:
        Map Unstructured element categories to simplified types used in the paper.

    Args:
        category (Optional[str]): Unstructured element category (e.g., "Title", "Table", "PageBreak").

    Returns:
        str: One of {"title", "table", "text", "pagebreak"}.
    """
    if not category:
        return "text"
    c = category.lower()
    if c == "title":
        return "title"
    if c == "table":
        return "table"
    return "text"

def extract_elements(doc_path: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Description:
        Extract structural elements from a PDF using the Unstructured library
        and normalize them into the schema used by the structural chunking pipeline.
        Configuration-driven strategy and cleaning settings applied.

    Args:
        doc_path (str): Absolute path to the PDF file.
        doc_id (str): Identifier for the document (usually filename stem).

    Returns:
        List[Dict[str, Any]]:
            A list of element dicts:
            {
              "source_doc": str,
              "doc_id": str,
              "type": "title" | "text" | "table",
              "text": str,
              "page": int | None,
            }
    """
    cfg = load_config()
    ucfg = get_section(cfg, "partitioning")
    cleaning_cfg = get_section(cfg, "cleaning")

    # Extract elements using configured options
    elements = partition_pdf(
        filename=doc_path,
        strategy=ucfg.get("strategy", "hi_res"),
        hi_res_model_name=ucfg.get("hi_res_model", "yolox"),
        languages=ucfg.get("languages", ["eng"]),
        infer_table_structure=ucfg.get("infer_table_structure", True),
    )

    out: List[Dict[str, Any]] = []

    for el in elements:
        t_raw = getattr(el, "category", None)
        t = _map_unstructured_category(t_raw)

        text_raw = getattr(el, "text", "") or ""
        text_norm = _norm_text(text_raw)
        if not text_norm:
            continue

        # Cleaning step if enabled
        text_clean = text_norm
        if cleaning_cfg.get("apply_unicode_quotes", False):
            text_clean = replace_unicode_quotes(text_clean)
        if cleaning_cfg.get("apply_clean", False):
            clean_opts = cleaning_cfg.get("clean_options", {})
            text_clean = clean(
                text_clean,
                bullets=clean_opts.get("bullets", True),
                extra_whitespace=clean_opts.get("extra_whitespace", True),
                dashes=clean_opts.get("dashes", True),
                trailing_punctuation=clean_opts.get("trailing_punctuation", False),
                lowercase=clean_opts.get("lowercase", False),
            )
        if not text_clean.strip():
            continue

        meta = getattr(el, "metadata", None)
        page = meta.page_number if (meta and hasattr(meta, "page_number")) else None

        out.append(
            {
                "source_doc": doc_id,
                "doc_id": f"{doc_id}_p{page}" if page is not None else doc_id,
                "type": t if t in ("title", "table") else "text",
                "text": text_clean,
                "page": page,
            }
        )

    # Ensure deterministic order by page number
    out.sort(key=lambda r: (r.get("page") or 0))
    logger.info(f"[INFO] Extracted {len(out)} elements from {doc_path}")
    return out