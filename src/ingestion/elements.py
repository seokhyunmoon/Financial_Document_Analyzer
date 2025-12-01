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
    """Normalize whitespace to improve downstream chunking quality.

    Args:
        s: Raw text.

    Returns:
        Text with leading/trailing whitespace trimmed and internal whitespace collapsed.
    """
    s = s.strip()
    if not s:
        return s
    return _WS_RE.sub(" ", s)


def extract_elements(doc_path: str, doc_id: str) -> List[Dict[str, Any]]:
    """Extract and normalize structural elements from a PDF.

    Args:
        doc_path: Absolute path to the PDF file.
        doc_id: Identifier for the document (usually filename stem).

    Returns:
        List of normalized element dictionaries (type/text/page metadata).
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

    for element in elements:
        # element type
        type = getattr(element, "category", None)

        # text
        text_raw = getattr(element, "text", "") or ""
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
            )
        if not text_clean.strip():
            continue

        #metadata
        meta = getattr(element, "metadata", None)
        element_id = meta.element_id if (meta and hasattr(meta, "element_id")) else None
        page = meta.page_number if (meta and hasattr(meta, "page_number")) else None
        table_as_html = meta.text_as_html if (meta and hasattr(meta, "text_as_html")) else None
        

        out.append(
            {
                "source_doc": doc_id,
                "type": type,
                "text": text_clean,
                "metadata": {
                    "element_id": element_id,
                    "page": page,
                    "table_as_html": table_as_html,
                },
            }
        )

    # Ensure deterministic order by page number
    out.sort(key=lambda r: (r.get("page") or 0))
    logger.info(f"[INFO] Extracted {len(out)} elements from {doc_path}")
    return out
