# src/graph/nodes/elements.py
"""
elements.py
------------
Unstructured-native PDF element extraction node.

This version preserves the Element objects (not dicts)
for downstream compatibility with Unstructured's chunking,
cleaning, and embedding modules.
"""

from typing import List
from pathlib import Path
import re
from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import replace_unicode_quotes, clean
from unstructured.staging.base import elements_to_json
from unstructured.documents.elements import Element
from utils.config import load_config, get_section
from utils.logger import get_logger


logger = get_logger(__name__)

_WS_RE = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    """Normalize whitespace to improve readability."""
    s = s.strip()
    return _WS_RE.sub(" ", s) if s else s


def extract_elements(doc_path: str, doc_id: str) -> List[Element]:
    """
    Extract structured Element objects from a PDF using Unstructured,
    applying cleaning based on configuration while preserving full metadata.

    Args:
        doc_path (str): Absolute path to the PDF document.
        doc_id (str): Unique identifier for the document (usually filename stem).

    Returns:
        List[Element]: List of Unstructured Element objects.
    """    
    cfg = load_config()
    ucfg = get_section(cfg, "partitioning")
    cclean = get_section(cfg, "cleaning")

    # --- Step 1. Extract elements from PDF ---
    logger.info(f"[INFO] Extracting elements from {Path(doc_path).name} ...")
    
    elements: List[Element] = partition_pdf(
        filename=doc_path,
        strategy=ucfg.get("strategy", "hi_res"),
        hi_res_model_name=ucfg.get("hi_res_model", "yolox"),
        include_page_breaks=ucfg.get("include_page_breaks", False),
        languages=ucfg.get("languages", ["eng"]),
        infer_table_structure=ucfg.get("infer_table_structure", True),
    )
    
    logger.info(f"[OK] Partitioned into {len(elements)} raw elements")
    
    # --- Step 2. Normalize and clean texts ---
    logger.info("[INFO] Cleaning and normalizing text ...")
    
    for el in elements:
        if not hasattr(el, "text") or not el.text:
            continue
        el.text = _norm_text(el.text)
        if cclean.get("apply_unicode_quotes", False):
            el.text = replace_unicode_quotes(el.text)
        if cclean.get("apply_clean", False):
            opts = cclean.get("clean_options", {})
            el.text = clean(
                el.text,
                bullets=opts.get("bullets", True),
                extra_whitespace=opts.get("extra_whitespace", True),
                dashes=opts.get("dashes", True),
                trailing_punctuation=opts.get("trailing_punctuation", False),
                lowercase=opts.get("lowercase", False),
            )

        # attach doc_id to metadata for traceability
        if hasattr(el, "metadata") and el.metadata is not None:
            el.metadata.filename = str(doc_id)

    logger.info(f"[OK] Extracted {len(elements)} native Elements from {Path(doc_path).name}")
    return elements


def save_elements_jsonl(elements: List[Element], output_path: Path) -> None:
    """
    Save Element objects to a JSONL file using Unstructured's staging API.

    Args:
        elements (List[Element]): List of Unstructured Elements.
        output_path (Path): Path to the output JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elements_to_json(elements, filename=output_path)
    logger.info(f"[OK] Saved {len(elements)} native elements → {output_path}")