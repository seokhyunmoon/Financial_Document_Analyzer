from typing import List, Dict, Any, Optional
import re
from pathlib import Path
import yaml

from unstructured.partition.pdf import partition_pdf
from unstructured.cleaners.core import replace_unicode_quotes, clean  # cleaning functions

# ----------------------- Helpers -----------------------

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
    if c == "pagebreak":
        return "pagebreak"
    return "text"

def _serialize_coordinates(metadata: Any) -> Optional[Dict[str, Any]]:
    """
    Description:
        Extract layout coordinates from Unstructured element metadata and convert
        into a normalized dictionary when available.

    Args:
        metadata (Any): Element metadata object (may contain coordinates).

    Returns:
        Optional[Dict[str, Any]]: Normalized coordinate dict, or None if not available.
    """
    if metadata is None:
        return None
    coordinates = getattr(metadata, "coordinates", None)
    if not coordinates:
        return None
    if hasattr(coordinates, "to_dict"):
        coord_dict = coordinates.to_dict()
    elif isinstance(coordinates, dict):
        coord_dict = dict(coordinates)
    else:
        coord_dict = {
            key: getattr(coordinates, key)
            for key in ("points", "system", "layout_width", "layout_height")
            if hasattr(coordinates, key)
        }
    points = coord_dict.get("points")
    if not points:
        return None

    normalized_points = []
    for point in points:
        if isinstance(point, dict):
            x = point.get("x")
            y = point.get("y")
        elif isinstance(point, (list, tuple)) and len(point) == 2:
            x, y = point
        else:
            continue
        try:
            x_f = float(x)
            y_f = float(y)
        except (TypeError, ValueError):
            continue
        normalized_points.append([x_f, y_f])

    if not normalized_points:
        return None

    coord_out = dict(coord_dict)
    coord_out["points"] = normalized_points
    return coord_out

def _load_config() -> Dict[str, Any]:
    """
    Description:
        Load default.yaml configuration and return as dict.

    Returns:
        Dict[str, Any]: Parsed configuration.
    """
    cfg_path = Path("configs/default.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ----------------------- Main extractor -----------------------

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
              "doc_id": str,
              "type": "title" | "text" | "table",
              "text": str,
              "page": int | None,
              "bbox": Dict[str, Any] | None,
              "caption": str | None       # Table caption if available
            }
    """
    cfg = _load_config()
    ucfg = cfg.get("unstructured", {})
    cleaning_cfg = cfg.get("cleaning", {})

    # Extract elements using configured options
    elements = partition_pdf(
        filename=doc_path,
        strategy=ucfg.get("strategy", "hi_res"),
        hi_res_model_name=ucfg.get("hi_res_model", "yolox"),
        include_page_breaks=ucfg.get("include_page_breaks", False),
        languages=ucfg.get("languages", ["eng"]),
        infer_table_structure=ucfg.get("infer_table_structure", True),
    )

    out: List[Dict[str, Any]] = []
    store_bbox = ucfg.get("store_bbox", False)

    for el in elements:
        t_raw = getattr(el, "category", None)
        t = _map_unstructured_category(t_raw)
        if t == "pagebreak":
            continue

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

        caption: Optional[str] = None
        if t == "table":
            caption = None  # placeholder, future improvement

        out.append(
            {
                "doc_id": doc_id,
                "type": t if t in ("title", "table") else "text",
                "text": text_clean,
                "page": page,
                "bbox": _serialize_coordinates(meta) if store_bbox else None,
                "caption": caption,
            }
        )

    # Ensure deterministic order by page number
    out.sort(key=lambda r: (r.get("page") or 0))
    return out