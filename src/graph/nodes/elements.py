from typing import List, Dict, Any, Optional
from unstructured.partition.pdf import partition_pdf


# --- Mapping helper -----------------------------------------------------------

def _map_unstructured_category(category: Optional[str]) -> str:
    """
    Description:
        Map Unstructured element categories to simplified types used in the paper.

    Args:
        category (Optional[str]): Unstructured element category (e.g., "Title", "Table", ...).

    Returns:
        str: One of {"title", "table", "text"}.
    """
    if not category:
        return "text"
    c = category.lower()
    if c == "title":
        return "title"
    if c == "table":
        return "table"
    # Many categories exist (NarrativeText, List, FigureCaption, etc.)
    # We treat all non-title/table as "text" for the first iteration.
    return "text"


def _serialize_coordinates(metadata: Any) -> Optional[Dict[str, Any]]:
    """
    Extract layout coordinates from Unstructured element metadata.
    Returns a dictionary with normalized point coordinates when available.
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

    coord_dict = dict(coord_dict)
    coord_dict["points"] = normalized_points
    return coord_dict


def extract_elements(doc_path: str, doc_id: str) -> List[Dict[str, Any]]:
    """
    Description:
        Extract structural elements from a PDF using the Unstructured library
        and normalize them into the schema used by the structural chunking pipeline.

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
    # Note:
    # - partition_pdf can be configured with strategies (hi_res etc.).
    # - We start simple; upgrade to layout-aware configs later if needed.
    elements = partition_pdf(filename=doc_path)
    if elements and not any(_serialize_coordinates(getattr(el, "metadata", None)) for el in elements):
        try:
            elements = partition_pdf(filename=doc_path, strategy="hi_res")
        except Exception:
            # Fall back to the original elements if the hi_res model is unavailable.
            pass

    out: List[Dict[str, Any]] = []
    for el in elements:
        t = _map_unstructured_category(getattr(el, "category", None))
        text = getattr(el, "text", "") or ""
        page = None
        meta = getattr(el, "metadata", None)
        if meta and hasattr(meta, "page_number"):
            page = meta.page_number

        caption: Optional[str] = None
        # For tables, Unstructured sometimes provides a caption as a separate element
        # or in metadata. We keep a placeholder for consistency.
        if t == "table":
            # Future: try to infer caption from nearby FigureCaption/List/Title.
            caption = None

        # Basic cleanup: skip empty text blocks
        if not text.strip():
            continue

        out.append(
            {
                "doc_id": doc_id,
                "type": t,
                "text": text,
                "page": page,
                "bbox": _serialize_coordinates(meta),
                "caption": caption,
            }
        )
    return out
