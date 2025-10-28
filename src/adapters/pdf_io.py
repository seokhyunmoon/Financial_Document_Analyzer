from pathlib import Path
from typing import Tuple


def ensure_pdf(path: str) -> Tuple[str, str]:
    """
    Description:
        Validate that the given path points to an existing PDF file and return
        a normalized absolute path plus a derived document id.

    Args:
        path (str): Input file path to a PDF.

    Returns:
        Tuple[str, str]: (abs_path, doc_id) where doc_id is the filename stem.
    """
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {p.suffix}")
    doc_id = p.stem
    return str(p), doc_id