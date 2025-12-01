#!/usr/bin/env python
"""Ad-hoc helper to run element extraction against a single PDF.

Example:
    python scripts/step1_elements.py --pdf data/pdfs/AMERICANEXPRESS_2022_10K.pdf
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.elements import extract_elements
from utils.config import load_config, get_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract elements from a PDF")
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        help="Path to the PDF to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to <elements_dir>/<doc_id>_elements.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    paths = get_section(cfg, "paths")
    elements_dir = Path(paths.get("elements_dir", "data/processed/elements")).resolve()

    pdf_path = args.pdf.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = pdf_path.stem
    out_path = args.output or (elements_dir / f"{doc_id}_elements.jsonl")

    print(f"[INFO] Extracting elements from {pdf_path}")
    elements = extract_elements(str(pdf_path), doc_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in elements:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(elements)} elements to {out_path}")


if __name__ == "__main__":
    main()
