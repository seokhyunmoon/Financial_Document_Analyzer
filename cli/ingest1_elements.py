#!/usr/bin/env python
"""Ad-hoc helper to run element extraction against a single PDF.

Examples:
    # Process every PDF under paths.raw_dir (default: data/pdfs)
    python cli/ingest1_elements.py
    
    # Process a single PDF
    python cli/ingest1_elements.py --pdf data/pdfs/AMERICANEXPRESS_2022_10K.pdf
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
        default=None,
        help="Path to a single PDF. If omitted, process every *.pdf under paths.raw_dir.",
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
    pdf_dir = Path(paths.get("raw_dir", "data/pdfs")).resolve()
    elements_dir = Path(paths.get("elements_dir", "data/processed/elements")).resolve()

    if args.output and args.pdf is None:
        raise ValueError("--output can only be used when processing a single --pdf file.")

    if args.pdf:
        pdf_paths = [args.pdf.resolve()]
    else:
        if not pdf_dir.exists():
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        pdf_paths = sorted(
            p for p in pdf_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
        )

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDF files found under {pdf_dir}. Provide --pdf to process a single file."
        )

    for pdf_path in pdf_paths:
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        doc_id = pdf_path.stem
        out_path = (args.output or (elements_dir / f"{doc_id}_elements.jsonl")).resolve()

        print(f"[INFO] Extracting elements from {pdf_path}")
        elements = extract_elements(str(pdf_path), doc_id)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            for row in elements:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[OK] Wrote {len(elements)} elements to {out_path}")


if __name__ == "__main__":
    main()
