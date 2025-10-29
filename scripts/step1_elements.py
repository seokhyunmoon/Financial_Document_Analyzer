#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description:
    Step 1 runner script: Load a PDF file, extract structural elements using Unstructured,
    apply cleaning and normalization, and save the output as a JSONL file of elements.

Usage:
    uv run python scripts/step1_elements.py --in <path_to_pdf_file>
"""

import argparse
import sys
import os
from pathlib import Path
import json
from typing import List, Dict, Any

# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.elements import extract_elements


def write_jsonl(output_path: str, records: List[Dict[str, Any]]) -> None:
    """
    Description:
        Write a list of dict records to JSON Lines file.

    Args:
        output_path (str): Path to the output .jsonl file.
        records (List[Dict[str, Any]]): List of dict objects.
    """
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    """
    Description:
        CLI entrypoint: parse arguments, call element extraction, write results.
    """
    parser = argparse.ArgumentParser(description="Step1: Extract elements from PDF")
    parser.add_argument("--in", dest="input_pdf", required=True, help="Path to input PDF file")
    args = parser.parse_args()

    input_path = Path(args.input_pdf)
    if not input_path.is_file():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    doc_id = input_path.stem
    print(f"[INFO] Processing document: {input_path} → doc_id: {doc_id}")

    elements = extract_elements(str(input_path), doc_id)
    print(f"[INFO] Number of elements extracted: {len(elements)}")

    # Compute output path via config or fallback to default
    # For simplicity, assume config path is data/processed/elements/<doc_id>.elements.jsonl
    output_dir = Path("data/processed/elements")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"{doc_id}_elements.jsonl")

    write_jsonl(output_path, elements)
    print(f"[OK] Elements written: {output_path}")
    if elements:
        print(f" - Sample [1] type={elements[0]['type']} page={elements[0]['page']} text[:80]={repr(elements[0]['text'][:80])}")
        if len(elements) > 1:
            print(f" - Sample [2] type={elements[1]['type']} page={elements[1]['page']} text[:80]={repr(elements[1]['text'][:80])}")


if __name__ == "__main__":
    main()