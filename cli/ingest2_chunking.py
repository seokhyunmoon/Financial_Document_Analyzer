#!/usr/bin/env python
"""
Simple CLI to run chunking on a single document's elements.

Example:
    python cli/ingest2_chunking.py --elements data/elements/AMERICANEXPRESS_2022_10K_elements.jsonl
"""

import argparse
import jsonlines
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.chunking import merge_elements_to_chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk extracted elements")
    parser.add_argument(
        "--elements",
        type=Path,
        required=True,
        help="Path to the <doc>_elements.jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to matching chunks directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    elements_path = args.elements.resolve()
    if not elements_path.exists():
        raise FileNotFoundError(f"Elements file not found: {elements_path}")

    chunks_dir = Path("data/processed/chunks")
    output_path = args.output or chunks_dir / elements_path.name.replace("_elements", "_chunks")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(elements_path, "r") as reader:
        elements = list(reader)

    chunks = merge_elements_to_chunks(elements)

    print(f"[INFO] Writing {len(chunks)} chunks to {output_path}")
    with jsonlines.open(output_path, "w") as writer:
        for c in chunks:
            writer.write(c)

    print(f"[OK] Chunking complete: {output_path}")


if __name__ == "__main__":
    main()
