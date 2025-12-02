#!/usr/bin/env python
"""
Generate embeddings for a single chunk file.

Example:
    python scripts/step3_embed.py --chunks data/processed/chunks/AMERICANEXPRESS_2022_10K_chunks.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.embeddings import generate_embeddings
from utils.files import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunk JSONL file")
    parser.add_argument(
        "--chunks",
        type=Path,
        required=True,
        help="Path to <doc>_chunks.jsonl produced by step2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; defaults to <embeddings_dir>/<doc>.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks_path = args.chunks.resolve()
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    with chunks_path.open("r", encoding="utf-8") as fh:
        chunks = [json.loads(line) for line in fh]

    embedded = generate_embeddings(chunks)

    default_out = (
        Path("data/processed/embeddings") / chunks_path.name.replace("_chunks", "")
    )
    out_path = (args.output or default_out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out_path), embedded)

    print(f"[OK] Embedded {len(embedded)} chunks → {out_path}")


if __name__ == "__main__":
    main()
