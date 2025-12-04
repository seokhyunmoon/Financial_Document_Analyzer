#!/usr/bin/env python
"""
Generate embeddings for one or more chunk files.

Examples:
    # Embed every chunk file under paths.chunks_dir
    python cli/ingest3_embed.py 

    # Embed a single chunk file
    python cli/ingest3_embed.py --chunks data/processed/chunks/AMERICANEXPRESS_2022_10K_chunks.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.embeddings import generate_embeddings
from utils.config import load_config, get_section
from utils.files import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed chunk JSONL file(s)")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to <doc>_chunks.jsonl. If omitted, embed every *_chunks.jsonl under paths.chunks_dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (single file only). Defaults to <embeddings_dir>/<doc>.jsonl.",
    )
    return parser.parse_args()


def _gather_chunk_files(single: Path | None, chunks_dir: Path) -> list[Path]:
    if single:
        return [single.resolve()]
    if not chunks_dir.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_dir}")
    files = sorted(
        p
        for p in chunks_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jsonl" and p.name.endswith("_chunks.jsonl")
    )
    if not files:
        raise FileNotFoundError(
            f"No *_chunks.jsonl files found in {chunks_dir}. Provide --chunks for a specific file."
        )
    return files


def main() -> None:
    args = parse_args()
    if args.output and args.chunks is None:
        raise ValueError("--output can only be used when processing a single --chunks file.")

    cfg = load_config()
    paths = get_section(cfg, "paths")
    chunks_dir = Path(paths.get("chunks_dir", "data/processed/chunks")).resolve()
    embed_dir = Path(paths.get("embed_dir", "data/processed/embeddings")).resolve()

    chunk_files = _gather_chunk_files(args.chunks, chunks_dir)

    for chunk_path in chunk_files:
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunk_path}")

        with chunk_path.open("r", encoding="utf-8") as fh:
            chunks = [json.loads(line) for line in fh if line.strip()]

        if not chunks:
            print(f"[WARN] No chunks loaded from {chunk_path}, skipping.")
            continue

        embedded = generate_embeddings(chunks)

        default_out = embed_dir / chunk_path.name.replace("_chunks", "")
        out_path = (args.output or default_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(str(out_path), embedded)

        print(f"[OK] Embedded {len(embedded)} chunks → {out_path}")


if __name__ == "__main__":
    main()
