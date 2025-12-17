#!/usr/bin/env python
"""
Generate LLM metadata (summary and keywords) for one or more chunk files.

Examples:
    # Enrich every *_chunks.jsonl under paths.chunks_dir
    python cli/ingest3_metadata.py

    # Enrich a single chunk file
    python cli/ingest3_metadata.py --chunks data/processed/chunks/AMERICANEXPRESS_2022_10K_chunks.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.metadata import enrich_chunks
from utils.config import load_config, get_section
from utils.files import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for metadata enrichment."""
    parser = argparse.ArgumentParser(description="Enrich chunk JSONL file(s) with LLM metadata")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help="Path to <doc>_chunks.jsonl. If omitted, process all *_chunks.jsonl under paths.chunks_dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (single file only). Defaults to paths.metadata_dir/<doc>_metadata.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate metadata even if summary/keywords already exist.",
    )
    return parser.parse_args()


def _gather_chunk_files(single: Path | None, chunks_dir: Path) -> list[Path]:
    """Collect chunk files to process."""
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


def _default_output_path(chunk_path: Path, metadata_dir: Path) -> Path:
    """Build a default metadata output path for a chunk file."""
    if chunk_path.name.endswith("_chunks.jsonl"):
        filename = chunk_path.name.replace("_chunks.jsonl", "_metadata.jsonl")
    else:
        filename = f"{chunk_path.stem}_metadata.jsonl"
    return metadata_dir / filename


def main() -> None:
    args = parse_args()
    if args.output and args.chunks is None:
        raise ValueError("--output can only be used when processing a single --chunks file.")

    cfg = load_config()
    paths = get_section(cfg, "paths")
    chunks_dir = Path(paths.get("chunks_dir", "data/processed/chunks")).resolve()
    metadata_dir = Path(paths.get("metadata_dir", "data/processed/metadata")).resolve()

    chunk_files = _gather_chunk_files(args.chunks, chunks_dir)

    for chunk_path in chunk_files:
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunk_path}")

        chunks = read_jsonl(str(chunk_path))
        if not chunks:
            print(f"[WARN] No chunks loaded from {chunk_path}, skipping.")
            continue

        enriched = enrich_chunks(chunks, overwrite=args.overwrite)

        default_out = _default_output_path(chunk_path, metadata_dir)
        out_path = (args.output or default_out).resolve()
        write_jsonl(str(out_path), enriched)
        print(f"[OK] Enriched {len(enriched)} chunks → {out_path}")


if __name__ == "__main__":
    main()
