#!/usr/bin/env python
"""
Generate embeddings for one or more chunk/metadata files.

Examples:
    # Embed every metadata file under paths.metadata_dir (if enabled), otherwise chunks_dir
    python cli/ingest4_embed.py 

    # Embed a single chunk/metadata file
    python cli/ingest4_embed.py --chunks data/processed/chunks/AMERICANEXPRESS_2022_10K_chunks.jsonl
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
    parser = argparse.ArgumentParser(description="Embed chunk/metadata JSONL file(s)")
    parser.add_argument(
        "--chunks",
        type=Path,
        default=None,
        help=(
            "Path to <doc>_chunks.jsonl or <doc>_metadata.jsonl. If omitted, embed "
            "metadata files under paths.metadata_dir when metadata.enabled=true, "
            "otherwise embed *_chunks.jsonl under paths.chunks_dir."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (single file only). Defaults to <embeddings_dir>/<doc>.jsonl.",
    )
    return parser.parse_args()


def _gather_input_files(
    single: Path | None,
    chunks_dir: Path,
    metadata_dir: Path,
    metadata_enabled: bool,
) -> list[Path]:
    if single:
        return [single.resolve()]

    if metadata_enabled and metadata_dir.exists():
        files = sorted(
            p
            for p in metadata_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".jsonl" and p.name.endswith("_metadata.jsonl")
        )
        if files:
            return files

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
    metadata_dir = Path(paths.get("metadata_dir", "data/processed/metadata")).resolve()
    embed_dir = Path(paths.get("embed_dir", "data/processed/embeddings")).resolve()

    msec = get_section(cfg, "metadata", {})
    metadata_enabled = bool(msec.get("enabled", True))
    chunk_files = _gather_input_files(args.chunks, chunks_dir, metadata_dir, metadata_enabled)

    for chunk_path in chunk_files:
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunk_path}")

        with chunk_path.open("r", encoding="utf-8") as fh:
            chunks = [json.loads(line) for line in fh if line.strip()]

        if not chunks:
            print(f"[WARN] No chunks loaded from {chunk_path}, skipping.")
            continue

        embedded = generate_embeddings(chunks)

        name = chunk_path.name
        if name.endswith("_metadata.jsonl"):
            name = name.replace("_metadata.jsonl", ".jsonl")
        elif name.endswith("_chunks.jsonl"):
            name = name.replace("_chunks.jsonl", ".jsonl")
        default_out = embed_dir / name
        out_path = (args.output or default_out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(str(out_path), embedded)

        print(f"[OK] Embedded {len(embedded)} chunks → {out_path}")


if __name__ == "__main__":
    main()
