#!/usr/bin/env python
"""
Simple CLI to run chunking on one or more element JSONL files.

Examples:
    # Chunk every *_elements.jsonl under paths.elements_dir
    python cli/ingest2_chunking.py
    
    # Chunk a single element file
    python cli/ingest2_chunking.py --elements data/processed/elements/AMERICANEXPRESS_2022_10K_elements.jsonl
"""

import argparse
import jsonlines
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ingestion.chunking import merge_elements_to_chunks
from utils.config import load_config, get_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk extracted elements")
    parser.add_argument(
        "--elements",
        type=Path,
        default=None,
        help="Path to the <doc>_elements.jsonl file. If omitted, chunk all files under paths.elements_dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (single file only). Defaults to paths.chunks_dir/<doc>_chunks.jsonl.",
    )
    return parser.parse_args()


def _gather_element_files(single: Path | None, elements_dir: Path) -> list[Path]:
    if single:
        return [single.resolve()]
    if not elements_dir.exists():
        raise FileNotFoundError(f"Elements directory not found: {elements_dir}")
    files = sorted(
        p
        for p in elements_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".jsonl" and p.name.endswith("_elements.jsonl")
    )
    if not files:
        raise FileNotFoundError(
            f"No *_elements.jsonl files found in {elements_dir}. Provide --elements for a specific file."
        )
    return files


def main():
    args = parse_args()
    if args.output and args.elements is None:
        raise ValueError("--output can only be used when processing a single --elements file.")

    cfg = load_config()
    paths = get_section(cfg, "paths")
    elements_dir = Path(paths.get("elements_dir", "data/processed/elements")).resolve()
    chunks_dir = Path(paths.get("chunks_dir", "data/processed/chunks")).resolve()

    element_files = _gather_element_files(args.elements, elements_dir)

    for element_path in element_files:
        if not element_path.exists():
            raise FileNotFoundError(f"Elements file not found: {element_path}")

        default_out = chunks_dir / element_path.name.replace("_elements", "_chunks")
        output_path = (args.output or default_out).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with jsonlines.open(element_path, "r") as reader:
            elements = list(reader)

        chunks = merge_elements_to_chunks(elements)

        print(f"[INFO] Writing {len(chunks)} chunks to {output_path}")
        with jsonlines.open(output_path, "w") as writer:
            for c in chunks:
                writer.write(c)

        print(f"[OK] Chunking complete: {output_path}")


if __name__ == "__main__":
    main()
