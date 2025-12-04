#!/usr/bin/env python
"""Upload embedding JSONL files to the vector store (Weaviate).

Examples:
    # Upload every embedding file under paths.embed_dir
    python cli/ingest4_vectorstore.py

    # Upload a single embedding file
    python cli/ingest4_vectorstore.py --embeddings data/processed/embeddings/AMEX.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.files import read_jsonl
from ingestion.vectorstore import init_client, close_client, ensure_collection, upload_objects, count_objects

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload embeddings to Weaviate.")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=None,
        help="Path to a single embedding JSONL file. If omitted, upload every *.jsonl under paths.embed_dir.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")
    paths = cfg.get("paths", {})

    emb_dir = Path(paths.get("embed_dir", "data/processed/embeddings")).resolve()
    if args.embeddings:
        files = [args.embeddings.resolve()]
    else:
        if not emb_dir.exists():
            logger.error(f"[ERROR] Embeddings directory not found: {emb_dir}")
            return
        files = sorted(emb_dir.glob("*.jsonl"))
        if not files:
            logger.warning(f"[WARN] No embedding files found in {emb_dir}")
            return

    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f"Embedding file not found: {fp}")

    collection_name = vsec.get("collection_name", "FinancialDocChunk")
    logger.info(f"[STEP] Preparing to upload {len(files)} embedding file(s) to {collection_name}")

    client = init_client()
    try:
        ensure_collection(client, name=collection_name)

        for fp in files:
            logger.info(f"[UPLOAD] {fp.name}")
            rows = read_jsonl(str(fp))
            if not rows:
                continue
            upload_objects(
                client=client,
                collection_name=collection_name,
                objects=rows,
            )

        total = count_objects(client, collection_name)
        logger.info(f"[OK] Collection '{collection_name}' now contains {total} objects")
    finally:
        close_client(client)


if __name__ == "__main__":
    main()
