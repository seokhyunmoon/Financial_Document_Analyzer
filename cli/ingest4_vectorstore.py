#!/usr/bin/env python
"""Upload embedding JSONL files to the vector store (Weaviate).

python cli/ingest4_vectorstore.py --reset \
    --embeddings-dir data/processed/embeddings \
    --collection FinancialDocChunk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.files import read_jsonl
from ingestion.vectorstore import (
    init_client,
    close_client,
    ensure_collection,
    reset_collection,
    upload_objects,
    count_objects,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload embeddings to Weaviate.")
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Directory of *.jsonl embedding files (defaults to config paths.embed_dir).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Weaviate collection name (defaults to vectordb.collection_name).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection before uploading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")
    paths = cfg.get("paths", {})

    emb_dir = args.embeddings_dir or Path(paths.get("embed_dir", "data/processed/embeddings"))
    emb_dir = emb_dir.resolve()
    if not emb_dir.exists():
        logger.error(f"[ERR] Embeddings directory not found: {emb_dir}")
        return

    files = sorted(emb_dir.glob("*.jsonl"))
    if not files:
        logger.warning(f"[WARN] No embedding files found in {emb_dir}")
        return

    collection_name = args.collection or vsec.get("collection_name", "FinancialDocChunk")
    logger.info(f"[STEP] Found {len(files)} embedding files under {emb_dir}")

    client = init_client()
    try:
        if args.reset:
            reset_collection(client, collection_name)
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
