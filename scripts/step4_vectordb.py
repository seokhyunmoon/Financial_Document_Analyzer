# scripts/step3_2_vectordb.py

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Any

# Make `src/` importable when running as a script
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from utils.config import load_config, get_section, resolve_path
from utils.files import read_jsonl
from graph.nodes.vectordb import (
    init_client,
    close_client,
    ensure_collection,
    reset_collection,
    upload_objects,
    count_objects,
)

logger = get_logger(__name__)


def main() -> None:
    """Connect, ensure collection, upload objects with vectors, and verify count."""
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")

    collection_name = vsec.get("collection_name", "FinancialDocChunk")

    logger.info(f"[STEP 1] Loading embeddedings")
    
    rows = read_jsonl("data/processed/embeddings/AMERICANEXPRESS_2022_10K.jsonl")
    with_vec = sum(1 for r in rows if r.get("embedding") is not None)
    logger.info(f"[INFO] Total rows: {len(rows)} (with vectors: {with_vec})")

    logger.info("[STEP 2] Connecting to Weaviate")
    client = init_client()

    try:
        # reset_collection(client, collection_name)
        logger.info(f"[OK] Reset collection '{collection_name}'")
        
        logger.info(f"[STEP 3] Ensuring collection '{collection_name}'")
        ensure_collection(client, name=collection_name)

        logger.info(f"[STEP 4] Uploading {len(rows)} objects to '{collection_name}'")
        upload_objects(
            client=client,
            collection_name=collection_name,
            objects=rows,
            batch_size=100,
            concurrent_requests=4,
        )

        logger.info(f"[STEP 5] Counting objects in '{collection_name}'")
        total = count_objects(client, collection_name)
        logger.info(f"[OK] Collection '{collection_name}' total objects: {total}")

    finally:
        close_client(client)


if __name__ == "__main__":
    main()