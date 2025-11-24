# scripts/step3_2_vectordb.py

from __future__ import annotations
import sys
from pathlib import Path
from typing import List, Dict, Any

# Make `src/` importable when running as a script
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


def main() -> None:
    """Connect, ensure collection, and upsert every embedding file under data/processed/embeddings."""
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")
    paths = cfg.get("paths", {})

    emb_dir = Path(paths.get("embed_dir", "data/processed/embeddings")).resolve()
    if not emb_dir.exists():
        logger.error(f"[ERR] Embeddings directory not found: {emb_dir}")
        return

    files = sorted(emb_dir.glob("*.jsonl"))
    if not files:
        logger.warning(f"[WARN] No embedding files found in {emb_dir}")
        return

    collection_name = vsec.get("collection_name", "FinancialDocChunk")
    logger.info(f"[STEP 1] Found {len(files)} embedding files under {emb_dir}")

    logger.info("[STEP 2] Connecting to Weaviate")
    client = init_client()

    try:
        reset_collection(client, collection_name)
        logger.info(f"[OK] Reset collection '{collection_name}'")
        
        logger.info(f"[STEP 3] Ensuring collection '{collection_name}'")
        ensure_collection(client, name=collection_name)

        for fp in files:
            logger.info(f"[STEP 4] Uploading {fp.name}")
            rows = read_jsonl(str(fp))
            with_vec = sum(1 for r in rows if r.get("embedding") is not None)
            logger.info(f"[INFO] Rows: {len(rows)} (with vectors: {with_vec})")
            if not rows:
                continue
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
