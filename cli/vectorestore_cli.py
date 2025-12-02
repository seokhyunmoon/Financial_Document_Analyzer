#!/usr/bin/env python
"""
Utility script to reset or inspect the Weaviate collection.

Examples:
    python cli/vectordb_admin.py --reset
    python cli/vectordb_admin.py --count
    python cli/vectordb_admin.py --reset --count
    python cli/vectordb_admin.py --list
    python cli/vectordb_admin.py --schema
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from utils.config import load_config, get_section
from ingestion.vectorstore import (
    init_client,
    close_client,
    ensure_collection,
    reset_collection,
    count_objects,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Admin tasks for the Weaviate collection."
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Collection name (defaults to vectordb.collection_name).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the collection.",
    )
    parser.add_argument(
        "--count",
        action="store_true",
        help="Output the number of objects in the collection.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all collections in the Weaviate instance.",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print the schema of the target collection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")
    collection_name = args.collection or vsec.get("collection_name", "FinancialDocChunk")

    if not (args.reset or args.count or args.list or args.schema):
        logger.info("No action specified (use --reset/--count/--list/--schema).")
        return

    client = init_client()
    try:
        if args.reset:
            reset_collection(client, collection_name)
            ensure_collection(client, collection_name)
            logger.info(f"[OK] Reset collection '{collection_name}'")

        if args.count:
            total = count_objects(client, collection_name)
            logger.info(f"[INFO] Collection '{collection_name}' contains {total} objects.")

        if args.list:
            names = client.collections.list_all(simple=True)
            logger.info(f"[INFO] Collections: {names}")

        if args.schema:
            col = client.collections.get(collection_name)
            logger.info(f"[INFO] Schema for '{collection_name}': {col.config}")
    finally:
        close_client(client)


if __name__ == "__main__":
    main()
