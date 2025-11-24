# src/utils/inventory.py
from typing import List, Tuple

from weaviate.collections.classes.aggregate import GroupByAggregate

from utils.config import load_config, get_section
from utils.logger import get_logger
from ingestion.vectorstore import init_client, close_client

logger = get_logger(__name__)


def list_available_documents(max_docs: int = 1000) -> List[Tuple[str, int]]:
    """
    Returns:
        List of tuples [(doc_name, chunk_count), ...] pulled directly from the
        vector database. Only documents that are already indexed (present in the
        collection) are returned.
    """
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")
    collection_name = vsec.get("collection_name", "FinancialDocChunk")

    client = None
    try:
        client = init_client()
        collection = client.collections.get(collection_name)
        group_by = GroupByAggregate(prop="source_doc", limit=max_docs)
        agg = collection.aggregate.over_all(group_by=group_by, total_count=True)

        groups = getattr(agg, "groups", []) or []
        docs: List[Tuple[str, int]] = []
        for group in groups:
            grouped = getattr(group, "grouped_by", None)
            doc_name = getattr(grouped, "value", None) if grouped else None
            if not doc_name:
                continue
            chunk_count = int(getattr(group, "total_count", 0) or 0)
            docs.append((str(doc_name), chunk_count))

        docs.sort(key=lambda item: item[0])
        return docs
    except Exception as exc:
        logger.warning(f"[inventory] Failed to list indexed documents: {exc}")
        return []
    finally:
        close_client(client)
