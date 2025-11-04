# src/graph/nodes/vectordb.py
"""
vectordb.py
-----------
Weaviate connection & collection helpers for manual (custom) vectors.
- Vectorizer disabled (none); vectors are provided from embed.py.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.init import AdditionalConfig, Timeout

from utils.config import load_config, get_section
from utils.logger import get_logger

logger = get_logger(__name__)


def init_client(skip_init_checks: Optional[bool] = None) -> weaviate.WeaviateClient:
    """Initialize a Weaviate client using config (section: `vectordb`).

    Args:
        skip_init_checks (Optional[bool]): If provided, overrides config value to skip
            initial readiness checks.

    Returns:
        weaviate.WeaviateClient: Connected client instance.

    Raises:
        RuntimeError: If client is not ready and checks are enabled.
    """
    cfg = load_config()
    wsec = get_section(cfg, "vectordb")
    host = wsec.get("host", "localhost")
    port = int(wsec.get("port", 8080))
    grpc_port = int(wsec.get("grpc_port", 50051))
    cfg_skip = bool(wsec.get("skip_init_checks", False))
    if skip_init_checks is None:
        skip_init_checks = cfg_skip

    logger.info(f"[INFO] Connecting Weaviate at {host}:{port} (gRPC={grpc_port}) ...")
    client = weaviate.connect_to_local(
        host=host,
        port=port,
        grpc_port=grpc_port,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=30, query=60, insert=60)
        ),
        skip_init_checks=skip_init_checks,
    )

    if not skip_init_checks and not client.is_ready():
        client.close()
        raise RuntimeError("Weaviate is not ready.")
    logger.info("[OK] Connected to Weaviate")
    return client


def close_client(client: Optional[weaviate.WeaviateClient]) -> None:
    """Close the Weaviate client.

    Args:
        client: Client instance to close.
    """
    try:
        if client is not None:
            client.close()
            logger.info("[OK] Closed Weaviate client")
    except Exception as e:
        logger.warning(f"[WARN] Failed to close client cleanly: {e}")


def ensure_collection(client: weaviate.WeaviateClient, name: str, vector_dim: int) -> None:
    """Ensure a collection for manual vectors exists; create if missing.

    Args:
        client: Connected Weaviate client.
        name: Collection name.
        vector_dim: Embedding dimension (must match embedder output).

    Notes:
        - Vectorizer is set to `none`; vectors must be provided on insert.
        - Properties align with the upload payload from embed.py/chunks.py.
        - `chunk_id` is TEXT to allow flexible identifiers.
    """
    props = [
        Property(name="source_doc",   data_type=DataType.TEXT),
        Property(name="doc_id",       data_type=DataType.TEXT),
        Property(name="chunk_id",     data_type=DataType.TEXT),  # keep TEXT for flexibility
        Property(name="element_type", data_type=DataType.TEXT),
        Property(name="text",         data_type=DataType.TEXT),
        Property(name="page_start",   data_type=DataType.INT),
        Property(name="page_end",     data_type=DataType.INT),
    ]

    # v4 SDK supports exists(); keep fallback for older versions
    exists = (
        client.collections.exists(name)
        if hasattr(client.collections, "exists")
        else name in client.collections.list_all(simple=True)
    )
    if exists:
        logger.info(f"[INFO] Collection '{name}' already exists.")
        return

    vector_cfg = Configure.Vectors.none(dimensions=int(vector_dim))
    client.collections.create(
        name=name,
        description=f"Financial document chunks (dim={vector_dim})",
        vector_config=vector_cfg,
        properties=props,
    )
    logger.info(f"[OK] Created collection '{name}' (vectorizer=None, dim={vector_dim})")


def upload_objects(
    client: weaviate.WeaviateClient,
    collection_name: str,
    objects: List[Dict[str, Any]],
    batch_size: int = 100,
    concurrent_requests: int = 4,
) -> None:
    """Batch upload objects (with optional vectors).

    Args:
        client: Weaviate client.
        collection_name: Target collection name.
        objects: Chunk rows; may include 'embedding' (List[float]).
        batch_size: Fixed batch size.
        concurrent_requests: Number of parallel insert workers.
    """
    col = client.collections.get(collection_name)
    total = 0

    # NOTE: vector is not a property; attach via `vector=` when adding objects.
    # Ref: Custom vectors guide.
    with col.batch.fixed_size(batch_size=batch_size, concurrent_requests=concurrent_requests) as batch:
        for obj in objects:
            props = {
                "source_doc":   obj.get("source_doc", "unknown"),
                "doc_id":       str(obj.get("doc_id", "")),
                "chunk_id":     str(obj.get("chunk_id", "")),   # cast to TEXT
                "element_type": str(obj.get("element_type", obj.get("type", ""))),
                "text":         obj.get("text", ""),
                "page_start":   obj.get("page_start"),
                "page_end":     obj.get("page_end"),
            }
            vec = obj.get("embedding") or obj.get("vector")
            if vec is not None:
                batch.add_object(properties=props, vector=vec)
            else:
                batch.add_object(properties=props)
            total += 1

    errs = getattr(batch, "number_errors", 0)
    if errs:
        logger.warning(f"[WARN] {errs} objects failed to upload.")
    logger.info(f"[OK] Uploaded {total - errs} objects to '{collection_name}'")


def count_objects(client: weaviate.WeaviateClient, collection_name: str) -> int:
    """Count total objects in a collection.

    Args:
        client: Weaviate client.
        collection_name: Target collection.

    Returns:
        int: Total count (aggregate).
    """
    col = client.collections.get(collection_name)
    agg = col.aggregate.over_all(total_count=True)
    return int(getattr(agg, "total_count", 0))