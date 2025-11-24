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
from weaviate.classes.config import Property, DataType, Configure, VectorDistances
from weaviate.classes.init import AdditionalConfig, Timeout
from weaviate.util import generate_uuid5
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
    vsec = get_section(cfg, "vectordb")
    use_docker = vsec.get("use_docker", True)  # False = embedded, True = external/docker
    
    init_cfg = get_section(vsec, "init")
    host = init_cfg.get("host", "localhost")
    port = init_cfg.get("port", 8080)
    grpc_port = init_cfg.get("grpc_port", 50051)
    cfg_skip = init_cfg.get("skip_init_checks", False)

    # Optional, only used in embedded mode
    persistence_data_path = init_cfg.get("persistence_data_path", None)
    binary_path = init_cfg.get("binary_path", None)
    
    if skip_init_checks is None:
        skip_init_checks = cfg_skip

    # Optional, only used in embedded mode
    persistence_data_path = init_cfg.get("persistence_data_path", None)
    binary_path = init_cfg.get("binary_path", None)

    logger.info(
        f"[INFO] Connecting Weaviate at {host}:{port} (gRPC={grpc_port}, "
        f"use_docker={use_docker}) ..."
    )

    additional_cfg = AdditionalConfig(
        timeout=Timeout(init=30, query=60, insert=60)
    )

    client: weaviate.WeaviateClient

    try:
        if use_docker:
            # Expect an external Weaviate (Docker, binary, etc.) on host:port
            client = weaviate.connect_to_local(
                host=host,
                port=port,
                grpc_port=grpc_port,
                additional_config=additional_cfg,
                skip_init_checks=skip_init_checks,
            )
        else:
            # Start (or reuse) embedded Weaviate
            client = weaviate.connect_to_embedded(
                hostname=host,
                port=port,
                grpc_port=grpc_port,
                additional_config=additional_cfg,
                persistence_data_path=persistence_data_path,
                binary_path=binary_path,
            )
    except Exception as e:
        logger.warning(f"[ERROR] Failed to initialize Weaviate client: {e}")
        raise

    # Optionally verify readiness (works for both embedded & local)
    if not skip_init_checks and not client.is_ready():
        client.close()
        raise RuntimeError("[ERROR] Weaviate is not ready.")

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


def ensure_collection(client: weaviate.WeaviateClient, name: str) -> None:
    """Ensure a collection for manual vectors exists; create if missing.

    Args:
        client: Connected Weaviate client.
        name: Collection name.

    Notes:
        - Vectorizer is set to `none`; vectors must be provided on insert.
        - Properties align with the upload payload from embed.py/chunks.py.
        - `chunk_id` is TEXT to allow flexible identifiers.
    """
    logger.info(f"[INFO] Creating collection '{name}' ...")
    
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

    #Bring your own vectors + 
    vector_cfg = Configure.Vectors.self_provided(
        vector_index_config=Configure.VectorIndex.hnsw(
            distance_metric=VectorDistances.COSINE,
        )
    )
    client.collections.create(
        name=name,
        description=f"Financial document chunks",
        vector_config=vector_cfg,
        properties=props,
    )
    logger.info(f"[OK] Created collection '{name}'")

 
def reset_collection(client: weaviate.WeaviateClient, name: str) -> None:
    """Drop and re-create the collection using ensure_collection.
    
    Args:
        client: Connected Weaviate client.
        name: Collection name.
    """
    if client.collections.exists(name):
        logger.info(f"[INFO] Dropping collection '{name}' ...")
        client.collections.delete(name)  
        logger.info(f"[OK] Dropped '{name}'")
    ensure_collection(client, name)
    logger.info(f"[OK] Reset collection '{name}'")


def upload_objects(
    client: weaviate.WeaviateClient,
    collection_name: str,
    objects: List[Dict[str, Any]],
    batch_size: int = 100,
    concurrent_requests: int = 4,
    upsert: bool = True,
) -> None:
    """Batch upload objects (with optional vectors).

    Args:
        client: Weaviate client.
        collection_name: Target collection name.
        objects: Chunk rows; may include 'embedding' (List[float]).
        batch_size: Fixed batch size.
        concurrent_requests: Number of parallel insert workers.
    """
    logger.info(f"[INFO] Upserting objects to '{collection_name}' ...")
    
    # load config 
    cfg = load_config()
    vsec = get_section(cfg, "vectordb")
    upload_cfg = get_section(vsec, "upload")
    batch_size = upload_cfg.get("batch_size", batch_size)
    concurrent_requests = upload_cfg.get("concurrent_requests", concurrent_requests)
    upsert = upload_cfg.get("upsert", upsert)

    col = client.collections.get(collection_name)
    total = 0
    failed = 0

    def _props_from_obj(obj: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source_doc":   obj.get("source_doc", "unknown"),
            "doc_id":       str(obj.get("doc_id", "")),
            "chunk_id":     str(obj.get("chunk_id", "")),   
            "element_type": str(obj.get("type", "")),       
            "text":         obj.get("text", ""),
            "page_start":   obj.get("page_start"),
            "page_end":     obj.get("page_end"),
        }

    if upsert:
        # Idempotent path: insert with deterministic UUID; on conflict, replace.
        for obj in objects:
            props = _props_from_obj(obj)
            vec = obj.get("embedding") or obj.get("vector")
            # Stable key for UUIDv5
            key = {
                "source_doc": props["source_doc"],
                "doc_id":     props["doc_id"],
                "chunk_id":   props["chunk_id"],
            }
            uuid = generate_uuid5(key)
            try:
                # Try create (fast path when not exists)
                col.data.insert(properties=props, uuid=uuid, vector=vec)
                total += 1
            except Exception:
                # If already exists (or other insert error), replace to achieve upsert semantics
                try:
                    col.data.replace(uuid=uuid, properties=props, vector=vec)
                    total += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"[WARN] Upsert failed for UUID={uuid}: {e}")
        if failed:
            logger.warning(f"[WARN] {failed} objects failed to upsert.")
        logger.info(f"[OK] Upserted {total} objects to '{collection_name}'")
        return

    # Standard insert path: batch insert without UUIDs (auto-generated).
    with col.batch.fixed_size(batch_size=batch_size, concurrent_requests=concurrent_requests) as batch:
        for obj in objects:
            props = _props_from_obj(obj)
            vec = obj.get("embedding") or obj.get("vector")
            key = {
                "source_doc": props["source_doc"],
                "doc_id":     props["doc_id"],
                "chunk_id":   props["chunk_id"],
            }
            uuid = generate_uuid5(key)
            if vec is not None:
                batch.add_object(properties=props, vector=vec, uuid=uuid)
            else:
                batch.add_object(properties=props, uuid=uuid)
            total += 1

        errs = getattr(batch, "number_errors", 0)
        if errs:
            logger.warning(f"[WARN] {errs} objects failed to upload.")
        logger.info(f"[OK] Uploaded {total - errs} objects to '{collection_name}'")