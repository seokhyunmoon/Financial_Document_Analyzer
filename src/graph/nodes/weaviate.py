# src/graph/nodes/weaviate.py
"""
weaviate.py
-----------
This module provides utilities for connecting to a Weaviate instance,
managing collections, and efficiently uploading document chunks with their embeddings.
"""

from typing import List, Dict, Any
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.init import AdditionalConfig, Timeout
from utils.logger import get_logger
from utils.config import load_config, get_section

logger = get_logger(__name__)


def init_client(host: str = "http://localhost:8080",
                grpc_port: int = 50051,
                skip_init_checks: bool = False) -> weaviate.WeaviateClient:
    """
    Initialize and return a WeaviateClient (v4) connected to a local or remote instance.

    Args:
        host (str): Base URL of the Weaviate REST endpoint.
        grpc_port (int): gRPC port number.
        skip_init_checks (bool): If True, skip readiness checks for faster startup.

    Returns:
        weaviate.WeaviateClient: The connected client instance.

    Raises:
        RuntimeError: If the client cannot confirm readiness of the instance.
    """
    cfg = load_config()
    wsec = get_section(cfg, "weaviate")
    host = wsec.get("host", host)
    grpc_port = int(wsec.get("grpc_port", grpc_port))
    
    
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=grpc_port,
        additional_config=AdditionalConfig(timeout=Timeout(init=30)),
        skip_init_checks=skip_init_checks
    )
    if not skip_init_checks and not client.is_ready():
        raise RuntimeError(f"Weaviate at {host} is not ready.")
    logger.info(f"[OK] Connected to Weaviate at {host}")
    return client

def create_collection(client: weaviate.WeaviateClient,
                      name: str,
                      vector_dim: int) -> None:
    """
    Create a collection (class) in Weaviate if it does not already exist.

    Args:
        client (weaviate.WeaviateClient): The connected client.
        name (str): Name of the collection/class.
        vector_dim (int): Dimensionality of the vectors to be stored.

    Returns:
        None
    """
    existing = client.collections.list_all(simple=False)
    names = [c for c in existing]
    if name in names:
        print(f"[INFO] Collection '{name}' already exists.")
        return

    props = [
        Property(name="source_doc", data_type=DataType.TEXT),
        Property(name="doc_id",      data_type=DataType.TEXT),
        Property(name="chunk_id",    data_type=DataType.TEXT),
        Property(name="element_type",data_type=DataType.TEXT),
        Property(name="text",        data_type=DataType.TEXT),
        Property(name="page_start",  data_type=DataType.INT),
        Property(name="page_end",    data_type=DataType.INT),
    ]

    client.collections.create(
        name=name,
        description=f"Document chunks stored as dense vectors (dim={vector_dim})",
        vector_config=Configure.Vectors.text2vec_huggingface(model="Qwen/Qwen3-Embedding-4B", dimensions=vector_dim),
        properties=props,
    )
    print(f"[OK] Created collection '{name}'")

def upload_objects(client: weaviate.WeaviateClient,
                   collection_name: str,
                   objects: List[Dict[str, Any]],
                   batch_size: int = 100,
                   concurrent_requests: int = 4) -> None:
    """
    Upload objects with embeddings to the specified collection in Weaviate.

    Args:
        client (weaviate.WeaviateClient): The connected client.
        collection_name (str): The target collection name.
        objects (List[Dict[str, Any]]): List of dicts with keys 'source_doc', 'doc_id', 'chunk_id', 'element_type', 'text', 'page_start', 'page_end', 'embedding'.
        batch_size (int): Number of objects per batch.
        concurrent_requests (int): Number of parallel requests.

    Returns:
        None
    """
    collection = client.collections.get(collection_name)
    with collection.batch.fixed_size(batch_size=batch_size, concurrent_requests=concurrent_requests) as batch:
        for obj in objects:
            props = {
                "source_doc":    obj.get("source_doc", "unknown"),
                "doc_id":        obj["doc_id"],
                "chunk_id":      obj.get("chunk_id", ""),
                "element_type":  obj.get("type", ""),
                "text":          obj["text"],
                "page_start":    obj.get("page_start"),
                "page_end":      obj.get("page_end")
            }
            vector = obj["embedding"]
            batch.add_object(
                properties=props,
                vector=vector
            )
        if batch.number_errors > 0:
            print(f"[WARN] {batch.number_errors} objects failed to upload.")
    print(f"[OK] Uploaded {len(objects)} objects to collection '{collection_name}'.")