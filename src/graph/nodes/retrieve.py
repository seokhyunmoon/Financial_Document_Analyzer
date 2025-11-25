# src/graph/nodes/retrieve.py
"""
retrieve.py
-----------
This module defines nodes for retrieving relevant document chunks from the vector database based on a query.
"""
from typing import List, Dict, Any, Optional
from weaviate.classes.query import Filter
from utils.logger import get_logger
from utils.config import load_config, get_section
from ingestion.vectorstore import init_client, close_client
from graph.nodes.query import query_embeddings 

logger = get_logger(__name__)

def retrieve_topk(
    question: str,
    question_vector: Optional[List[float]],
    topk: Optional[int] = None,
    source_doc: Optional[str] = None,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k relevant document chunks from the vector database
    based on the provided question embedding vector.

    Args:
        question: The input query string.
        question_vector: A list of floats representing the embedding vector of the query.
        topk: top-k to retrieve.
        source_doc: A string for filtering by doc Name (ex: "PEPSICO_2022_10K")

    Returns:
        A list of dictionaries, where each dictionary contains the metadata
        and text of a retrieved document chunk.
    """
    cfg = load_config()
    qsec = get_section(cfg, "qa")
    vsec = get_section(cfg, "vectordb")
    topk = topk or qsec.get("topk", 10)
    retriever_mode = (mode or qsec.get("retriever_mode", "vector")).lower()
    hybrid_alpha = float(qsec.get("hybrid_alpha", 0.5))
    collection_name = vsec.get("collection_name", "FinancialDocChunk")

    needs_vector = retriever_mode in ("vector", "hybrid")
    if needs_vector:
        if question_vector is None:
            question_vector = query_embeddings(question)
    else:
        question_vector = None

    # 2) search
    client = init_client()
    try:
        collection = client.collections.get(collection_name)
        
        w_filter = None
        if source_doc:
            w_filter = Filter.by_property("source_doc").equal(source_doc)

        return_props = [
            "source_doc",
            "doc_id",
            "chunk_id",
            "element_type",
            "text",
            "page_start",
            "page_end",
        ]

        if retriever_mode == "vector":
            res = collection.query.near_vector(
                near_vector=question_vector,
                limit=topk,
                filters=w_filter,
                return_properties=return_props,
                include_vector=False,
            )
        elif retriever_mode == "keyword":
            res = collection.query.bm25(
                query=question,
                limit=topk,
                filters=w_filter,
                return_properties=return_props,
            )
        elif retriever_mode == "hybrid":
            res = collection.query.hybrid(
                query=question,
                vector=question_vector,
                alpha=hybrid_alpha,
                limit=topk,
                filters=w_filter,
                return_properties=return_props,
            )
        else:
            raise ValueError(f"Unsupported retriever_mode: {retriever_mode}")

        hits: List[Dict[str, Any]] = []
        for o in getattr(res, "objects", []):
            props = o.properties or {}
            hits.append({
                "chunk_id":   props.get("chunk_id"),
                "source_doc": props.get("source_doc"),
                "doc_id":     props.get("doc_id"),
                "type":       props.get("element_type"),
                "page_start": props.get("page_start"),
                "page_end":   props.get("page_end"),
                "text":       props.get("text"),
            })
        logger.info(
            f"[OK] Retrieved {len(hits)}/{topk} hits from '{collection_name}' mode={retriever_mode}"
        )
        return hits
    finally:
        close_client(client)
