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
from graph.nodes.vectordb import init_client, close_client
from graph.nodes.query import query_embeddings 

logger = get_logger(__name__)

def retrieve_topk(
    question: str,
    topk: Optional[int] = None,
    source_doc: Optional[str] = None,  
) -> List[Dict[str, Any]]:
    """Retrieves the top-k most relevant document chunks for a given question.

    This function first converts the input question into a vector embedding.
    It then uses this embedding to perform a similarity search in the Weaviate
    vector database to find the most relevant document chunks. An optional
    filter can be applied to limit the search to a specific source document.

    Args:
        question: The question to find relevant documents for.
        topk: The number of top results to retrieve. If None, defaults to the
              value specified in the configuration file.
        source_doc: Optional. The name of the source document to filter results by.
                      If None, the search is performed across all documents.

    Returns:
        A list of dictionaries, where each dictionary contains the metadata
        and text of a retrieved document chunk.
    """
    # load config
    cfg = load_config()
    qsec = get_section(cfg, "qa")
    vsec = get_section(cfg, "vectordb")
    collection = vsec.get("collection_name", "FinancialDocChunk")
    k = qsec.get("topk", 10) if topk is None else topk

    # 1) generate query embedding
    qvec = query_embeddings(question)

    # 2) search
    client = init_client()
    try:
        col = client.collections.get(collection)
        flt = Filter.by_property("source_doc").equal(source_doc) if source_doc else None

        res = col.query.near_vector(
            near_vector=qvec,
            limit=k,
            filters=flt,
            return_properties=[
                "source_doc","doc_id","chunk_id","element_type","text","page_start","page_end"
            ],
            include_vector=False,
        )

        hits: List[Dict[str, Any]] = []
        for o in res.objects:
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
        logger.info(f"[OK] Retrieved {len(hits)}/{k} hits from '{collection}'")
        return hits
    finally:
        close_client(client)
