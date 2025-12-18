# src/graph/nodes/retrieve.py
"""
retrieve.py
-----------
This module defines nodes for retrieving relevant document chunks from the vector database based on a query.
"""
from typing import List, Dict, Any, Optional
from collections import defaultdict
from weaviate.classes.query import Filter
from utils.logger import get_logger
from utils.config import load_config, get_section
from ingestion.vectorstore import init_client, close_client
from graph.nodes.query import query_embeddings 

logger = get_logger(__name__)


def _hit_from_obj(o) -> Dict[str, Any]:
    """Convert a Weaviate object to a hit dict.

    Args:
        o: Weaviate object returned from a query.

    Returns:
        Dictionary containing chunk fields used downstream.
    """
    props = o.properties or {}
    return {
        "chunk_id": props.get("chunk_id"),
        "source_doc": props.get("source_doc"),
        "type": props.get("element_type"),
        "section_title": props.get("section_title"),
        "page_start": props.get("page_start"),
        "page_end": props.get("page_end"),
        "text": props.get("text"),
        "text_as_html": props.get("text_as_html"),
        "summary": props.get("summary"),
        "keywords": props.get("keywords"),
    }


def _rrf_merge(
    vec_hits: List[Dict[str, Any]],
    kw_hits: List[Dict[str, Any]],
    rrf_k: float,
    merge_topk: int,
) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion merge of vector/BM25 results.

    Args:
        vec_hits: Hits from the vector search leg.
        kw_hits: Hits from the BM25 search leg.
        rrf_k: RRF damping constant (larger → flatter scores).
        merge_topk: Maximum merged results to return.

    Returns:
        Deduplicated, RRF-scored hits limited to ``merge_topk``.
    """
    scores = defaultdict(float)
    chosen = {}
    for rank, h in enumerate(vec_hits, start=1):
        key = (h.get("source_doc"), h.get("chunk_id"))
        scores[key] += 1.0 / (rrf_k + rank)
        chosen.setdefault(key, h)
    for rank, h in enumerate(kw_hits, start=1):
        key = (h.get("source_doc"), h.get("chunk_id"))
        scores[key] += 1.0 / (rrf_k + rank)
        chosen.setdefault(key, h)
    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [chosen[key] for key, _ in merged[:merge_topk]]

def retrieve_topk(
    question: str,
    question_vector: Optional[List[float]],
    topk: Optional[int] = None,
    source_doc: Optional[str] = None,
    mode: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieve relevant document chunks from the vector database.

    Args:
        question: Query text to search for.
        question_vector: Precomputed embedding for the question, if available.
        topk: Number of chunks to return; defaults to config.
        source_doc: Optional document name filter.
        mode: Retrieval mode (``vector``, ``keyword``, ``hybrid``); defaults to config.

    Returns:
        List of chunk dictionaries containing text and metadata.
    """
    cfg = load_config()
    qsec = get_section(cfg, "qa")
    vsec = get_section(cfg, "vectordb")
    topk = topk or qsec.get("topk", 10)
    retriever_mode = (mode or qsec.get("retriever_mode", "vector")).lower()
    hybrid_alpha = float(qsec.get("hybrid_alpha", 0.5))
    keyword_props = qsec.get("keyword_properties", ["text", "section_title"])
    vector_topk = int(qsec.get("vector_topk", topk))
    keyword_topk = int(qsec.get("keyword_topk", topk))
    merge_topk = int(qsec.get("merge_topk", topk))
    rrf_k = float(qsec.get("rrf_k", 60.0))
    collection_name = vsec.get("collection_name", "FinancialDocChunk")

    needs_vector = retriever_mode in ("vector", "hybrid")
    needs_vector = needs_vector or retriever_mode == "fusion"
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
            "chunk_id",
            "element_type",
            "section_title",
            "text",
            "text_as_html",
            "summary",
            "keywords",
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
            hits = [_hit_from_obj(o) for o in getattr(res, "objects", [])]
            logger.info(
                f"[OK] Retrieved {len(hits)}/{topk} hits from '{collection_name}' mode=vector"
            )
            return hits
        elif retriever_mode == "keyword":
            res = collection.query.bm25(
                query=question,
                limit=keyword_topk,
                query_properties=keyword_props,
                filters=w_filter,
                return_properties=return_props,
            )
            hits = [_hit_from_obj(o) for o in getattr(res, "objects", [])]
            logger.info(
                f"[OK] Retrieved {len(hits)}/{keyword_topk} hits from '{collection_name}' mode=keyword"
            )
            return hits
        elif retriever_mode == "hybrid":
            res = collection.query.hybrid(
                query=question,
                vector=question_vector,
                alpha=hybrid_alpha,
                limit=topk,
                query_properties=keyword_props,
                filters=w_filter,
                return_properties=return_props,
            )
            hits = [_hit_from_obj(o) for o in getattr(res, "objects", [])]
            logger.info(
                f"[OK] Retrieved {len(hits)}/{topk} hits from '{collection_name}' mode=hybrid"
            )
            return hits
        elif retriever_mode == "fusion":
            # Vector leg
            res_vec = collection.query.near_vector(
                near_vector=question_vector,
                limit=vector_topk,
                filters=w_filter,
                return_properties=return_props,
                include_vector=False,
            )
            vec_hits = [_hit_from_obj(o) for o in getattr(res_vec, "objects", [])]
            # BM25 leg
            res_kw = collection.query.bm25(
                query=question,
                limit=keyword_topk,
                query_properties=keyword_props,
                filters=w_filter,
                return_properties=return_props,
            )
            kw_hits = [_hit_from_obj(o) for o in getattr(res_kw, "objects", [])]
            merged = _rrf_merge(vec_hits, kw_hits, rrf_k, merge_topk)
            logger.info(
                f"[OK] Retrieved vec={len(vec_hits)}/{vector_topk}, kw={len(kw_hits)}/{keyword_topk}, merged={len(merged)} mode=fusion"
            )
            return merged
        else:
            raise ValueError(f"Unsupported retriever_mode: {retriever_mode}")
    finally:
        close_client(client)
