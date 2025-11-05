# scripts/step5_query.py
"""Run a semantic search against Weaviate using a custom query embedding.

Assumptions:
- vectordb: vectorizer=None (manual vectors)
- embedding config in configs/default.yaml (model_name, normalize_embeddings)
- We'll embed the query here with the same model Settings used in embed.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

# make src importable
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import load_config, get_section
from utils.logger import get_logger
from graph.nodes.vectordb import init_client, close_client

logger = get_logger(__name__)

def _l2_normalize(vec):
    n = np.linalg.norm(vec)
    return (vec / n).tolist() if n > 0 else vec

def main():
    cfg = load_config()
    esec = get_section(cfg, "embedding")
    vsec = get_section(cfg, "vectordb")
    collection_name = vsec.get("collection_name", "FinancialDocChunk")
    model_name = esec.get("model_name", "Qwen/Qwen3-Embedding-4B")
    normalize = bool(esec.get("normalize_embeddings", True))
    topk = get_section(cfg, "topk") or 5

    # 1) Build a query embedding (same model as embed.py)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    query = "share repurchases and dividend policy in 2022"
    qvec = model.encode([query], batch_size=1, convert_to_numpy=True)[0]
    qvec = _l2_normalize(qvec) if normalize else qvec.tolist()

    logger.info(f"[INFO] Query: {query}")
    logger.info(f"[INFO] Embedding dim: {len(qvec)} (normalized={normalize})")

    # 2) Connect and run near_vector search
    client = init_client()
    try:
        col = client.collections.get(collection_name)
        res = col.query.near_vector(
            near_vector=qvec,
            limit=topk,
            return_metadata=["distance"],
            return_properties=["source_doc", "doc_id", "chunk_id", "element_type", "page_start", "page_end", "text"],
        )

        logger.info(f"[OK] Retrieved {len(res.objects)} hits (topk={topk})")
        for i, obj in enumerate(res.objects, 1):
            meta = obj.metadata or {}
            props = obj.properties or {}
            dist = getattr(meta, "distance", None)
            logger.info(
                f"[{i}] d={dist:.4f} "
                f"{props.get('source_doc')}/{props.get('doc_id')} "
                f"#{props.get('chunk_id')} [{props.get('element_type')}] "
                f"p{props.get('page_start')}-{props.get('page_end')}"
            )
            # Optional: print a short preview
            preview = (props.get("text") or "")[:180].replace("\n", " ")
            logger.info(f"    {preview}...")
    finally:
        close_client(client)

if __name__ == "__main__":
    main()