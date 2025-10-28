from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class State(BaseModel):
    # Chunking
    doc_paths: List[str] = []
    elements: List[Dict[str, Any]] = []
    chunks: List[Dict[str, Any]] = []
    metadata: Dict[str, Dict[str, Any]] = {}
    dense_index_id: Optional[str] = None

    # Query
    question: Optional[str] = None
    q_vector: Optional[List[float]] = None
    hits: List[Dict[str, Any]] = []
    prompt: Optional[str] = None
    answer: Optional[str] = None
    cited_chunk_ids: List[str] = []

    # Control
    run_mode: str = "E2E"   # E2E | INDEX_ONLY | QUERY_ONLY
    topk: int = 10
    trace: Dict[str, Any] = {}