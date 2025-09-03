# ingest/index.py — Chroma 버전 (FAISS 제거)
import os, shutil
from typing import List
from langchain_core.documents import Document

# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma   # ✅ 새 래퍼

from .embedder import get_embeddings

INDEX_DIR = "indexes/chroma"   # ← 경로만 바뀜
_COLLECTION = "rag"

_VS: Chroma | None = None

def _vs() -> Chroma:
    """
    현재 세션에서 사용할 Chroma 인덱스를 가져오거나 생성.
    persist_directory가 존재하면 자동 로드됨.
    """
    global _VS
    if _VS is not None:
        return _VS
    emb = get_embeddings()
    _VS = Chroma(
        collection_name=_COLLECTION,
        persist_directory=INDEX_DIR,
        embedding_function=emb,
    )
    return _VS

def add_chunks(chunks: List[Document]) -> int:
    vs = _vs()
    vs.add_documents(chunks)
    return len(chunks)

def get_retriever(k: int = 5):
    return _vs().as_retriever(search_kwargs={"k": k})

def reset_index():
    global _VS
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    _VS = None
    _vs()  # 재초기화

def index_status() -> str:
    exists = os.path.exists(INDEX_DIR)
    return f"Index path: {INDEX_DIR} (exists={exists})"
