# imports/index.py
import os, shutil
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from .embedder import get_embeddings  # 같은 임베딩 모델을 계속 재사용

# 인덱스를 디스크에 저장할 경로
INDEX_DIR = "indexes/faiss_index"
os.makedirs("indexes", exist_ok=True)

# 전역 캐시(싱글톤): 프로세스가 살아있는 동안 한 번만 로드해서 재사용
_VS: FAISS | None = None

def _vs() -> FAISS:
    """
    현재 세션에서 사용할 FAISS 인덱스를 '가져오거나(로드) 없으면 만든다'.
    - 있으면: 메모리 캐시(_VS) 즉시 반환
    - 없으면: 디스크에서 load_local(있다면), 없으면 새로 생성해서 save_local
    """
    global _VS
    if _VS is not None:
        return _VS

    emb = get_embeddings()  # 로컬 HF 임베딩(무료)

    if os.path.exists(INDEX_DIR):
        # 이미 저장된 인덱스가 있으면 로드
        _VS = FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)
    else:
        # 빈 인덱스는 만들 수 없어서 '시드' 한 줄로 초기화
        _VS = FAISS.from_texts(["(seed)"], emb, metadatas=[{"source": "__seed__"}])
        _VS.save_local(INDEX_DIR)

    return _VS

def add_chunks(chunks: List[Document]) -> int:
    """
    청크(List[Document])들을 현재 인덱스에 추가하고, 디스크에 저장.
    반환값: 추가된 청크 수
    """
    vs = _vs()
    vs.add_documents(chunks)     # 메모리에 추가
    vs.save_local(INDEX_DIR)     # 디스크에도 저장(재시작 후에도 재사용)
    return len(chunks)

def get_retriever(k: int = 5):
    """
    검색기(retriever) 반환. 나중에 retrieve 노드에서:
      retriever.get_relevant_documents(question)
    로 상위 k개를 가져온다.
    """
    return _vs().as_retriever(search_kwargs={"k": k})

def reset_index():
    """
    인덱스를 완전히 초기화(디스크 폴더 삭제) → 다음 접근 시 _vs()가 새로 생성.
    개발/테스트 중에 '처음부터 다시' 하고 싶을 때 사용.
    """
    global _VS
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
    _VS = None
    # 즉시 새로 만들어 둘 수도 있음(선택):
    _vs()

def index_status() -> str:
    """상태 표시용(선택): UI에 경로/존재 여부 보여주기."""
    return f"Index path: {INDEX_DIR} (exists={os.path.exists(INDEX_DIR)})"
