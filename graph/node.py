# graph/node.py
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

# 네가 만든 상태 타입을 그대로 사용 (이름만 별칭)
# 기존: from graph.state import StateGraph as RAGState  (혼란 유발)
from graph.state import RAGState


from ingest.index import get_retriever
from models.llm import get_llm

def _format_ctx(docs: List[Document]) -> str:
    """프롬프트에 넣을 컨텍스트 문자열 생성"""
    parts = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page")
        head = f"[{i}] {src}" + (f" p.{page}" if page else "")
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(parts) if parts else "(no context)"

def retrieve_node(state: RAGState) -> dict:
    """질문으로 상위 k개 문서 검색 → docs 반환"""
    retriever = get_retriever(k=5)
    # docs = retriever.get_relevant_documents(state["question"])  # ← 경고 유발
    docs = retriever.invoke(state["question"])  # ← 권장 API
    return {"docs": docs}

def generate_node(state: RAGState) -> dict:
    """컨텍스트를 주입해 LLM이 답변 생성 → answer, sources 반환"""
    llm = get_llm()
    messages = [
        SystemMessage(
            content="Answer ONLY from the context; cite sources like [1][2]. "
                    "If the answer is not in the context, say you don't know."
        ),
        HumanMessage(content=f"Q: {state['question']}\n\nContext:\n{_format_ctx(state['docs'])}")
    ]
    resp = llm.invoke(messages)

    sources = [
        {"id": i+1, "source": d.metadata.get("source"), "page": d.metadata.get("page")}
        for i, d in enumerate(state["docs"])
    ]
    return {"answer": resp.content, "sources": sources}
