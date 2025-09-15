# graph/node.py
from typing import List
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import RAGState


from ingest.index import get_retriever
from models.llm import get_llm

def _format_ctx(docs: List[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source", "?")
        page = meta.get("page")
        h2 = meta.get("h2")
        h3 = meta.get("h3")
        head = f"[{i}] {src}"
        if page: head += f" p.{page}"
        sect = " | ".join([x for x in [h2, h3] if x])
        if sect:
            head += f"  ({sect})"
        parts.append(f"{head}\n{d.page_content}")
    return "\n\n---\n\n".join(parts) if parts else "(no context)"

def retrieve_node(state: RAGState) -> dict:
    """질문으로 상위 k개 문서 검색 → docs 반환"""
    retriever = get_retriever(k=5)
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

    sources = []
    for i, d in enumerate(state["docs"]):
        m = d.metadata or {}
        sources.append({
            "id": i+1,
            "source": m.get("source"),
            "page": m.get("page"),
            "h2": m.get("h2"),
            "h3": m.get("h3"),
        })
    return {"answer": resp.content, "sources": sources}
