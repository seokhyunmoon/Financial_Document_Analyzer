from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.documents import Document

class RAGState(TypedDict):
    question: str
    answer: str
    chat_history: Annotated[List[AnyMessage], add_messages]  # 자동 누적

    # ❌ docs: Annotated[list, Document]  (Annotated 메타는 이렇게 쓰지 않음)
    # ✅ 정확한 타입으로
    docs: List[Document]
    # ❌ sources: list[str]
    # ✅ 파일/페이지 등 메타를 구조화
    sources: List[dict]   # e.g. {"id":1,"source":"file.pdf","page":3}
