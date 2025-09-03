from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from langchain_core.documents import Document

class RAGState(TypedDict):
    question: Annotated[str, "Questions"]           #user's question
    answer: Annotated[str, "Answers"]               #generated answer by llm
    chat_history: Annotated[list, add_messages]     #chat history (cumulative)

    docs: Annotated[list, Document]       #lists of docsuments retrieved by vector db
    sources: list[str]                              #metadata of source documents