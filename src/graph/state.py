# src/graph/app.py
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from utils.logger import get_logger
from graph.nodes.query import query_embeddings
from graph.nodes.retrieve import retrieve_topk
# from graph.nodes.rerank import rerank
from graph.nodes.generate import generator

logger = get_logger(__name__)

class QAState(TypedDict, total=False):
    question: str
    question_vector: List[float]
    hits: List[Dict[str, Any]]
    answer: Dict[str, Any]
    topk: int
    source_doc: Optional[str]

def node_encode(state: QAState) -> QAState:
    state["question_vector"] = query_embeddings(state["question"])
    return state

def node_retrieve(state: QAState) -> QAState:
    state["hits"] = retrieve_topk(
        state["question"], 
        state["question_vector"],
        state["topk"],
        state["source_doc"]
        )
    return state

# def node_rerank(state: QAState) -> QAState:
#     state["hits"] = rerank(state["question"], state["hits"], topk=state.get("topk", 10))
#     return state

def node_generate(state: QAState) -> QAState:
    state["answer"] = generator(state["question"], state["hits"])
    return state

def build_graph(use_rerank: bool = False):
    g = StateGraph(QAState)
    g.add_node("encode", node_encode)
    g.add_node("retrieve", node_retrieve)
    # if use_rerank: g.add_node("rerank", node_rerank)
    g.add_node("generate", node_generate)

    g.add_edge(START, "encode")
    g.add_edge("encode", "retrieve")
    # if use_rerank: g.add_edge("retrieve","rerank"); g.add_edge("rerank","generate")
    # else:
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", END)
    return g.compile()

# Expose the compiled graph for external use
compiled_graph = build_graph()
