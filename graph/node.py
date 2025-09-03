from graph import *
from langgraph.graph import StateGraph, START, END


def Retrieve() -> StateGraph:
    """
    Retrieve Node
    """
    graph = StateGraph(name="Retrieve Node", description="Retrieve Node")

    with graph.node(name="Start", state=START):
        pass

    with graph.node(name="End", state=END):
        pass

    with graph.node(name="Retrieve", state=StateGraph):
        """
        Retrieve relevant documents from the vector database based on the user's question.
        """
        pass

    graph.edge("Start", "Retrieve")
    graph.edge("Retrieve", "End")

    return graph