import json, sys
from pathlib import Path
import jsonlines
# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.query import query_embeddings

question = "What are the geographies that Pepsico primarily operates in as of FY2022?"
embedding = query_embeddings(question)
print(f"Query Embedding Length: {len(embedding)}")
