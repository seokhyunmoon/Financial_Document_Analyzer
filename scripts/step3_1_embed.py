import json, sys
from pathlib import Path
# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.embed import generate_embeddings

with open("data/processed/chunks/28_chunks.jsonl") as f:
    chunks = [json.loads(line) for line in f]

embedded = generate_embeddings(chunks, batch_size=4)
print(embedded[0].keys())        # expect: includes 'embedding'
print(len(embedded[0]["embedding"]))  # expect: 1536
