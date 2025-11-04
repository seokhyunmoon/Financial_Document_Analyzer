import json, sys
from pathlib import Path
import jsonlines
# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.embed import generate_embeddings

with open("data/processed/chunks/28_chunks.jsonl") as f:
    chunks = [json.loads(line) for line in f]

embedded = generate_embeddings(chunks, batch_size=4)
output_path = Path(f"data/processed/embeddings/28_embedded_chunks.jsonl")

with jsonlines.open(output_path, "w") as writer:
    for c in embedded:
        writer.write(c)
        
print(embedded[0].keys())        # expect: includes 'embedding'
print(len(embedded[0]["embedding"]))  # expect: 1536
