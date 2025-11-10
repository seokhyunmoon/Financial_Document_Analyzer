import json, sys
from pathlib import Path
import jsonlines
# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.embed import generate_embeddings
from utils.files import write_jsonl

pdf = Path("data/processed/chunks/AMERICANEXPRESS_2022_10K_chunks.jsonl")  # For testing a single file
# pdf = Path("data/processed/chunks/BESTBUY_2024Q2_10Q_chunks.jsonl")  # For testing a single file
# pdf = Path("data/processed/chunks/JOHNSON_JOHNSON_2023_8K_dated-2023-08-30_chunks.jsonl")  # For testing a single file
# pdf = Path("data/processed/chunks/PEPSICO_2022_10K_chunks.jsonl")  # For testing a single file
# pdf = Path("data/processed/chunks/ULTABEAUTY_2023Q4_EARNINGS_chunks.jsonl")  # For testing a single file

with open(pdf) as f:
    chunks = [json.loads(line) for line in f]

embedded = generate_embeddings(chunks)
output_path = Path("data/processed/embeddings")

c_out = output_path / f"AMERICANEXPRESS_2022_10K.jsonl"
write_jsonl(str(c_out), embedded)

        
print(embedded[0].keys())        # expect: includes 'embedding'
print(len(embedded[0]["embedding"])) 
