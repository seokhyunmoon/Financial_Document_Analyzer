import sys
import jsonlines
from pathlib import Path

# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes._chunks import merge_elements_to_chunks


def main():

    input_path = Path("data/processed/elements/28_elements.jsonl")

    output_path = Path(f"data/processed/chunks/28_chunks.jsonl")

    with jsonlines.open(input_path, "r") as reader:
        elements = list(reader)

    chunks = merge_elements_to_chunks(elements)

    print(f"[INFO] Writing {len(chunks)} chunks to {output_path}")
    with jsonlines.open(output_path, "w") as writer:
        for c in chunks:
            writer.write(c)

    print(f"[OK] Chunking complete: {output_path}")


if __name__ == "__main__":
    main()