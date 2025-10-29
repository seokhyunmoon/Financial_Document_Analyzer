# scripts/step2_chunks.py

import sys
import jsonlines
from pathlib import Path
import argparse

# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.chunks import merge_elements_to_chunks


def main():
    parser = argparse.ArgumentParser(description="Step2: Merge elements into chunks")
    parser.add_argument("--in", dest="input_file", required=True, help="Path to elements JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    doc_id = input_path.stem.replace("_elements", "")
    output_path = Path(f"data/processed/chunks/{doc_id}_chunks.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading elements from: {input_path}")
    with jsonlines.open(input_path, "r") as reader:
        elements = list(reader)

    print(f"[INFO] Merging {len(elements)} elements → chunks...")
    chunks = merge_elements_to_chunks(elements)

    print(f"[INFO] Writing {len(chunks)} chunks to {output_path}")
    with jsonlines.open(output_path, "w") as writer:
        for c in chunks:
            writer.write(c)

    print(f"[OK] Chunking complete: {output_path}")


if __name__ == "__main__":
    main()