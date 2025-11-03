"""
step3_index_dense.py
--------------------
End-to-end indexing pipeline:
1. Load processed chunks
2. Generate embeddings with Qwen3-Embedding-4B
3. Upload vectors to Weaviate collection
4. Verify index stats
"""

import sys
import json
from pathlib import Path
import yaml
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.embed import generate_embeddings
from graph.nodes.weaviate import init_client, create_collection, upload_objects

def load_jsonl(path: Path):
    """Load JSONL file into a list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    """Build dense vector index in Weaviate from chunked documents."""
    print("[STEP 0] Loading configuration...")
    cfg_path = Path("configs/default.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    chunk_dir = Path(cfg["paths"]["chunks_dir"])
    embedding_cfg = cfg["embedding"]
    weaviate_cfg = cfg["weaviate"]

    # 1️⃣ Load all chunk files
    print("[STEP 1] Loading chunk files...")
    chunk_files = sorted(chunk_dir.glob("*.jsonl"))
    if not chunk_files:
        print("[ERROR] No chunk files found in:", chunk_dir)
        return

    all_chunks = []
    for file in chunk_files:
        data = load_jsonl(file)
        for i, d in enumerate(data):
            d["chunk_id"] = f"{d['doc_id']}_{i+1}"
            d["source_doc"] = file.stem.replace("_chunks", "")
        all_chunks.extend(data)
    print(f"[INFO] Loaded {len(all_chunks)} chunks from {len(chunk_files)} files.")

    # 2️⃣ Generate embeddings
    print("[STEP 2] Generating embeddings...")
    embedded = generate_embeddings(
        chunks=all_chunks,
        model_name=embedding_cfg["model_name"],
        batch_size=embedding_cfg["batch_size"],
        normalize_embeddings=embedding_cfg["normalize_embeddings"],
    )

    # 3️⃣ Connect to Weaviate
    print("[STEP 3] Connecting to Weaviate...")
    client = init_client(
        host=weaviate_cfg["host"],
        grpc_port=weaviate_cfg["grpc_port"]
    )

    try:
        # 4️⃣ Ensure collection
        print("[STEP 4] Ensuring collection schema...")
        create_collection(
            client,
            weaviate_cfg["collection_name"],
            embedding_cfg["vector_dimension"]
        )

        # 5️⃣ Upload
        print("[STEP 5] Uploading objects...")
        upload_objects(
            client,
            collection_name=weaviate_cfg["collection_name"],
            objects=embedded
        )

        # 6️⃣ Verify
        print("[STEP 6] Verifying insertion count...")
        collection = client.collections.get(weaviate_cfg["collection_name"])
        result = collection.aggregate.over_all(total_count=True)
        print(f"[OK] Indexed {result.total_count} total objects in '{weaviate_cfg['collection_name']}'")

    finally:
        print("[STEP 7] Closing connection...")
        client.close()
        print("[OK] Connection closed successfully.")

if __name__ == "__main__":
    main()