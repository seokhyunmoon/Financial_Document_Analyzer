# scripts/step3_2_weaviate.py
import json
import sys
from pathlib import Path

# Adjust path so we can import from src
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.weaviate import init_client, create_collection, upload_objects


def main():
    """
    Test the Weaviate client functions (v4):
    1. Connect to Weaviate.
    2. Ensure collection exists.
    3. Upload sample objects.
    4. Verify insertion count.
    """
    host = "http://localhost:8080"
    collection_name = "FinancialDocChunk"
    vector_dim = 2560  # Same as Qwen3-Embedding-4B dimension

    # 1️⃣ Connect to Weaviate
    print("[STEP 1] Connecting to Weaviate...")
    client = init_client(host)

    # 2️⃣ Ensure collection
    print("[STEP 2] Ensuring collection...")
    create_collection(client, collection_name, vector_dim)

    # 3️⃣ Prepare sample objects (mock)
    print("[STEP 3] Uploading sample objects...")
    sample_objects = [
        {
            "source_doc": "test_file.pdf",
            "doc_id": "test_doc_001",
            "chunk_id": "001-1",
            "type": "text",
            "text": "This is a sample chunk about Weaviate embeddings.",
            "page_start": 1,
            "page_end": 1,
            "embedding": [0.001 * i for i in range(vector_dim)],
        },
        {
            "source_doc": "test_file.pdf",
            "doc_id": "test_doc_002",
            "chunk_id": "002-1",
            "type": "title",
            "text": "This chunk talks about vector databases and retrieval.",
            "page_start": 2,
            "page_end": 2,
            "embedding": [0.002 * i for i in range(vector_dim)],
        },
    ]

    # 4️⃣ Upload
    upload_objects(client, collection_name, sample_objects)

    # 5️⃣ Verify count
    print("[STEP 4] Verifying insertion count...")
    collection = client.collections.get(collection_name)
    result = collection.aggregate.over_all(total_count=True)
    print(f"[OK] Verified: {result.total_count} total objects in '{collection_name}'")


if __name__ == "__main__":
    main()