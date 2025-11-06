# scripts/step0_ingest_all.py
import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.files import write_jsonl
from graph.nodes.elements import extract_elements
from graph.nodes.chunks import merge_elements_to_chunks
from graph.nodes.embed import generate_embeddings
from graph.nodes.vectordb import init_client, close_client, ensure_collection, upload_objects

logger = get_logger(__name__)

def main():
    cfg = load_config()
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/raw"))
    elements_dir = Path(paths.get("elements_dir", "data/processed/elements"))
    chunks_dir = Path(paths.get("chunks_dir", "data/processed/chunks"))
    emb_dir = Path("data/processed/embeddings")
    collection = get_section(cfg, "vectordb").get("collection_name", "FinancialDocChunk")

    # pdfs = sorted(raw_dir.glob("*.pdf")) # For processing all files
    pdfs = [Path("data/raw/28.pdf")]  # For testing a single file
    if not pdfs:
        logger.warning("No PDFs under data/raw")
        return

    client = init_client()
    try:
        ensure_collection(client, collection)
        for pdf in pdfs:
            doc_id = pdf.stem
            logger.info(f"[DOC] {pdf.name}")

            # 1) elements
            elements = extract_elements(str(pdf), doc_id)
            e_out = elements_dir / f"{doc_id}_elements.jsonl"
            write_jsonl(str(e_out), elements)

            # 2) chunks
            chunks = merge_elements_to_chunks(elements)
            c_out = chunks_dir / f"{doc_id}_chunks.jsonl"
            write_jsonl(str(c_out), chunks)

            # 3) embeddings
            embedded = generate_embeddings(chunks)
            m_out = emb_dir / f"{doc_id}_embedded_chunks.jsonl"
            write_jsonl(str(m_out), embedded)

            # 4) upsert
            upload_objects(client, collection, embedded)
            logger.info(f"[OK] Ingested: {pdf.name} → {len(embedded)} chunks")
    finally:
        close_client(client)

if __name__ == "__main__":
    main()