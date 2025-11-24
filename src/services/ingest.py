from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Any, List, Union
import time

from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.files import write_jsonl
from ingestion.elements import extract_elements
from ingestion.chunking import merge_elements_to_chunks
from ingestion.embeddings import generate_embeddings
from ingestion.vectorstore import (
    init_client,
    close_client,
    ensure_collection,
    reset_collection,
    upload_objects,
)

logger = get_logger(__name__)

def _safe_stem(name: str) -> str:
    stem = Path(name).stem
    return "".join(ch for ch in stem if ch.isalnum() or ch in ("_", "-", "."))[:128]

def _save_uploaded_to_local(uploaded: Union[Path, "UploadedFile"], raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(uploaded, "read"):
        filename = getattr(uploaded, "name", "uploaded.pdf")
        dest = raw_dir / _safe_stem(filename)
        if dest.suffix.lower() != ".pdf":
            dest = dest.with_suffix(".pdf")
        dest.write_bytes(uploaded.read())
        return dest
    else:
        src = Path(uploaded)
        dest = raw_dir / _safe_stem(src.name)
        if dest.suffix.lower() != ".pdf":
            dest = dest.with_suffix(".pdf")
        if src.resolve() != dest.resolve():
            dest.write_bytes(src.read_bytes())
        return dest

def ingest_single_pdf(pdf_path: Path, out_dirs: Dict[str, Path]) -> Dict[str, Any]:
    t0 = time.time()
    doc_id = _safe_stem(pdf_path.name)

    # 1) elements
    elements = extract_elements(str(pdf_path), doc_id)
    e_out = out_dirs["elements_dir"] / f"{doc_id}_elements.jsonl"
    write_jsonl(str(e_out), elements)

    # 2) chunks
    chunks = merge_elements_to_chunks(elements)
    c_out = out_dirs["chunks_dir"] / f"{doc_id}_chunks.jsonl"
    write_jsonl(str(c_out), chunks)

    # 3) embeddings
    embedded = generate_embeddings(chunks)
    m_out = out_dirs["embeddings_dir"] / f"{doc_id}.jsonl"
    write_jsonl(str(m_out), embedded)

    return {
        "doc_id": doc_id,
        "n_elements": len(elements),
        "n_chunks": len(chunks),
        "n_vectors": len(embedded),
        "elapsed_sec": round(time.time() - t0, 2),
        "rows": embedded,  # DB 업서트 입력
        "paths": {"elements": e_out, "chunks": c_out, "embeddings": m_out},
    }

def ingest_files(
    uploaded_files: Iterable[Union[Path, "UploadedFile"]],
    reset: bool = False,
) -> List[Dict[str, Any]]:
    cfg = load_config()
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/pdfs")).resolve()
    out_dirs = {
        "elements_dir": Path(paths.get("elements_dir", "data/processed/elements")).resolve(),
        "chunks_dir":   Path(paths.get("chunks_dir", "data/processed/chunks")).resolve(),
        "embeddings_dir": Path(paths.get("embeddings_dir", "data/processed/embeddings")).resolve(),
    }
    for p in out_dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    collection = get_section(cfg, "vectordb").get("collection_name", "FinancialDocChunk")
    client = init_client()
    try:
        if reset:
            try:
                reset_collection(client, collection)
            except TypeError:
                if client.collections.exists(collection):
                    client.collections.delete(collection)
        ensure_collection(client, collection)

        results: List[Dict[str, Any]] = []
        for up in uploaded_files:
            dest = _save_uploaded_to_local(up, raw_dir)
            logger.info(f"[INGEST] {dest.name}")
            info = ingest_single_pdf(dest, out_dirs)
            upload_objects(client, collection, info["rows"])
            results.append(info)
        return results
    finally:
        close_client(client)
