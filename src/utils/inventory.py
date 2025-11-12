# src/utils/inventory.py
from pathlib import Path
from typing import List, Tuple

from utils.config import load_config, get_section


def list_available_documents(emb_dir: str = "data/processed/embeddings") -> List[Tuple[str, int]]:
    """
    """
    cfg = load_config()
    pcfg = get_section(cfg, "paths")
    emb_dir = pcfg.get("embed_dir", "data/processed/embeddings")
    
    p = Path(emb_dir)
    if not p.exists():
        return []
    docs = []
    for f in sorted(p.glob("*.jsonl")):
        doc_id = f.name.replace(".jsonl", "")
        try:
            with f.open("r", encoding="utf-8") as fh:
                cnt = sum(1 for _ in fh)
        except Exception:
            cnt = 0
        docs.append((doc_id, cnt))
    return docs