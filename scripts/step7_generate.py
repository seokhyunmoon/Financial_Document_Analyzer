# scripts/step7_generate.py
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.logger import get_logger
from graph.nodes.retrieve import retrieve_topk
from graph.nodes.generate import generator

logger = get_logger(__name__)

QUESTION = "What is the total amount that PepsiCo expects to pay in dividends to shareholders in 2023?"
TOPK = 5
SOURCE_DOC = "28"          # set to None to disable filtering
PREVIEW_LEN = 400          # characters to preview per hit

def main():
    """Retrieve context chunks and generate an answer for the demo question."""
    logger.info(f"[STEP] Retrieve → question='{QUESTION}' (topk={TOPK}, source_doc={SOURCE_DOC})")
    hits = retrieve_topk(QUESTION, topk=TOPK, source_doc=SOURCE_DOC)

    if not hits:
        print("[HITS] 0 results — retrieval returned nothing.")
        print("Answer: No answer")
        return

    print(f"\n[HITS] {len(hits)} results")
    for i, h in enumerate(hits, 1):
        loc = f"{h.get('doc_id')} p{h.get('page_start')}-{h.get('page_end')}"
        typ = h.get("type")
        text = (h.get("text") or "").replace("\n", " ")
        if len(text) > PREVIEW_LEN:
            text = text[:PREVIEW_LEN] + " ..."
        print(f"\n[{i}] {loc} — {typ}")
        print(text)

    logger.info("[STEP] Generate → calling LLM via Ollama (see configs/default.yaml: generator settings)")
    out = generator(QUESTION, hits)

    print("\n=== ANSWER ===")
    print(out.get("answer", "").strip())
    cites = out.get("citations", [])
    if cites:
        print(f"\nCitations present: {cites}")
    print(f"Context used: {out.get('used', 0)} chunk(s)")

if __name__ == "__main__":
    main()
