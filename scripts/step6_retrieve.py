# scripts/step5_qa_smoke.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.retrieve import retrieve_topk

if __name__ == "__main__":
    question = "What was PepsiCo's total revenue in 2022?"
    hits = retrieve_topk(question, topk=5, source_doc="28")  # 28.pdf로 제한
    for i, h in enumerate(hits, 1):
        loc = f"{h['doc_id']} (p{h['page_start']}-{h['page_end']})"
        print(f"\n[{i}] {loc} — {h['type']}")
        print(h["text"][:500], "...")