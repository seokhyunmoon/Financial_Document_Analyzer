# scripts/step8_graph_run.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.app import build_graph

if __name__ == "__main__":
    app = build_graph(use_rerank=False)
    out = app.invoke({"question": "What is the total amount that PepsiCo expects to pay in dividends to shareholders in 2023?",
                      "topk": 5,
                      "source_doc": None})

    print("\n=== ANSWER ===")
    print(out.get("answer", {}).strip())
    print("Citations:", out.get("citations"))
    print("Used:", out.get("used"))