# scripts/step8_graph_run.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.app import build_graph

if __name__ == "__main__":
    app = build_graph(use_rerank=False)
    out = app.invoke({"question": "What is the total amount that PepsiCo expects to pay in dividends to shareholders in 2023?"})
    
    ans = out.get("answer", {})
    print("\n=== ANSWER ===")
    print(ans.get("answer","").strip())