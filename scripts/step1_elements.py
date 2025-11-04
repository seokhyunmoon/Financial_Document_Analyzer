# scripts/step0_2_test_elements_native.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.elements import extract_elements, save_elements_jsonl

def main():
    pdf_path = Path("data/raw/28.pdf")
    doc_id = pdf_path.stem
    out_path = Path("data/processed/elements/") / f"{doc_id}_elements.jsonl"

    elements = extract_elements(str(pdf_path), doc_id)
    save_elements_jsonl(elements, out_path)

    # show first 3 elements (category + text preview)
    print("\n=== Sample Output ===")
    for el in elements[:3]:
        cat = getattr(el, "category", "unknown")
        txt = (el.text[:100] + "...") if len(el.text) > 100 else el.text
        print(f"[{cat}] {txt}")

if __name__ == "__main__":
    main()