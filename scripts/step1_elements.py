import sys, json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.nodes.elements import extract_elements
from utils.config import load_config, get_section

def main():
    # --- Configs ---
    config = load_config()
    paths = get_section(config, "paths")
    data_pdfs = paths.get("raw_dir", "data/raw")
    
    elements_dir = paths.get("elements_dir", "data/processed/elements")
    
    pdf_path = Path("data/raw/28.pdf")
    doc_id = pdf_path.stem
    out_path = Path("data/processed/elements/") / f"{doc_id}_elements.jsonl"

    elements = extract_elements(str(pdf_path), doc_id)

    # --- save as JSONL ---
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in elements:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(elements)} elements → {out_path}")

if __name__ == "__main__":
    main()