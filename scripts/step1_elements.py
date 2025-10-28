#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Description:
    Step 1 runner: Convert a single PDF into normalized elements.jsonl using Unstructured.
    Usage:
        uv run python scripts/step1_elements.py --in data/raw/sample1.pdf
"""

import argparse
import sys
from pathlib import Path
import yaml
from typing import Any, Dict

# Local imports
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from adapters.pdf_io import ensure_pdf
from storage.files import write_jsonl
from graph.nodes.elements import extract_elements


def load_config() -> Dict[str, Any]:
    """
    Description:
        Load default YAML config.

    Returns:
        Dict[str, Any]: Config dictionary.
    """
    cfg_path = Path("configs/default.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Description:
        CLI entry: read PDF -> extract elements -> write elements.jsonl.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Path to input PDF")
    args = ap.parse_args()

    cfg = load_config()
    elements_dir = Path(cfg["paths"]["elements_dir"])

    pdf_path, doc_id = ensure_pdf(args.inp)
    elements = extract_elements(pdf_path, doc_id)

    out_path = elements_dir / f"{doc_id}_elements.jsonl"
    write_jsonl(str(out_path), elements)

    print(f"[OK] Elements written: {out_path}")
    print(f" - Count: {len(elements)}")
    # Quick sanity print of first 2 items
    for i, e in enumerate(elements[:2], start=1):
        print(f"   [{i}] type={e['type']} page={e['page']} text[:80]={e['text'][:80]!r}")


if __name__ == "__main__":
    main()