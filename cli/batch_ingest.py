#!/usr/bin/env python
"""
Batch-ingest PDFs into the vector store (elements → chunks → embeddings → upload).

Examples:
    # Ingest every PDF under paths.raw_dir
    python cli/batch_ingest.py

    # Ingest a subset of documents and reset the Weaviate collection first
    python cli/batch_ingest.py --docs AMERICANEXPRESS_2022_10K,PEPSICO_2022_10K --reset
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from utils.config import load_config
from utils.logger import get_logger
from services.ingest import ingest_files

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full ingestion pipeline over PDFs.")
    parser.add_argument(
        "--docs",
        type=str,
        default="",
        help="Comma-separated doc names (stems) to ingest. Default processes every PDF.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the vector collection before ingesting.",
    )
    return parser.parse_args()


def _collect_pdfs(raw_dir: Path) -> List[Path]:
    """Return every PDF (case-insensitive) beneath raw_dir."""
    if not raw_dir.exists():
        return []
    pdfs = []
    for path in raw_dir.glob("**/*"):
        if path.is_file() and path.suffix.lower() == ".pdf":
            pdfs.append(path)
    return sorted(pdfs)


def _filter_docs(pdfs: Iterable[Path], names: Iterable[str]) -> List[Path]:
    """Filter PDFs by requested stems (case-sensitive match)."""
    requested = {Path(name).stem for name in names if name.strip()}
    if not requested:
        return list(pdfs)

    matches: List[Path] = []
    missing = set(requested)
    for pdf in pdfs:
        stem = pdf.stem
        if stem in requested:
            matches.append(pdf)
            missing.discard(stem)

    for miss in sorted(missing):
        logger.warning(f"[WARN] Requested document '{miss}' not found in raw_dir.")
    return matches


def main() -> None:
    args = parse_args()
    cfg = load_config()
    paths = cfg.get("paths", {})
    raw_dir = Path(paths.get("raw_dir", "data/pdfs")).resolve()

    docs = [item.strip() for item in args.docs.split(",") if item.strip()] if args.docs else []
    pdfs = _collect_pdfs(raw_dir)
    if not pdfs and not args.reset:
        logger.error(f"[ERROR] No PDFs found under {raw_dir}. Nothing to ingest.")
        return

    targets = _filter_docs(pdfs, docs)
    if not targets and not args.reset:
        logger.warning("[WARN] No matching documents to ingest.")
        return

    logger.info(
        "[INFO] Starting batch ingest",
        extra={
            "raw_dir": str(raw_dir),
            "selected": [p.name for p in targets] if targets else [],
            "reset": args.reset,
        },
    )
    ingest_files(targets, reset=args.reset)
    logger.info("[OK] Batch ingest complete")


if __name__ == "__main__":
    main()
