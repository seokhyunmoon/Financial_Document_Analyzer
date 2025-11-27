#!/usr/bin/env python
"""Run LangGraph QA against FinanceBench dataset and log results to JSONL."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional

import sys

# add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.state import build_graph
from services.evaluate import qa_evaluate
from utils.config import load_config
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for FinanceBench evaluation."""
    parser = argparse.ArgumentParser(description="FinanceBench batch evaluation")
    parser.add_argument(
        "--docs",
        type=str,
        default="",
        help="Comma-separated doc_name list to restrict evaluation. Default=all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path (JSONL). Default=data/logs/financebench_eval_<ts>.jsonl",
    )
    parser.add_argument(
        "--topk", type=int, default=10, help="Retrieval top-k passed to LangGraph"
    )
    return parser.parse_args()


def iter_questions(dataset_path: Path, allowed_docs: Optional[Iterable[str]]):
    """Yield question rows from the dataset, optionally filtered by doc set.

    Args:
        dataset_path: Path to the FinanceBench JSONL file.
        allowed_docs: Iterable of doc_name strings to include; include all if None/empty.

    Yields:
        Dict rows from the dataset.
    """
    allowed = None
    if allowed_docs:
        allowed = {name.strip() for name in allowed_docs if name.strip()}
    count = 0
    with dataset_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if allowed and row.get("doc_name") not in allowed:
                continue
            yield row
            count += 1

def main() -> None:
    """Execute batch QA + evaluation against FinanceBench."""
    args = parse_args()
    cfg = load_config()
    paths = cfg.get("paths", {})
    dataset_path = Path(paths.get("financebench_dir", "data/financebench")) / "financebench_open_source.jsonl"
    if not dataset_path.exists():
        logger.error(f"FinanceBench dataset not found: {dataset_path}")
        return

    docs = [d for d in args.docs.split(",") if d.strip()] if args.docs else []

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        args.output
        if args.output
        else Path(paths.get("logs_dir", "data/logs")) / f"financebench_eval_{ts}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building LangGraph (QA)...")
    app = build_graph()

    processed = 0
    correct = 0
    errors = 0

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in iter_questions(dataset_path, docs):
            question = str(row.get("question", "")).strip()
            ground_truth = str(row.get("answer", "")).strip()
            doc_name = str(row.get("doc_name", "")).strip()
            question_type = row.get("question_type")
            question_reasoning = row.get("question_reasoning")

            processed += 1
            logger.info("[%s] Q%d: %s", doc_name, processed, question[:120])
            record: Dict[str, Any] = {
                "doc_name": doc_name,
                "question": question,
                "ground_truth": ground_truth,
                "question_type": question_type,
                "question_reasoning": question_reasoning,
            }
            try:
                result = app.invoke(
                    {
                        "question": question,
                        "topk": args.topk,
                        "source_doc": doc_name,
                    }
                )
                answer_block = result.get("answer", {}) or {}
                hits: List[Dict[str, Any]] = result.get("hits", []) or []
                model_answer = answer_block.get("answer", "")
                citations = answer_block.get("citations") or []

                eval_result = qa_evaluate(
                    question=question,
                    ground_truth=ground_truth,
                    generated_answer=model_answer,
                )
                classification = eval_result.get("classification")
                is_correct = classification == "CORRECT"
                correct += 1 if is_correct else 0

                record.update(
                    {
                        "answer": model_answer,
                        "citations": citations,
                        "hits": hits,
                        "eval_classification": classification,
                        "eval_reasoning": eval_result.get("reasoning"),
                    }
                )
            except Exception as exc:
                errors += 1
                logger.exception("Failed to process question %s", question[:80])
                record.update(
                    {
                        "answer": "",
                        "citations": [],
                        "hits": [],
                        "eval_classification": "ERROR",
                        "error": str(exc),
                    }
                )

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Finished. processed=%d correct=%d errors=%d output=%s",
        processed,
        correct,
        errors,
        out_path,
    )


if __name__ == "__main__":
    main()
