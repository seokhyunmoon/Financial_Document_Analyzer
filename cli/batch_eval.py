#!/usr/bin/env python
"""
Batch FinanceBench QA evaluator.

Examples:
    # Evaluate every FinanceBench question
    python cli/batch_eval.py

    # Restrict to one document and write to a custom path
    python cli/batch_eval.py --docs AMERICANEXPRESS_2022_10K --output data/logs/amex.jsonl

    # Evaluate only documents currently indexed in the vector store
    python cli/batch_eval.py --indexed-only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import sys

# add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.state import build_graph
from services.evaluate import qa_evaluate
from utils.config import load_config, get_section
from utils.logger import get_logger
from utils.inventory import list_available_documents

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
        "--indexed-only",
        action="store_true",
        help="Restrict evaluation to documents that are currently indexed in the vector store.",
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


def _normalize_hosts(raw: Any) -> List[str]:
    """Normalize host inputs into a list of URLs."""
    if not raw:
        return []
    if isinstance(raw, str):
        return [raw] if raw.strip() else []
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def _resolve_hosts(cfg: Dict[str, Any]) -> List[str]:
    """Resolve Ollama hosts for batch eval with global fallback."""
    batch_cfg = get_section(cfg, "batch_eval", {})
    global_cfg = get_section(cfg, "ollama", {})
    hosts = _normalize_hosts(batch_cfg.get("ollama_hosts"))
    if not hosts:
        hosts = _normalize_hosts(global_cfg.get("hosts"))
    return hosts


def _process_row(
    index: int,
    row: Dict[str, Any],
    app,
    topk: int,
    host: Optional[str],
) -> tuple[int, Dict[str, Any], bool]:
    """Run retrieve/generate/evaluate for a single row."""
    doc_name = str(row.get("doc_name", "")).strip()
    question_type = row.get("question_type")
    question = str(row.get("question", "")).strip()
    ground_truth = str(row.get("answer", "")).strip()
    evidences = row.get("evidence") or []
    evidence_items = [
        {
            "evidence_text": ev.get("evidence_text"),
            "evidence_page_num": ev.get("evidence_page_num"),
        }
        for ev in evidences
        if isinstance(ev, dict)
    ]

    record: Dict[str, Any] = {
        "doc_name": doc_name,
        "question_type": question_type,
        "question": question,
        "ground_truth": ground_truth,
        "evidence": evidence_items,
    }

    try:
        state: Dict[str, Any] = {
            "question": question,
            "topk": topk,
            "source_doc": doc_name,
        }
        if host:
            state["ollama_host"] = host

        result = app.invoke(state)
        answer_block = result.get("answer", {}) or {}
        hits: List[Dict[str, Any]] = result.get("hits", []) or []
        model_answer = answer_block.get("answer", "")
        citations = answer_block.get("citations") or []

        eval_result = qa_evaluate(
            question=question,
            ground_truth=ground_truth,
            generated_answer=model_answer,
            host=host,
        )
        classification = eval_result.get("classification")
        reasoning = eval_result.get("reasoning")

        record.update(
            {
                "answer": model_answer,
                "citations": citations,
                "hits": hits,
                "eval_classification": classification,
                "reasoning": reasoning,
            }
        )
        return index, record, False
    except Exception as exc:
        record.update(
            {
                "answer": "",
                "citations": [],
                "hits": [],
                "eval_classification": "ERROR",
                "error": str(exc),
                "reasoning": "",
            }
        )
        return index, record, True

def main() -> None:
    """Execute batch QA + evaluation against FinanceBench."""
    args = parse_args()
    cfg = load_config()
    qa_cfg = get_section(cfg, "qa")
    topk = int(qa_cfg.get("topk", 10))
    paths = cfg.get("paths", {})
    dataset_path = Path(paths.get("financebench_dir", "data/financebench")) / "financebench_open_source.jsonl"
    if not dataset_path.exists():
        logger.error(f"FinanceBench dataset not found: {dataset_path}")
        return

    docs = [d for d in args.docs.split(",") if d.strip()] if args.docs else []
    if args.indexed_only:
        indexed_docs = [name for name, _ in list_available_documents()]
        if not indexed_docs:
            logger.warning("[WARN] No indexed documents found; nothing to evaluate.")
            return
        if docs:
            orig = docs.copy()
            docs = [d for d in docs if d in indexed_docs]
            missing = sorted(set(orig) - set(docs))
            if missing:
                logger.warning(
                    "[WARN] Skipping %d doc(s) not indexed: %s",
                    len(missing),
                    ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""),
                )
        else:
            docs = indexed_docs

    if args.indexed_only and not docs:
        logger.warning("[WARN] Indexed-only evaluation requested but no matching documents remain.")
        return

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = (
        args.output
        if args.output
        else Path(paths.get("logs_dir", "data/logs")) / f"financebench_eval_{ts}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Building LangGraph (QA)...")
    app = build_graph()

    rows = list(iter_questions(dataset_path, docs))
    if not rows:
        logger.warning("[WARN] No questions to evaluate.")
        return

    batch_cfg = get_section(cfg, "batch_eval", {})
    max_workers = int(batch_cfg.get("max_workers", 1))
    hosts = _resolve_hosts(cfg)

    processed = 0
    errors = 0
    results: List[tuple[int, Dict[str, Any], bool]] = []

    if max_workers <= 1:
        for idx, row in enumerate(rows):
            host = hosts[idx % len(hosts)] if hosts else None
            _, record, is_error = _process_row(idx, row, app, topk, host)
            processed += 1
            errors += 1 if is_error else 0
            results.append((idx, record, is_error))
    else:
        max_workers = max(1, min(max_workers, len(rows)))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _process_row,
                    idx,
                    row,
                    app,
                    topk,
                    hosts[idx % len(hosts)] if hosts else None,
                ): idx
                for idx, row in enumerate(rows)
            }
            for fut in as_completed(futures):
                idx, record, is_error = fut.result()
                processed += 1
                errors += 1 if is_error else 0
                results.append((idx, record, is_error))

    results.sort(key=lambda item: item[0])
    with out_path.open("w", encoding="utf-8") as out_f:
        for _, record, _ in results:
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Finished. processed=%d errors=%d output=%s",
        processed,
        errors,
        out_path,
    )


if __name__ == "__main__":
    main()
