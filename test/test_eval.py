# test/test_eval.py
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import itertools

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from services.evaluate import qa_evaluate
from graph.state import build_graph
from utils.config import load_config
from utils.logger import get_logger

# Use the custom logger from src/utils/logger.py
logger = get_logger(__name__)

TARGET_DOCS = ["PEPSICO_2022_10K"]  # 특정 문서 스템(파일명에서 .pdf 뺀 것). None이면 전체

def main():
    """Run LangGraph QA over filtered FinanceBench questions."""
    # 1. Load configuration and paths
    cfg = load_config()
    paths = cfg.get("paths", {})
    financebench_path = Path(paths.get("financebench_dir", "data/financebench")) / "financebench_open_source.jsonl"
    pdfs_dir = Path(paths.get("raw_dir", "data/pdfs"))

    logger.info("Building the graph application...")
    app = build_graph(use_rerank=False)

    # 2. Create a list of available PDFs to test against (one-time operation for efficiency)
    try:
        available_pdf_stems = TARGET_DOCS
        # available_pdf_stems = {pdf.stem for pdf in pdfs_dir.glob("*.pdf")}
        if not available_pdf_stems:
            logger.error(f"No PDF files found in {pdfs_dir}. Aborting test.")
            return
        logger.info(f"Found {len(available_pdf_stems)} PDFs to test against: {', '.join(list(available_pdf_stems)[:5])}...")
    except FileNotFoundError:
        logger.error(f"PDF directory not found: {pdfs_dir}")
        return

    # 3. Load the question dataset and filter for relevant questions (efficiently)
    try:
        df = pd.read_json(financebench_path, lines=True)
        # Use pandas' isin for fast filtering
        df = df[df["doc_name"].isin(available_pdf_stems)].copy()
        logger.info(f"Found {len(df)} questions corresponding to the available PDFs.")
    except FileNotFoundError:
        logger.error(f"Question dataset file not found: {financebench_path}")
        return

    if df.empty:
        logger.warning("No matching questions found for the available PDFs in the dataset.")
        return

    # 4. Execute the Evaluation pipeline for the filtered questions
    rows = []
    n_total = 0
    n_correct = 0 
    
    for _, row in itertools.islice(df.iterrows(), 1):
        question = str(row['question']).strip()
        ground_truth = str(row['answer']).strip()
        doc_name = str(row['doc_name']).strip()

        n_total += 1
        logger.info(f"Testing question {n_total}/{len(df)} for document '{doc_name}'...")

        try:
            # Invoke the graph
            out = app.invoke({"question": question, "topk": 10, "source_doc": doc_name})

            # Extract and print the answer
            ans = (out.get("answer", {}) or {}).get("answer", "").strip()
            
            # Evaluate
            eval_result = qa_evaluate(question=question, ground_truth=ground_truth, generated_answer=ans)
            is_same = bool(eval_result.get("is_same"))
            n_correct += 1 if is_same else 0
            
            rows.append({
                "doc_name": doc_name,
                "question": question,
                "ground_truth": ground_truth,
                "answer": ans,
                "num_correct": int(is_same),
                "raw_result": eval_result.get("result", ""),
                "eval_reasoning": eval_result.get("reasoning", ""),
            })

        except Exception as e:
            logger.error(f"Error while processing: {e}", exc_info=True)
            rows.append({
                "doc_name": doc_name,
                "question": question,
                "ground_truth": ground_truth,
                "answer": "",
                "num_correct": 0,
                "raw_result": f"error: {e}",
                "eval_reasoning": "",
            })

    acc = (n_correct / max(1, n_total)) * 100.0
    print(f"\n[RESULT] Q&A Accuracy (LLM-Judge / Figure 4): {acc:.2f}%  ({n_correct}/{n_total})")

    out_dir = Path(paths.get("logs_dir", "data/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"eval_results_{ts}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved per-question results → {out_csv}")

    logger.info("All tests completed.")

if __name__ == "__main__":
    main()
