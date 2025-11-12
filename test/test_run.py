# test/test_run.py
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from graph.app import build_graph
from utils.config import load_config
from utils.logger import get_logger

# Use the custom logger from src/utils/logger.py
logger = get_logger(__name__)

def main():
    """
    Runs a QA test against all documents found in the data/pdfs folder.
    """
    # 1. Load configuration and paths
    cfg = load_config()
    paths = cfg.get("paths", {})
    financebench_path = Path(paths.get("financebench_dir", "data/financebench")) / "financebench_open_source.jsonl"
    pdfs_dir = Path(paths.get("raw_dir", "data/pdfs"))

    logger.info("Building the graph application...")
    app = build_graph(use_rerank=False)

    # 2. Create a list of available PDFs to test against (one-time operation for efficiency)
    try:
        available_pdf_stems = {pdf.stem for pdf in pdfs_dir.glob("*.pdf")}
        if not available_pdf_stems:
            logger.error(f"No PDF files found in {pdfs_dir}. Aborting test.")
            return
        logger.info(f"Found {len(available_pdf_stems)} PDFs to test against: {', '.join(list(available_pdf_stems)[:5])}...")
    except FileNotFoundError:
        logger.error(f"PDF directory not found: {pdfs_dir}")
        return

    # 3. Load the question dataset and filter for relevant questions (efficiently)
    try:
        all_questions_df = pd.read_json(financebench_path, lines=True)
        # Use pandas' isin for fast filtering
        questions_to_test = all_questions_df[all_questions_df["doc_name"].isin(available_pdf_stems)].copy()
        logger.info(f"Found {len(questions_to_test)} questions corresponding to the available PDFs.")
    except FileNotFoundError:
        logger.error(f"Question dataset file not found: {financebench_path}")
        return

    if questions_to_test.empty:
        logger.warning("No matching questions found for the available PDFs in the dataset.")
        return

    # 4. Execute the QA pipeline for the filtered questions
    for index, row in questions_to_test.iterrows():
        question = row['question']
        ground_truth = row['answer']
        doc_name = row['doc_name']

        print("\n" + "="*60)
        logger.info(f"Testing question {index + 1}/{len(questions_to_test)} for document '{doc_name}'...")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        try:
            # Invoke the graph
            out = app.invoke({"question": question})

            # Extract and print the answer
            ans_dict = out.get("answer", {})
            generated_answer = ans_dict.get("answer", "Could not generate an answer.").strip()

            print("\n--- Generated Answer ---")
            print(generated_answer)
            print("------------------------")

        except Exception as e:
            logger.error(f"An error occurred while processing the question: {e}", exc_info=True)

        print("="*60)

    logger.info("All tests completed.")

if __name__ == "__main__":
    main()