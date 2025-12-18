# Financial Document Analyzer

Streamlit-based RAG application for financial PDFs. The pipeline includes Unstructured extraction, chunking, LLM metadata enrichment, embedding, Weaviate ingestion, fusion retrieval, LLM rerank, and answer generation.

## Repository Layout

- `src/ingestion`: element extraction, chunking, metadata generation, embeddings, vectorstore utilities.
- `src/graph`: LangGraph QA pipeline (encode → retrieve → optional rerank → generate). Retrieval supports vector / keyword / hybrid / fusion; rerank uses an LLM judge; generation uses Ollama.
- `configs/default.yaml`: central configuration for every stage (chunking, embeddings, retrieval, rerank, generation, metadata, evaluation, Ollama hosts).
- `cli/ingest[1-5]_*.py`: batch scripts for the ingestion stages.
- `cli/batch_eval.py`: FinanceBench batch evaluator.
- `main.py`: Streamlit UI for querying, filtering, uploading, and resetting.

## Prerequisites

1. Python environment per `pyproject.toml`. Install via `uv sync` or `pip install -e .`.
2. Ollama server, e.g.:
   ```bash
   CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=http://127.0.0.1:11435 ollama serve
   ```
   Models used: `qwen3:8b`, `gpt-oss:20b`, etc.
3. Weaviate Embedded (no Docker). Ports 8090/50061 must be free.
4. Data folders under `data/` (`pdfs`, `processed`, `logs`). Note: `data/logs` is gitignored.

## Pipeline Overview

1. **Elements**  
   ```bash
   python cli/ingest1_elements.py --pdfs data/pdfs
   ```
2. **Chunking**  
   ```bash
   python cli/ingest2_chunking.py --elements data/processed/elements
   ```
3. **Metadata (optional but recommended)**  
   ```bash
   OLLAMA_HOST=http://127.0.0.1:11435 python cli/ingest3_metadata.py --chunks data/processed/chunks
   ```
4. **Embeddings**  
   ```bash
   python cli/ingest4_embed.py
   ```
5. **Vectorstore Upload**  
   ```bash
   python cli/ingest5_vectorstore.py
   ```
6. **Streamlit UI**  
   ```bash
   OLLAMA_HOST=http://127.0.0.1:11435 streamlit run main.py
   ```
7. **Batch Evaluation**  
   ```bash
   PYTHONPATH=src OLLAMA_HOST=http://127.0.0.1:11435 \
   python cli/batch_eval.py --indexed-only --output data/logs/financebench_eval_<ts>.jsonl
   ```

## Key Config (`configs/default.yaml`)

- `paths`: data/log directories, FinanceBench path.
- `chunking`: token/char mode and limits.
- `embedding`: model name, batch size (changing this requires re-embedding + re-upload).
- `retrieve`: mode (`vector|keyword|hybrid|fusion`), default topk, fusion-specific parameters (`vector_topk`, `keyword_topk`, `merge_topk`, `rrf_k`), BM25 properties.
- `rerank`: enable flag, model, candidate count, token limit.
- `generate`: provider/model (default Ollama).
- `metadata`: LLM model, worker count, retry settings.
- `evaluate`: evaluator LLM.
- `ollama`: default host list (overrides the library’s 11434 default when desired).

## Tips & Troubleshooting

- **Port conflicts**: embedded Weaviate requires 8090/50061; stop prior processes if necessary.
- **Ollama port**: default is 11434; set `OLLAMA_HOST` or configure `ollama.hosts` if you run on another port (e.g. 11435).
- **Vector dimension mismatch**: If embeddings were created with an 8B model (4096 dims), queries must use the same. Switching to a 4B embedder means re-running `ingest4_embed.py` and `ingest5_vectorstore.py`.
- **Rerank without metadata**: still works—keywords/summary default to empty strings. Metadata simply improves ranking quality.
- **Logs**: `data/logs` isn’t tracked, so copy out evaluation results if you need to preserve them.

## Future Work

- Explore a metadata-first retrieval pipeline (BM25 on section_title/keywords → vector re-score → LLM rerank).
- Expose retrieval/rerank toggles in the UI.
- Experiment with different fusion parameters, rerank candidate counts, or alternative embedding models.

Use this README as the starting point for handoffs or further documentation.

