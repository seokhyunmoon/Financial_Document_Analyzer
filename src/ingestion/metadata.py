"""
metadata.py
-----------
This module enriches document chunks with LLM-generated metadata such as
short summaries and representative keywords.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from adapters.ollama import ollama_chat_structured
from graph.schemas import ChunkMetadata
from utils.config import load_config, get_section
from utils.logger import get_logger
from utils.prompts import load_prompt, render_prompt

logger = get_logger(__name__)


def _normalize_keywords(raw: List[str], max_keywords: int) -> List[str]:
    """Normalize and cap keywords returned by the model.

    Args:
        raw: Raw keyword list from the model.
        max_keywords: Maximum number of keywords to keep.

    Returns:
        Deduplicated, trimmed keywords capped to ``max_keywords``.
    """
    seen: set[str] = set()
    out: List[str] = []
    for item in raw or []:
        cleaned = (item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= max_keywords:
            break
    return out


def _build_messages(
    prompt: Dict[str, str],
    chunk: Dict[str, Any],
    max_keywords: int,
    summary_lines: int,
) -> Optional[List[Dict[str, str]]]:
    """Builds the prompt messages for a single chunk.

    Args:
        prompt: Prompt dict with ``system`` and ``user`` templates.
        chunk: Chunk dictionary to enrich.
        max_keywords: Maximum keyword count to request.
        summary_lines: Target number of summary lines.

    Returns:
        List of chat messages, or ``None`` if the chunk has no usable text.
    """
    text = (chunk.get("text") or "").strip()
    if not text:
        return None
    user = render_prompt(
        prompt["user"],
        section_title=chunk.get("section_title"),
        type=chunk.get("type"),
        text=text,
        max_keywords=max_keywords,
        summary_lines=summary_lines,
    )
    return [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]


def enrich_chunks(chunks: List[Dict[str, Any]], overwrite: bool = False) -> List[Dict[str, Any]]:
    """Enrich chunks with LLM-generated summaries and keywords.

    Args:
        chunks: List of chunk dictionaries from the chunking step.
        overwrite: Whether to regenerate metadata for chunks that already
            contain ``summary`` or ``keywords``.

    Returns:
        List of chunks with ``summary`` and ``keywords`` fields added.
    """
    
    # Load Config
    cfg = load_config()
    msec = get_section(cfg, "metadata", {})
    provider = msec.get("provider", "ollama")
    model_name = msec.get("model_name", "qwen3:8b")
    max_keywords = int(msec.get("max_keywords", 6))
    summary_lines = int(msec.get("summary_lines", 3))
    max_workers = int(msec.get("max_workers", 1))
    retry = int(msec.get("retry", 0))
    ollama_hosts = msec.get("ollama_hosts", []) or []
    if isinstance(ollama_hosts, str):
        ollama_hosts = [ollama_hosts] if ollama_hosts else []
    if not isinstance(ollama_hosts, list):
        ollama_hosts = []

    if provider != "ollama":
        raise NotImplementedError(f"[ERROR] Provider '{provider}' is not supported for metadata.")

    prompt = load_prompt("metadata_prompt")
    def _process_one(index: int, chunk: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        """Generate metadata for a single chunk.

        Args:
            index: Position of the chunk in the input list.
            chunk: Chunk dictionary to enrich.

        Returns:
            Tuple of (index, updated chunk dict).
        """
        # Skip generation if metadata already exists and overwrite is disabled.
        if not overwrite and (chunk.get("summary") or chunk.get("keywords")):
            return index, chunk

        messages = _build_messages(prompt, chunk, max_keywords, summary_lines)
        if messages is None:
            return index, chunk

        # Round-robin across multiple Ollama servers if provided.
        host = None
        if ollama_hosts:
            host = ollama_hosts[index % len(ollama_hosts)]

        last_exc: Optional[Exception] = None
        # Retry transient failures to reduce dropped chunks.
        for attempt in range(retry + 1):
            try:
                response = ollama_chat_structured(
                    model_name,
                    messages,
                    ChunkMetadata,
                    host=host,
                )
                summary = (response.get("summary") or "").strip()
                keywords = _normalize_keywords(response.get("keywords") or [], max_keywords)

                updated = dict(chunk)
                updated["summary"] = summary
                updated["keywords"] = keywords
                return index, updated
            except Exception as exc:
                last_exc = exc
                if attempt < retry:
                    continue
        if last_exc:
            logger.error("[ERROR] Metadata generation failed: %s", last_exc)
        return index, chunk

    if max_workers <= 1:
        # Single-threaded path for deterministic or resource-limited runs.
        enriched = [
            _process_one(idx, chunk)[1]
            for idx, chunk in enumerate(tqdm(chunks, desc="Generating metadata"))
        ]
        logger.info("[OK] Metadata enrichment complete: %d chunks", len(enriched))
        return enriched

    max_workers = max(1, min(max_workers, len(chunks)))
    results: List[Tuple[int, Dict[str, Any]]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Dispatch chunk jobs in parallel, then restore original order by index.
        futures = {
            executor.submit(_process_one, idx, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating metadata"):
            results.append(fut.result())

    results.sort(key=lambda item: item[0])
    enriched = [item[1] for item in results]
    logger.info("[OK] Metadata enrichment complete: %d chunks", len(enriched))
    return enriched
