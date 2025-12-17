"""
metadata.py
-----------
This module enriches document chunks with LLM-generated metadata such as
short summaries and representative keywords.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

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
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
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
    cfg = load_config()
    msec = get_section(cfg, "metadata", {})
    provider = msec.get("provider", "ollama")
    model_name = msec.get("model_name", "qwen3:8b")
    think = msec.get("think", None)
    max_keywords = int(msec.get("max_keywords", 6))
    summary_lines = int(msec.get("summary_lines", 3))

    if provider != "ollama":
        raise NotImplementedError(f"[ERROR] Provider '{provider}' is not supported for metadata.")

    prompt = load_prompt("metadata_prompt")
    enriched: List[Dict[str, Any]] = []

    for chunk in tqdm(chunks, desc="Generating metadata"):
        if not overwrite and (chunk.get("summary") or chunk.get("keywords")):
            enriched.append(chunk)
            continue

        messages = _build_messages(prompt, chunk, max_keywords, summary_lines)
        if messages is None:
            enriched.append(chunk)
            continue

        try:
            response = ollama_chat_structured(model_name, messages, ChunkMetadata, think=think)
        except Exception as exc:
            logger.error("[ERROR] Metadata generation failed: %s", exc)
            enriched.append(chunk)
            continue

        summary = (response.get("summary") or "").strip()
        keywords = _normalize_keywords(response.get("keywords") or [], max_keywords)

        chunk["summary"] = summary
        chunk["keywords"] = keywords
        enriched.append(chunk)

    logger.info("[OK] Metadata enrichment complete: %d chunks", len(enriched))
    return enriched
