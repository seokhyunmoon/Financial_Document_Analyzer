"""
rerank.py
---------
LLM-judge reranker that reorders retrieved hits before generation.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional
import tiktoken

from adapters.ollama import ollama_chat_structured
from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.prompts import load_prompt, render_prompt
from graph.schemas import RerankResponse

logger = get_logger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

def _truncate_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to a maximum number of tokens."""
    if not text or max_tokens <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _ENCODER.decode(tokens[:max_tokens]).rstrip() + "..."


def _candidate_content(hit: Dict[str, Any], max_tokens: int) -> str:
    """Build a compact candidate description for LLM reranking.

    Args:
        hit: Retrieved hit with optional metadata fields.
        max_tokens: Maximum token budget for fallback text excerpts.

    Returns:
        Compact text describing the candidate.
    """
    parts: List[str] = []
    section = (hit.get("section_title") or "").strip()
    if section:
        parts.append(f"section_title: {section}")

    keywords = hit.get("keywords") or []
    if isinstance(keywords, list) and keywords:
        joined = ", ".join(k.strip() for k in keywords if k)
        if joined:
            parts.append(f"keywords: {joined}")

    summary = (hit.get("summary") or "").strip()
    if summary:
        parts.append(f"summary: {summary}")

    text = (hit.get("text") or "").strip()
    if text:
        excerpt = _truncate_tokens(text, max_tokens)
        parts.append(f"excerpt: {excerpt}")

    etype = (hit.get("type") or "").strip()
    if etype:
        parts.append(f"type: {etype}")

    return " | ".join(parts) if parts else ""


def rerank_hits(
    hits: List[Dict[str, Any]],
    topk: Optional[int] = None,
    question: Optional[str] = None,
    max_candidates: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Rerank retrieved hits using an LLM judge.

    Args:
        hits: Retrieved chunk records containing text and metadata.
        topk: Maximum number of reranked hits to return; defaults to length of hits.
        question: Original question text (required for LLM rerank).
        max_candidates: Maximum number of candidates to pass into the LLM.

    Returns:
        Reranked hits sorted by relevance, truncated to ``topk``.
    """
    if not hits or not question:
        return hits

    # Load config
    cfg = load_config()
    rerank_cfg = get_section(cfg, "rerank")
    topk = min(topk or rerank_cfg.get("topk", len(hits)), len(hits))
    max_candidates = max_candidates or int(rerank_cfg.get("max_candidates", 5))
    max_candidates = min(max_candidates, len(hits))
    model = rerank_cfg.get("model_name", "qwen3:8b")
    think = rerank_cfg.get("think", None)
    max_tokens = int(rerank_cfg.get("max_tokens", 128))

    # Build candidate list
    candidates = []
    id_to_hit: Dict[int, Dict[str, Any]] = {}
    skipped_hits: List[Dict[str, Any]] = []
    for idx, hit in enumerate(hits[:max_candidates], start=1):
        content = _candidate_content(hit, max_tokens=max_tokens)
        if not content:
            skipped_hits.append(hit)
            continue
        candidates.append({"id": idx, "content": content})
        id_to_hit[idx] = hit
    remaining_hits = hits[max_candidates:]

    if not candidates:
        return hits[:topk]

    prompt = load_prompt("rerank_prompt")
    user = render_prompt(prompt["user"], question=question, candidates=candidates)
    messages = [{"role": "system", "content": prompt["system"]}, {"role": "user", "content": user}]

    try:
        response = ollama_chat_structured(model, messages, RerankResponse, think=think)
    except Exception as exc:
        logger.error("[ERROR] LLM rerank failed: %s", exc)
        return hits[:topk]

    ranked_ids = response.get("ranked_ids") or []
    seen: set[int] = set()
    ordered: List[Dict[str, Any]] = []
    for cid in ranked_ids:
        if cid is None or cid in seen or cid not in id_to_hit:
            continue
        seen.add(cid)
        ordered.append(id_to_hit[cid])

    if not ordered:
        return hits[:topk]
    for idx, hit in id_to_hit.items():
        if idx not in seen:
            ordered.append(hit)
    if skipped_hits:
        ordered.extend(skipped_hits)
    if remaining_hits:
        ordered.extend(remaining_hits)

    logger.info(f"[OK] Reranked hits (top {topk}) using LLM judge")
    return ordered[:topk]
