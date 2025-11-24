# src/graph/nodes/generate.py
"""
generate.py
-----------
This module defines nodes for generating natural language answers based on a user's question and retrieved document chunks using llms.
"""
from typing import List, Dict, Any
import re
from adapters.ollama import _generate_ollama
from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.prompts import load_prompt, render_prompt

logger = get_logger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")
def _extract_idx_from_text(text: str, max_idx: int) -> List[int]:
    seen, out = set(), []
    for m in _CITE_RE.finditer(text or ""):
        try:
            i = int(m.group(1))
            if 1 <= i <= max_idx and i not in seen:
                seen.add(i); out.append(i)
        except:
            pass
    return out


def _pack_citations(hits: List[Dict[str, Any]], idxs: List[int]) -> List[Dict[str, Any]]:
    """LLM response index → actual meta data mapping to dict"""
    out = []
    for i in idxs:
        if 1 <= i <= len(hits):
            h = hits[i-1]
            out.append({
                "i": i,
                "source_doc": h.get("source_doc"),
                "chunk_id":   h.get("chunk_id"),
                "element":    h.get("type"),
                "page_start": h.get("page_start"),
                "page_end":   h.get("page_end"),
                "text":       h.get("text"),
            })
    return out

def _build_messages(question: str, topk: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Builds the list of messages for the language model prompt.

    This function constructs the system and user messages based on a predefined
    prompt template. It injects the user's question and the retrieved
    document chunks (top-k) into the user prompt.

    Args:
        question: The user's question.
        topk: A list of retrieved document chunks to be used as context.

    Returns:
        A list of message dictionaries formatted for the chat model,
        typically containing a system message and a user message.
    """
    prompt = load_prompt("qa_prompt")
    system = prompt["system"]
    
    _topk = [
        {
            "source_doc": k.get("source_doc"),
            "chunk_id": k.get("chunk_id"),
            "element": k.get("type"),
            "text": (k.get("text") or "").strip(),
            "page_start": k.get("page_start"),
            "page_end": k.get("page_end"),   
        }
        for k in topk if (k.get("text") or "").strip()
    ]
    
    user = render_prompt(prompt["user"], question=question, topk=list(enumerate(_topk, start=1)))
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def generator(question: str, hits: List[Dict[str, Any]]) -> dict:
    """Generates an answer to a question using retrieved documents.

    This function orchestrates the generation process. It first checks if any
    documents were retrieved. If so, it builds the prompt messages, calls the
    configured language model provider (e.g., Ollama) to get an answer,
    and then extracts citations from the answer.

    Args:
        question: The user's question.
        hits: The list of retrieved document chunks to use as context.

    Returns:
        A dictionary containing the generated 'answer', a list of 'Source'
        document indices that were cited in the answer, and the number of
        documents 'used' to generate the answer.
    """
    if not hits:
        logger.warning("[WARN] No retrieved documents found. Returning empty response.")
        return {"answer": "No Answer", "citations": []}
    
    # load config
    cfg = load_config()
    gsec = get_section(cfg, "generate")
    provider = gsec.get("provider", "ollama")
    model = gsec.get("model", "qwen3:8b")
    
    # build messages
    message = _build_messages(question, hits)
    
    # generate answer
    if provider == "ollama":
        response = _generate_ollama(model, message)
    else:
        raise NotImplementedError(f"[ERROR] Provider '{provider}' is not supported.")
    
    # Get answer + citations from generated response
    answer = response.get("answer", "") # {'answer': str, 'citations_idx': List[int]}
    idxs   = [int(x) for x in (response.get("citations") or []) if 1 <= int(x) <= len(hits)]

    if not idxs:
        idxs = _extract_idx_from_text(answer, len(hits))

    citations = _pack_citations(hits, idxs)
    
    return {"answer": answer, "citations": citations, "citations_idx": idxs}
