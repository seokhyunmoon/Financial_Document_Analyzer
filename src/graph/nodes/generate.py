# src/graph/nodes/generate.py
"""
generate.py
-----------
This module defines nodes for generating natural language answers based on a user's question and retrieved document chunks using llms.
"""
from typing import List, Dict, Any
from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.prompts import load_prompt, render_prompt

logger = get_logger(__name__)


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
            "doc_id": k.get("doc_id"),
            "page_start": k.get("page_start"),
            "page_end": k.get("page_end"),
            "text": (k.get("text") or "").strip(),
        }
        for k in topk
        if (k.get("text") or "").strip()
    ]
    
    user = render_prompt(prompt["user"], question=question, topk=list(enumerate(_topk, start=1)))
    return [{"role": "system", "content": system},
            {"role": "user",   "content": user}]


def _generate_ollama(model_name: str, messages: List[Dict[str,str]], gsec: dict) -> str:
    """Generates a response using an Ollama chat model.

    Args:
        model_name: The name of the Ollama model to use for generation.
        messages: The list of messages (prompt) to send to the model.
        gsec: The 'generate' section of the configuration dictionary (currently unused).

    Returns:
        The generated answer as a string.
    """
    from ollama import chat
    
    # options = {
    #     "temperature": gsec.get("temperature", 0.1),
    #     "num_ctx": gsec.get("num_ctx", 512),
    #     "top_k": gsec.get("top_k", 10),
    #     "top_p": gsec.get("top_p", 0.8),
    # }
    response = chat(model=model_name, messages=messages, options={})
    return response['message']['content']
    

def generator(question: str, topk: List[Dict[str, Any]]) -> dict:
    """Generates an answer to a question using retrieved documents.

    This function orchestrates the generation process. It first checks if any
    documents were retrieved. If so, it builds the prompt messages, calls the
    configured language model provider (e.g., Ollama) to get an answer,
    and then extracts citations from the answer.

    Args:
        question: The user's question.
        topk: The list of retrieved document chunks to use as context.

    Returns:
        A dictionary containing the generated 'answer', a list of 'Source'
        document indices that were cited in the answer, and the number of
        documents 'used' to generate the answer.
    """
    if not topk:
        logger.warning("[WARN] No retrieved documents found. Returning empty response.")
        return {"answer": "No Answer", "source": [], "used": 0}
    
    # load config
    cfg = load_config()
    gsec = get_section(cfg, "generate")
    provider = gsec.get("provider", "ollama")
    model = gsec.get("model", "qwen2.5:7b-instruct")
    
    # build messages
    message = _build_messages(question, topk)
    
    # generate answer
    if provider == "ollama":
        answer = _generate_ollama(model, message, gsec)
    else:
        raise NotImplementedError(f"[ERROR] Provider '{provider}' is not supported.")
    

    citation = [i for i in range(1, len(topk)+1) if f"[{i}]" in answer]
    return {"answer": answer, "source": citation, "used": len(topk)}