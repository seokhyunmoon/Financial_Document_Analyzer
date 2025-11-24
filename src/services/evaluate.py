# src/services/evaluate.py

from typing import Dict, Any
from pydantic import BaseModel
from adapters.ollama import _generate_ollama, ollama_chat_structured
from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.prompts import load_prompt, render_prompt

logger = get_logger(__name__)

class EvalResponse(BaseModel):
    result: str 
    reasoning: str | None = None

def qa_evaluate(question: str, ground_truth: str, generated_answer: str) -> dict:
    # load config
    cfg = load_config()
    gsec = get_section(cfg, "generate")
    provider = gsec.get("provider", "ollama")
    model = gsec.get("model", "qwen3:8b")
    
    # load prompt and build message
    prompt = load_prompt("eval_prompt")
    user = render_prompt(
        prompt["user"],
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer
    )
    message = [{"role": "user",   "content": user}]
    
    # generate answer
    if provider == "ollama":
        response = ollama_chat_structured(model, message, EvalResponse)
    else:
        raise NotImplementedError(f"[ERROR] Provider '{provider}' is not supported.")
    
    result = (response.get("result") or "").strip()
    first = result.split()[0].strip("\n").lower() if result else ""
    is_same = True if first == "true" else False
    
    return {"result": result, "is_same": is_same, "reasoning": response.get("reasoning")}
