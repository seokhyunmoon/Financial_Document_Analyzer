# src/services/evaluate.py
import json
from typing import Dict, Any
from adapters.ollama import ollama_chat_structured
from graph.schemas import EvalResponse
from utils.logger import get_logger
from utils.config import load_config, get_section
from utils.prompts import load_prompt, render_prompt

logger = get_logger(__name__)

def qa_evaluate(
    question: str,
    ground_truth: str,
    generated_answer: str,
    host: str | None = None,
) -> dict:
    """Compare generated answer with ground truth using an LLM judge.

    Args:
        question: The original question text.
        ground_truth: Reference answer from the dataset.
        generated_answer: Answer produced by the model.
        host: Optional Ollama host override for this request.

    Returns:
        Dict containing the evaluation classification and reasoning.
    """
    # load config
    cfg = load_config()
    esec = get_section(cfg, "evaluate")
    provider = esec.get("provider", "ollama")
    model = esec.get("model_name", "gpt-oss:20b")
    think = esec.get("think", None)
    
    # load prompt and build message
    prompt = load_prompt("eval_prompt")
    system_prompt = prompt.get("system", "")
    user_prompt = render_prompt(
        prompt["user"],
        question=question,
        ground_truth=ground_truth,
        generated_answer=generated_answer
    )
    
    # The new prompt has a dedicated system message.
    message = []
    if system_prompt:
        message.append({"role": "system", "content": system_prompt})
    message.append({"role": "user", "content": user_prompt})

    try:
        logger.info(f"[INFO] Running evaluator provider={provider} model={model}")
        # The `ollama_chat_structured` helper attempts to parse the LLM's JSON
        # output into the `EvalResponse` Pydantic model.
        response_data = ollama_chat_structured(model, message, EvalResponse, think=think, host=host)
        
        if response_data:
            logger.info(f"[OK] Evaluation completed classification={response_data.get('classification')}")
            return {
                "classification": response_data.get("classification", "INCORRECT"),
                "reasoning": response_data.get("reasoning", "Failed to get reasoning from evaluator.")
            }

    except Exception as e:
        logger.error(f"[ERROR] Failed to evaluate response: {e}")

    logger.warning("[WARN] Falling back to default INCORRECT classification")
    # Fallback response in case of parsing failure or other errors
    return {
        "classification": "INCORRECT",
        "reasoning": "Failed to get a valid structured response from the evaluation model."
    }
