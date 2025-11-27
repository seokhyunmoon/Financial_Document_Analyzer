# src/graph/models/ollama.py
from typing import List, Dict, Any, Type, Optional
from ollama import Client
from pydantic import BaseModel


class QAResponse(BaseModel):
    """Default schema used by the QA generator."""
    answer: str
    citations: list[int] | None = None


def ollama_chat_structured(
    model_name: str,
    messages: List[Dict[str, str]],
    schema_model: Type[BaseModel],
    think: Optional[Any] = None,
) -> Dict[str, Any]:
    """Call the Ollama chat endpoint and validate the response.

    Args:
        model_name: Name of the Ollama model to use.
        messages: Chat message dicts passed to the model.
        schema_model: Pydantic schema used to validate the JSON response.
        think: Optional thinking-mode setting (bool for most models or str such as 'low').

    Returns:
        Parsed response as ``schema_model.model_dump()``.
    """
    
    client = Client()
    try:
        chat_kwargs: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "format": schema_model.model_json_schema(),
            "options": {},
        }
        if think is not None:
            chat_kwargs["think"] = think
        resp = client.chat(**chat_kwargs)
        return schema_model.model_validate_json(resp["message"]["content"]).model_dump()
    finally:
        if hasattr(client, "_client") and client._client:
            client._client.close()
