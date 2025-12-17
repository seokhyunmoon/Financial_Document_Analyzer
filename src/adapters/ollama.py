# src/graph/models/ollama.py
from typing import List, Dict, Any, Type, Optional
import json

from ollama import Client
from pydantic import BaseModel, ValidationError

from utils.logger import get_logger


logger = get_logger(__name__)


def ollama_chat_structured(
    model_name: str,
    messages: List[Dict[str, str]],
    schema_model: Type[BaseModel],
    think: Optional[Any] = None,
    host: Optional[str] = None,
) -> Dict[str, Any]:
    """Call the Ollama chat endpoint and validate the response.

    Args:
        model_name: Name of the Ollama model to use.
        messages: Chat message dicts passed to the model.
        schema_model: Pydantic schema used to validate the JSON response.
        think: Optional thinking-mode setting (bool for most models or str such as 'low').
        host: Optional Ollama host URL to target (e.g. ``http://127.0.0.1:11435``).

    Returns:
        Parsed response as ``schema_model.model_dump()``.
    """
    
    client = Client(host=host) if host else Client()
    schema = schema_model.model_json_schema()
    schema_str = json.dumps(schema, ensure_ascii=False)
    base_messages = list(messages)
    last_error: Optional[Exception] = None

    try:
        for attempt in range(3):
            chat_kwargs: Dict[str, Any] = {
                "model": model_name,
                "messages": base_messages,
                "format": schema,
                "options": {},
            }
            if think is not None:
                chat_kwargs["think"] = think
            resp = client.chat(**chat_kwargs)
            content = resp["message"]["content"]
            try:
                return schema_model.model_validate_json(content).model_dump()
            except ValidationError as err:
                last_error = err
                logger.warning(
                    "[WARN] Ollama returned invalid JSON (attempt %d/%d): %s",
                    attempt + 1,
                    3,
                    str(err),
                )
                # Append guidance and retry.
                base_messages = list(messages)
                base_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your previous response was not valid JSON. "
                            "Respond again with JSON only, matching this schema:\n"
                            f"{schema_str}"
                        ),
                    }
                )
        if last_error:
            raise last_error
        raise RuntimeError("Failed to obtain structured response from Ollama.")
    finally:
        if hasattr(client, "_client") and client._client:
            client._client.close()
