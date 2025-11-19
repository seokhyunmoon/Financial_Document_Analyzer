# src/graph/models/ollama.py
from typing import List, Dict, Any, Type
from ollama import Client
from pydantic import BaseModel

# Ollama response format
class QAResponse(BaseModel):
    answer: str
    citations: list[int] | None = None

def _generate_ollama(model_name: str, messages: List[Dict[str,str]]) -> Dict[str, Any]:
    
    """Generates a response using an Ollama chat model.

    Args:
        model_name: The name of the Ollama model to use for generation.
        messages: The list of messages (prompt) to send to the model.
        gsec: The 'generate' section of the configuration dictionary (currently unused).

    Returns:
        The generated answer as a string.
    """
    client = Client()
    try:
        # options = {
        #     "temperature": gsec.get("temperature", 0.1),
        #     "num_ctx": gsec.get("num_ctx", 512),
        #     "top_k": gsec.get("top_k", 10),
        #     "top_p": gsec.get("top_p", 0.8),
        # }
        response = client.chat(
            model=model_name, 
            messages=messages, 
            format=QAResponse.model_json_schema(),
            options={}
        )
        return QAResponse.model_validate_json(response.message.content).model_dump()
    
    finally:
        # Manually close the underlying httpx client to prevent ResourceWarning
        if hasattr(client, '_client') and client._client:
            client._client.close()
            
            
def ollama_chat_structured(model_name: str, messages: List[Dict[str, str]], schema_model: Type[BaseModel]) -> Dict[str, Any]:
    
    client = Client()
    try:
        resp = client.chat(
            model=model_name,
            messages=messages,
            format=schema_model.model_json_schema(),
        )
        return schema_model.model_validate_json(resp["message"]["content"]).model_dump()
    finally:
        if hasattr(client, "_client") and client._client:
            client._client.close()