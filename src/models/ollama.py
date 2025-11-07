# src/graph/modesl/ollama.py
from typing import List, Dict
from ollama import Client

def _generate_ollama(model_name: str, messages: List[Dict[str,str]], gsec: dict) -> str:
    
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
        response = client.chat(model=model_name, messages=messages, options={})
        return response['message']['content']
    finally:
        # Manually close the underlying httpx client to prevent ResourceWarning
        if hasattr(client, '_client') and client._client:
            client._client.close()