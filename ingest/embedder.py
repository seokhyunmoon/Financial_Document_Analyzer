from langchain_huggingface import HuggingFaceEmbeddings

_EMB = None

def get_embeddings():
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMB