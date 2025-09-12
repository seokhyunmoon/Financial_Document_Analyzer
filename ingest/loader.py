from typing import List
import pathlib
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

def load_docs(file_paths: List[str]) -> List[Document]:
    """
    Load documents from the given file paths.
    Supports MD and TXT files.
    """
    
    all_docs : List[Document] = []
    for file_path in file_paths:
        path = pathlib.Path(file_path)
        
        if not path.exists():
            print(f"File {file_path} does not exist.")
            continue
        
        if path.suffix.lower() == '.txt':
            loader = TextLoader(file_path, encoding='utf8')
        elif path.suffix.lower() == '.md':
            loader = TextLoader(file_path, encoding='utf8')
        else:
            print(f"Unsupported file type: {file_path}.")
            continue
        
        docs = loader.load()
        
        all_docs.extend(docs)
    
    return all_docs