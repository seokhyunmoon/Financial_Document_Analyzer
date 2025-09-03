from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
)

def split_docs(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks.
    """
    # all_splitted_docs : List[Document] = []
    # for doc in docs:
    #     splitted_docs = _splitter.split_documents([doc])
    #     all_splitted_docs.extend(splitted_docs)
    
    # return all_splitted_docs
    
    return text_splitter.split_documents(docs)
