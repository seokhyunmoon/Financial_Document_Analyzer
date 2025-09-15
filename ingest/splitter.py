from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from ingest.embedder import get_embeddings
from configs.config_loader import load_config, get

config = load_config()

_CHUNK_SIZE = get(config, "retrieval", "chunking", "chunk_size", default=1000)
_CHUNK_OVERLAP = get(config, "retrieval", "chunking", "chunk_overlap", default=150)
_RESET_PER_DOC = get(config, "retrieval", "chunking", "reset_chunk_id_per_doc", default=True)

def recursive_chunking(docs: List[Document], 
                       chunk_size: int = _CHUNK_SIZE, 
                       chunk_overlap: int = _CHUNK_OVERLAP, 
                       reset_chunk_id_per_doc: bool = _RESET_PER_DOC
                       ) -> List[Document]:
    """
    Split documents into fixed-size character chunks.

    This is a baseline splitter using RecursiveCharacterTextSplitter. It preserves
    original metadata and assigns `chunk_id` either per document (1..N) or globally.

    Args:
        docs: Input documents to split.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Overlap characters between adjacent chunks.
        reset_chunk_id_per_doc: If True, chunk_id restarts from 1 for each source
            document (grouped by metadata["source"]). If False, chunk_id is global.

    Returns:
        A list of chunked Documents with updated metadata including `chunk_id`.
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    if reset_chunk_id_per_doc:
        # Assign chunk_id per source document (based on metadata["source"])
        by_source = {}
        for ch in chunks:
            src = (ch.metadata or {}).get("source")
            by_source.setdefault(src, 0)
            by_source[src] += 1
            m = dict(ch.metadata or {})
            m["chunk_id"] = by_source[src]
            ch.metadata = m
    else:
        # Assign a single global sequence of chunk_id
        for i, ch in enumerate(chunks, start=1):
            m = dict(ch.metadata or {})
            m["chunk_id"] = i
            ch.metadata = m
            
    return chunks

#Markdown-header-based chunking with Recursive chunking
def md_recursive_chunking(docs: List[Document],
                          chunk_size: int = _CHUNK_SIZE, 
                          chunk_overlap: int = _CHUNK_OVERLAP,
                          reset_chunk_id_per_doc: bool = True
                          ) -> List[Document]:
    """
    Hybrid splitter: Markdown header -> character-based splitting.

    Pipeline:
      1) Use MarkdownHeaderTextSplitter to split by section headers (h1/h2/h3).
      2) For each header block, use RecursiveCharacterTextSplitter to get uniform chunks.
      3) Merge original metadata with header metadata and assign `chunk_id`.

    Notes:
      - strip_headers=False keeps header text inside chunks, which often helps retrieval.
      - If a document has no headers, it degrades gracefully (acts like plain char split).

    Args:
        docs: Input documents (e.g., .md transcripts) to split.
        chunk_size: Target number of characters per chunk (second-stage splitter).
        chunk_overlap: Overlap characters between adjacent chunks.
        reset_chunk_id_per_doc: If True, chunk_id restarts from 1 for each document;
            otherwise chunk_id is assigned globally across all chunks.

    Returns:
        A list of chunked Documents with merged metadata and `chunk_id`.
    """
    
    headers = [("#","h1"),("##","h2"),("###","h3")]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers,
        strip_headers=False,
    )
    
    rec_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    chunks: List[Document] = []
    
    for doc in docs:
        # 1) split into header blocks (each block carries header metadata in b.metadata)
        md_blocks = md_splitter.split_text(doc.page_content or "")
        local = 0  # reset for each doc

        for block in md_blocks:
            # merge metadata: original + header metadata
            block_text = block.page_content or ""
            merged_meta = dict(doc.metadata or {})
            merged_meta.update(block.metadata or {})

            # 2) split each block into fixed-size character chunks
            char_parts = rec_splitter.split_text(block_text)

            # 3) wrap each string as a Document and assign `chunk_id`
            if reset_chunk_id_per_doc:
                for part in char_parts:
                    local += 1
                    meta = dict(merged_meta)
                    meta["chunk_id"] = local
                    chunks.append(Document(page_content=part, metadata=meta))
            else:
                for part in char_parts:
                    meta = dict(merged_meta)
                    meta["chunk_id"] = len(chunks) + 1
                    chunks.append(Document(page_content=part, metadata=meta))

    return chunks

#Semantic chunking
# Semantic chunking
def semantic_chunking(
    docs: List[Document],
    breakpoint_type: str = "percentile",          # "percentile" | "standard_deviation"
    breakpoint_threshold: float = 0.95,           # e.g., 0.95 for percentile, or 1.0 for std-dev
    reset_chunk_id_per_doc: bool = _RESET_PER_DOC
) -> List[Document]:
    """
    Split documents using semantic boundaries based on embedding similarity.

    - breakpoint_type:
        * "percentile": split when distance > percentile(breakpoint_threshold)
        * "standard_deviation": split when distance > mean + k*std (k = breakpoint_threshold)
    - breakpoint_threshold:
        * for "percentile": 0~1 (e.g., 0.95)
        * for "standard_deviation": positive float (e.g., 1.0)

    Returns:
        List[Document]: semantically chunked documents with chunk_id assigned.
    """
    # 1) build chunker
    chunker = SemanticChunker(
        embeddings=get_embeddings(),
        breakpoint_type=breakpoint_type,
        breakpoint_threshold=breakpoint_threshold,
    )

    # 2) split
    chunks: List[Document] = chunker.split_documents(docs)

    # 3) assign chunk_id
    if reset_chunk_id_per_doc:
        by_source = {}
        for ch in chunks:
            src = (ch.metadata or {}).get("source")
            by_source.setdefault(src, 0)
            by_source[src] += 1
            m = dict(ch.metadata or {})
            m["chunk_id"] = by_source[src]
            ch.metadata = m
    else:
        for i, ch in enumerate(chunks, start=1):
            m = dict(ch.metadata or {})
            m["chunk_id"] = i
            ch.metadata = m

    return chunks
    