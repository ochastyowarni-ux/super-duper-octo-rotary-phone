#!/usr/bin/env python3
"""
ingest_and_index.py
-------------------
RAG Pipeline Ingestion & Vector Indexing Script for University Academic Advisor System.

Responsibilities:
1. Load university academic catalog / policies markdown/text file (e.g., catalog_cs.md).
2. Segment document into semantic chunks using RecursiveCharacterTextSplitter.
3. Generate vector embeddings using HuggingFace 'all-MiniLM-L6-v2'.
4. Index and persist vector embeddings locally into a FAISS index directory ('vectorstore_db').
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# LangChain Imports with backward/forward compatibility
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


def load_catalog_file(file_path: str) -> list[Document]:
    """Reads a markdown or text document and wraps it in a LangChain Document object."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Academic catalog file not found: {file_path}")

    logger.info(f"Loading academic catalog file from: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        text_content = f.read()

    if not text_content.strip():
        raise ValueError(f"File {file_path} is empty.")

    # Wrap as a LangChain Document with source metadata
    docs = [
        Document(
            page_content=text_content,
            metadata={
                "source": path.name,
                "file_path": str(path.resolve()),
                "total_chars": len(text_content)
            }
        )
    ]
    logger.info(f"Loaded {len(docs)} document(s) with total {len(text_content)} characters.")
    return docs


def split_documents(docs: list[Document], chunk_size: int = 600, chunk_overlap: int = 100) -> list[Document]:
    """Splits documents into manageable chunks preserving markdown headers and section flow."""
    logger.info(f"Splitting documents (chunk_size={chunk_size}, chunk_overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    
    chunks = text_splitter.split_documents(docs)
    logger.info(f"Successfully generated {len(chunks)} text chunks.")
    
    # Display preview of first chunk for verification
    if chunks:
        logger.info(f"Preview of Chunk 1 [length: {len(chunks[0].page_content)} chars]:\n"
                    f"----------------------------------------\n"
                    f"{chunks[0].page_content[:200]}...\n"
                    f"----------------------------------------")
    return chunks


def build_and_save_vectorstore(
    chunks: list[Document], 
    output_dir: str = "vectorstore_db", 
    model_name: str = "all-MiniLM-L6-v2"
) -> FAISS:
    """Builds HuggingFace embeddings and creates/saves FAISS vector index."""
    logger.info(f"Initializing HuggingFaceEmbeddings model: '{model_name}'...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    logger.info(f"Embedding {len(chunks)} chunks and building FAISS vector index...")
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    
    # Save index to local directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    vectorstore.save_local(str(output_path))
    logger.info(f"Vector store successfully saved to local directory: '{output_path.resolve()}'")
    
    # Perform a quick sanity-check retrieval test
    test_query = "What are the prerequisites for Operating Systems?"
    logger.info(f"Running sanity check query: '{test_query}'")
    results = vectorstore.similarity_search_with_score(test_query, k=2)
    for idx, (doc, score) in enumerate(results, start=1):
        logger.info(f"Test Result #{idx} (Distance Score: {score:.4f}):\n{doc.page_content[:150]}...\n")
        
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Ingest and Index University Academic Catalog for RAG.")
    parser.add_argument(
        "--file", 
        type=str, 
        default="catalog_cs.md", 
        help="Path to the academic catalog markdown or text file (default: catalog_cs.md)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="vectorstore_db", 
        help="Output directory to save FAISS vectorstore (default: vectorstore_db)"
    )
    parser.add_argument(
        "--chunk_size", 
        type=int, 
        default=600, 
        help="Chunk size for RecursiveCharacterTextSplitter (default: 600)"
    )
    parser.add_argument(
        "--chunk_overlap", 
        type=int, 
        default=100, 
        help="Chunk overlap for RecursiveCharacterTextSplitter (default: 100)"
    )
    parser.add_argument(
        "--embedding_model", 
        type=str, 
        default="all-MiniLM-L6-v2", 
        help="HuggingFace embedding model name (default: all-MiniLM-L6-v2)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎓 UNIVERSITY ACADEMIC ADVISOR: INGESTION & VECTOR INDEXING")
    print("=" * 70)
    
    try:
        # Step 1: Load file
        docs = load_catalog_file(args.file)
        
        # Step 2: Split text
        chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        
        # Step 3: Embed and index with FAISS
        build_and_save_vectorstore(chunks, output_dir=args.output_dir, model_name=args.embedding_model)
        
        print("=" * 70)
        print("✅ Ingestion & indexing completed successfully!")
        print(f"📁 Local vector store ready at: '{args.output_dir}'")
        print("🚀 You can now launch the advisor app: streamlit run app.py")
        print("=" * 70)
    except Exception as e:
        logger.exception(f"Fatal error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
