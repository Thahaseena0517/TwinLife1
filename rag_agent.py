"""
Module 5 -- rag_agent.py
RAG (Retrieval-Augmented Generation) Pipeline for TwinLife AI.

Chunks guideline documents, embeds them using sentence-transformers
(all-MiniLM-L6-v2), stores in ChromaDB (persistent local collection),
and retrieves relevant context chunks for narration.

Usage:
    rag = RAGAgent()
    rag.build_index()                       # one-time indexing
    chunks = rag.retrieve("blood pressure") # query-time retrieval
"""

from __future__ import annotations

import os
import glob
import warnings

from langchain.text_splitter import RecursiveCharacterTextSplitter
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
#  PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GUIDELINES_DIR = os.path.join(BASE_DIR, "data", "guidelines")
CHROMA_PERSIST_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Embedding model (lightweight, runs locally, no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "twinlife_guidelines"

# Chunking parameters (~500 tokens ≈ ~2000 characters)
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200


class RAGAgent:
    """Retrieval-Augmented Generation agent for TwinLife AI.

    Indexes guideline documents into a persistent ChromaDB collection
    using sentence-transformer embeddings, and retrieves the most
    relevant chunks for a given query topic.

    Attributes:
        embeddings:  HuggingFace sentence-transformer embedding model.
        chroma_client: Persistent ChromaDB client.
        collection:  ChromaDB collection storing guideline chunks.
    """

    def __init__(self) -> None:
        """Initialize the RAG agent with embedding model and ChromaDB."""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.chroma_client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    #  INDEXING
    # ------------------------------------------------------------------
    def build_index(self, force_rebuild: bool = False) -> dict:
        """Chunk and index all guideline documents into ChromaDB.

        Args:
            force_rebuild: If True, deletes existing collection and
                           rebuilds from scratch. Otherwise, skips if
                           the collection already has documents.

        Returns:
            Summary dict: {"documents_processed": int, "chunks_created": int}
        """
        # Check if already indexed
        existing_count = self.collection.count()
        if existing_count > 0 and not force_rebuild:
            return {
                "documents_processed": 0,
                "chunks_created": existing_count,
                "status": "already_indexed",
            }

        # Force rebuild: delete and recreate
        if force_rebuild and existing_count > 0:
            self.chroma_client.delete_collection(COLLECTION_NAME)
            self.collection = self.chroma_client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        # Load all guideline text files
        guideline_files = glob.glob(os.path.join(GUIDELINES_DIR, "*.txt"))
        if not guideline_files:
            raise FileNotFoundError(
                f"No guideline files found in {GUIDELINES_DIR}. "
                "Create .txt files in data/guidelines/ before indexing."
            )

        # Chunk with LangChain's RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks = []
        all_metadatas = []
        all_ids = []
        doc_count = 0

        for filepath in guideline_files:
            filename = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            chunks = splitter.split_text(text)
            doc_count += 1

            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}__chunk_{i}"
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                })
                all_ids.append(chunk_id)

        # Embed and store in ChromaDB
        embeddings = self.embeddings.embed_documents(all_chunks)
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=embeddings,
            metadatas=all_metadatas,
        )

        return {
            "documents_processed": doc_count,
            "chunks_created": len(all_chunks),
            "status": "indexed",
        }

    # ------------------------------------------------------------------
    #  RETRIEVAL
    # ------------------------------------------------------------------
    def retrieve(self, topic: str, top_k: int = 3) -> list[str]:
        """Retrieve the most relevant guideline chunks for a topic.

        Args:
            topic: Natural language query or topic string.
            top_k: Number of top results to return (default: 3).

        Returns:
            List of relevant text chunks, ordered by relevance.
        """
        if self.collection.count() == 0:
            warnings.warn(
                "ChromaDB collection is empty. Call build_index() first."
            )
            return []

        # Embed the query
        query_embedding = self.embeddings.embed_query(topic)

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks = results.get("documents", [[]])[0]
        return chunks

    def retrieve_with_metadata(
        self, topic: str, top_k: int = 3,
    ) -> list[dict]:
        """Retrieve chunks with source metadata and relevance scores.

        Args:
            topic: Natural language query or topic string.
            top_k: Number of top results to return.

        Returns:
            List of dicts: [{"text": str, "source": str, "score": float}, ...]
        """
        if self.collection.count() == 0:
            return []

        query_embedding = self.embeddings.embed_query(topic)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(docs, metas, dists):
            output.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": round(1.0 - dist, 4),  # cosine similarity
            })
        return output


# ======================================================================
#  STANDALONE VERIFICATION (python rag_agent.py)
# ======================================================================
if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  Module 5: RAG Pipeline — Build & Retrieve Test")
    print("=" * 60)

    rag = RAGAgent()

    # Build index (force rebuild for testing)
    print("\n[1] Building index from guideline documents...")
    result = rag.build_index(force_rebuild=True)
    print(f"    Documents processed: {result['documents_processed']}")
    print(f"    Chunks created: {result['chunks_created']}")
    print(f"    Status: {result['status']}")

    # Test retrieval queries
    test_queries = [
        "blood pressure hypertension management",
        "diabetes HbA1c glucose levels",
        "savings rate emergency fund financial health",
        "insurance coverage sum insured riders",
        "BMI weight management obesity",
    ]

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"  Query: \"{query}\"")
        print(f"{'─' * 60}")
        results = rag.retrieve_with_metadata(query, top_k=2)
        for i, r in enumerate(results, 1):
            print(f"\n  [{i}] Source: {r['source']} | Score: {r['score']}")
            # Show first 200 chars of the chunk
            preview = r["text"][:200].replace("\n", " ")
            print(f"      Preview: {preview}...")
