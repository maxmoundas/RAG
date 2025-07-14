import os
import chromadb
from typing import List, Dict, Any, Optional
from langchain.schema import Document as LangchainDocument
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import uuid


class VectorStore:
    """Handles vector database operations for the RAG system."""

    def __init__(self, persist_directory: str = "embeddings"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )
        self.vector_store = None
        self._initialize_vector_store()

    def _initialize_vector_store(self):
        """Initialize the vector store."""
        try:
            # Always create a fresh vector store instance
            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
            )

            # Check if the directory exists to determine if it's new or existing
            if os.path.exists(self.persist_directory):
                st.success("Loaded existing vector store")
            else:
                st.info("Created new vector store")

        except Exception as e:
            st.error(f"Error initializing vector store: {str(e)}")
            self.vector_store = None

    def add_documents(self, documents: List[LangchainDocument]) -> bool:
        """Add documents to the vector store."""
        if not self.is_ready():
            st.error("Vector store not initialized or not ready")
            return False

        try:
            # Add documents to the vector store
            self.vector_store.add_documents(documents)

            # Persist the changes
            self.vector_store.persist()

            st.success(f"Added {len(documents)} document chunks to vector store")
            return True
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def similarity_search(self, query: str, k: int = 4) -> List[LangchainDocument]:
        """Perform similarity search on the vector store."""
        if not self.is_ready():
            st.error("Vector store not initialized or not ready")
            return []

        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            st.error(f"Error performing similarity search: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[tuple]:
        """Perform similarity search with scores."""
        if not self.is_ready():
            st.error("Vector store not initialized or not ready")
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return results
        except Exception as e:
            st.error(f"Error performing similarity search with scores: {str(e)}")
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection."""
        if not self.is_ready():
            return {"count": 0, "exists": False}

        try:
            collection = self.vector_store._collection
            count = collection.count()
            return {
                "count": count,
                "exists": True,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            st.error(f"Error getting collection info: {str(e)}")
            return {"count": 0, "exists": False}

    def set_new_directory(self):
        """Set a new unique directory for the vector store and reinitialize."""
        self.persist_directory = f"embeddings_{uuid.uuid4().hex}"
        self._initialize_vector_store()

    def clear_vector_store(self) -> bool:
        """Clear all documents from the vector store and use a new directory."""
        try:
            import shutil
            import gc

            self.vector_store = None
            gc.collect()
            if os.path.exists(self.persist_directory):
                shutil.rmtree(self.persist_directory)
            self.set_new_directory()
            if not self.is_ready():
                st.error("Failed to reinitialize vector store after clearing")
                return False
            st.success("Vector store cleared successfully")
            return True
        except Exception as e:
            st.error(f"Error clearing vector store: {str(e)}")
            return False

    def get_document_sources(self) -> List[str]:
        """Get list of unique document sources in the vector store."""
        if not self.is_ready():
            return []

        try:
            # Get all documents
            all_docs = self.vector_store.get()
            if not all_docs or "metadatas" not in all_docs:
                return []

            # Extract unique sources
            sources = set()
            for metadata in all_docs["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])

            return list(sources)
        except Exception as e:
            st.error(f"Error getting document sources: {str(e)}")
            return []

    def force_reinitialize(self) -> bool:
        """Force reinitialize the vector store."""
        try:
            self.vector_store = None
            self._initialize_vector_store()
            return self.vector_store is not None
        except Exception as e:
            st.error(f"Error reinitializing vector store: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """Check if the vector store is ready for operations."""
        if not self.vector_store:
            return False

        try:
            # Try to access the collection to verify it's working
            collection = self.vector_store._collection
            return collection is not None
        except Exception:
            return False
