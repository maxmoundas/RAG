import os
import chromadb
from typing import List, Dict, Any, Optional, Tuple
from langchain.schema import Document as LangchainDocument
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import uuid
import re
from config import Config


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

    def _expand_query(self, query: str) -> List[str]:
        """Expand the query with related terms and synonyms."""
        if not Config.USE_QUERY_EXPANSION:
            return [query]

        # Simple query expansion - extract key terms and create variations
        expanded_queries = [query]

        # Extract key terms (words with 4+ characters)
        key_terms = re.findall(r"\b\w{4,}\b", query.lower())

        # Create variations with key terms
        for term in key_terms[:3]:  # Limit to top 3 terms
            if len(term) > 4:
                expanded_queries.append(term)

        return expanded_queries

    def _filter_by_similarity(
        self, results: List[Tuple[LangchainDocument, float]]
    ) -> List[Tuple[LangchainDocument, float]]:
        """Filter results by minimum similarity threshold."""
        threshold = Config.MIN_SIMILARITY_THRESHOLD
        filtered_results = []

        for doc, score in results:
            # Convert cosine similarity to a more intuitive score
            # Higher cosine similarity = more similar, so we want scores above threshold
            if score >= threshold:
                filtered_results.append((doc, score))

        return filtered_results

    def _rerank_by_content_relevance(
        self, query: str, results: List[Tuple[LangchainDocument, float]]
    ) -> List[Tuple[LangchainDocument, float]]:
        """Rerank results based on content relevance to the query."""
        if not Config.USE_RERANKING:
            return results

        def calculate_content_relevance(doc_content: str, query: str) -> float:
            """Calculate content relevance score based on term overlap and positioning."""
            query_terms = set(re.findall(r"\b\w+\b", query.lower()))
            content_terms = set(re.findall(r"\b\w+\b", doc_content.lower()))

            # Term overlap
            overlap = len(query_terms.intersection(content_terms))
            if not query_terms:
                return 0.0

            overlap_ratio = overlap / len(query_terms)

            # Position bonus (terms appearing early in content get higher score)
            position_bonus = 0.0
            content_lower = doc_content.lower()
            for term in query_terms:
                if term in content_lower:
                    pos = content_lower.find(term)
                    # Earlier positions get higher bonus
                    position_bonus += (1.0 - (pos / len(content_lower))) * 0.1

            return overlap_ratio + position_bonus

        # Calculate combined scores
        scored_results = []
        for doc, similarity_score in results:
            content_relevance = calculate_content_relevance(doc.page_content, query)
            # Combine similarity score with content relevance (weighted average)
            combined_score = (similarity_score * 0.7) + (content_relevance * 0.3)
            scored_results.append((doc, combined_score))

        # Sort by combined score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        return scored_results

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

    def advanced_similarity_search(
        self, query: str, k: int = None
    ) -> List[Tuple[LangchainDocument, float]]:
        """Perform advanced similarity search with query expansion, filtering, and reranking."""
        if not self.is_ready():
            st.error("Vector store not initialized or not ready")
            return []

        try:
            # Use default k if not specified
            if k is None:
                k = Config.DEFAULT_K

            # Expand query
            expanded_queries = self._expand_query(query)

            # Collect results from all expanded queries
            all_results = []
            for expanded_query in expanded_queries:
                try:
                    results = self.vector_store.similarity_search_with_score(
                        expanded_query, k=k
                    )
                    all_results.extend(results)
                except Exception as e:
                    st.warning(
                        f"Error with expanded query '{expanded_query}': {str(e)}"
                    )
                    continue

            # Remove duplicates based on document content
            unique_results = {}
            for doc, score in all_results:
                content_hash = hash(doc.page_content)
                if (
                    content_hash not in unique_results
                    or score > unique_results[content_hash][1]
                ):
                    unique_results[content_hash] = (doc, score)

            # Convert back to list
            results = list(unique_results.values())

            # Filter by similarity threshold
            filtered_results = self._filter_by_similarity(results)

            # Rerank by content relevance
            reranked_results = self._rerank_by_content_relevance(
                query, filtered_results
            )

            # Return top results
            max_chunks = Config.MAX_CHUNKS_TO_RETURN
            return reranked_results[:max_chunks]

        except Exception as e:
            st.error(f"Error performing advanced similarity search: {str(e)}")
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
