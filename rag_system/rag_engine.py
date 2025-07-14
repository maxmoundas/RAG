from typing import List, Dict, Any, Optional
from .document_processor import DocumentProcessor
from .vector_store import VectorStore
from .llm_interface import LLMInterface
import streamlit as st


class RAGEngine:
    """Main RAG engine that orchestrates all components."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.document_processor = DocumentProcessor(chunk_size, chunk_overlap)
        self.vector_store = VectorStore()
        self.llm_interface = LLMInterface()
        self.conversation_history = []

    def process_documents(
        self, file_paths: List[str], original_names: List[str] = None
    ) -> Dict[str, Any]:
        """Process uploaded documents and add them to the vector store."""
        if not file_paths:
            return {"success": False, "message": "No files provided"}

        try:
            # Ensure vector store is ready
            if not self.ensure_vector_store_ready():
                return {
                    "success": False,
                    "message": "Vector store is not ready. Please try clearing data and uploading again.",
                }

            # Process documents
            documents = self.document_processor.process_files(
                file_paths, original_names
            )

            if not documents:
                return {
                    "success": False,
                    "message": "No documents were successfully processed",
                }

            # Add to vector store
            success = self.vector_store.add_documents(documents)

            if success:
                return {
                    "success": True,
                    "message": f"Successfully processed {len(documents)} document chunks",
                    "chunks": len(documents),
                    "files": len(file_paths),
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to add documents to vector store",
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Error processing documents: {str(e)}",
            }

    def ask_question(self, question: str, k: int = 4) -> Dict[str, Any]:
        """Ask a question and get an answer using the RAG system."""
        if not question.strip():
            return {
                "success": False,
                "answer": "Please provide a question.",
                "sources": [],
                "source_chunks": [],
                "follow_up_questions": [],
            }

        try:
            # Ensure vector store is ready
            if not self.ensure_vector_store_ready():
                return {
                    "success": False,
                    "answer": "Vector store is not ready. Please try clearing data and uploading documents again.",
                    "sources": [],
                    "source_chunks": [],
                    "follow_up_questions": [],
                }

            # Search for relevant documents with scores
            relevant_docs_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
            
            # Extract just the documents for LLM processing
            relevant_docs = [doc for doc, score in relevant_docs_with_scores]

            # Generate answer
            result = self.llm_interface.generate_answer(question, relevant_docs)

            # Prepare source chunks information with scores
            source_chunks = []
            for i, (doc, score) in enumerate(relevant_docs_with_scores):
                # Convert similarity score to a more readable format
                # ChromaDB returns cosine similarity scores (higher is better)
                similarity_percentage = round((1 - score) * 100, 2)  # Convert to percentage
                
                chunk_info = {
                    "chunk_id": i + 1,
                    "content": doc.page_content,
                    "source": (
                        doc.metadata.get("source", "Unknown")
                        if hasattr(doc, "metadata")
                        else "Unknown"
                    ),
                    "chunk_index": (
                        doc.metadata.get("chunk_index", i)
                        if hasattr(doc, "metadata")
                        else i
                    ),
                    "similarity_score": score,
                    "similarity_percentage": similarity_percentage,
                }
                source_chunks.append(chunk_info)

            # Generate follow-up questions
            follow_up_questions = []
            if result["answer"] and not result.get("error", False):
                follow_up_questions = self.llm_interface.generate_follow_up_questions(
                    question, result["answer"]
                )

            # Add to conversation history
            conversation_entry = {
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
                "source_chunks": source_chunks,
                "timestamp": st.session_state.get("current_time", "Unknown"),
            }
            self.conversation_history.append(conversation_entry)

            return {
                "success": True,
                "answer": result["answer"],
                "sources": result["sources"],
                "source_chunks": source_chunks,
                "follow_up_questions": follow_up_questions,
                "relevant_docs": relevant_docs,
            }

        except Exception as e:
            return {
                "success": False,
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "source_chunks": [],
                "follow_up_questions": [],
            }

    def get_system_status(self) -> Dict[str, Any]:
        """Get the current status of all RAG system components."""
        vector_info = self.vector_store.get_collection_info()
        llm_available = self.llm_interface.is_available()

        return {
            "vector_store": {
                "initialized": vector_info["exists"],
                "document_count": vector_info["count"],
                "sources": self.vector_store.get_document_sources(),
            },
            "llm": {
                "available": llm_available,
                "model": (
                    self.llm_interface.model_name if llm_available else "Not configured"
                ),
            },
            "document_processor": {
                "chunk_size": self.document_processor.chunk_size,
                "chunk_overlap": self.document_processor.chunk_overlap,
            },
            "conversation_history": {"total_questions": len(self.conversation_history)},
        }

    def clear_data(self) -> Dict[str, Any]:
        """Clear all data from the RAG system."""
        try:
            # Clear vector store
            vector_cleared = self.vector_store.clear_vector_store()

            # Clear conversation history
            self.conversation_history = []

            # Force a small delay to ensure the vector store is properly reinitialized
            import time

            time.sleep(0.1)

            return {
                "success": vector_cleared,
                "message": (
                    "Data cleared successfully"
                    if vector_cleared
                    else "Failed to clear data"
                ),
            }
        except Exception as e:
            return {"success": False, "message": f"Error clearing data: {str(e)}"}

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history

    def export_conversation(self) -> str:
        """Export conversation history as a formatted string."""
        if not self.conversation_history:
            return "No conversation history to export."

        export_text = "RAG System Conversation History\n"
        export_text += "=" * 40 + "\n\n"

        for i, entry in enumerate(self.conversation_history, 1):
            export_text += f"Q{i}: {entry['question']}\n"
            export_text += f"A{i}: {entry['answer']}\n"
            if entry["sources"]:
                export_text += f"Sources: {', '.join(entry['sources'])}\n"
            if entry.get("source_chunks"):
                export_text += "Source Chunks:\n"
                for chunk in entry["source_chunks"]:
                    export_text += (
                        f"  Chunk {chunk['chunk_id']} (from {chunk['source']}):\n"
                    )
                    export_text += f"  {chunk['content']}\n"
                    export_text += "  ---\n"
            export_text += f"Timestamp: {entry['timestamp']}\n"
            export_text += "-" * 30 + "\n\n"

        return export_text

    def ensure_vector_store_ready(self) -> bool:
        """Ensure the vector store is ready for operations."""
        if not self.vector_store.is_ready():
            # Try to reinitialize
            return self.vector_store.force_reinitialize()
        return True
