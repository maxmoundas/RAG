import streamlit as st
import os
import tempfile
import warnings
from datetime import datetime
from dotenv import load_dotenv
from rag_system.rag_engine import RAGEngine
import plotly.graph_objects as go
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch.classes.*")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

# Suppress HuggingFace tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 350px;
        background-color: #555;
        color: #fff;
        text-align: left;
        border-radius: 6px;
        padding: 12px;
        position: fixed;
        z-index: 999999;
        top: 50%;
        left: 20%;
        transform: translate(-50%, -50%);
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 14px;
        line-height: 1.5;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    .tooltip .tooltiptext::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #555 transparent transparent transparent;
    }
    .source-chunks {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #333333;
    }
    .chunk-content {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 0.25rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        max-height: 200px;
        overflow-y: auto;
        color: #333333;
        background-color: #f8f9fa;
    }
    .similarity-score {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1976d2;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "current_time" not in st.session_state:
    st.session_state.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
if "data_cleared" not in st.session_state:
    st.session_state.data_cleared = False


def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG System</h1>', unsafe_allow_html=True)

    # Show notification if data was recently cleared
    if st.session_state.data_cleared:
        st.info("🗑️ All data has been cleared. You can now upload new documents.")

    # Sidebar
    with st.sidebar:
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            st.success(f"Uploaded {len(uploaded_files)} files")

            # Show uploaded files
            st.subheader("Uploaded Files:")
            for file in uploaded_files:
                st.write(f"📄 {file.name} ({file.size} bytes)")

        # Process documents button
        process_button = st.button(
            "🔄 Process Documents",
            type="primary",
        )
        if process_button:
            if st.session_state.uploaded_files:
                with st.spinner("Processing documents..."):
                    # Save uploaded files temporarily and track original names
                    temp_files = []
                    original_names = []
                    for uploaded_file in st.session_state.uploaded_files:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                        ) as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_files.append(tmp_file.name)
                            original_names.append(uploaded_file.name)

                    # Process documents with original filenames
                    result = st.session_state.rag_engine.process_documents(
                        temp_files, original_names
                    )

                    # Clean up temporary files
                    for temp_file in temp_files:
                        os.unlink(temp_file)

                    if result["success"]:
                        st.session_state.data_cleared = False
                        st.success(result["message"])
                    else:
                        st.error(result["message"])
                        # If it failed due to vector store issues, suggest reinitializing
                        if "Vector store is not ready" in result["message"]:
                            st.info(
                                "💡 Try clicking '🔄 Reinitialize Vector Store' button above to fix this issue."
                            )
            else:
                st.warning("Please upload documents first")

        st.divider()

        # System status and analytics
        st.header("📊 System Status & Analytics")
        status = st.session_state.rag_engine.get_system_status()

        # Document chunks
        if st.session_state.data_cleared:
            st.markdown(
                """
                <div class="tooltip">
                    📄 Document Chunks: 0
                    <span class="tooltiptext">
                        Number of document chunks stored in the vector database. Chunks are created by splitting documents into smaller, semantically meaningful pieces for better retrieval.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="tooltip">
                    📄 Document Chunks: {status['vector_store']['document_count']}
                    <span class="tooltiptext">
                        Number of document chunks stored in the vector database. Chunks are created by splitting documents into smaller, semantically meaningful pieces for better retrieval.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Conversations
        st.markdown(
            f"""
            <div class="tooltip">
                💬 Conversations: {status['conversation_history']['total_questions']}
                <span class="tooltiptext">
                    Total number of questions asked in the current session. This tracks user interaction and helps monitor system usage patterns.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Chunk settings
        st.markdown(
            f"""
            <div class="tooltip">
                📝 Chunk Size: {status['document_processor']['chunk_size']}
                <span class="tooltiptext">
                    Maximum number of characters per document chunk. Smaller chunks provide more precise retrieval but may lose context, while larger chunks maintain context but may be less specific.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="tooltip">
                📝 Chunk Overlap: {status['document_processor']['chunk_overlap']}
                <span class="tooltiptext">
                    Number of characters that overlap between consecutive chunks. Overlap helps maintain context across chunk boundaries and improves retrieval quality.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Retrieval settings
        if "retrieval_settings" in status:
            st.subheader("🔍 Retrieval Settings")
            st.markdown(
                f"""
                <div class="tooltip">
                    🔍 Search Candidates: {status['retrieval_settings']['default_k']}
                    <span class="tooltiptext">
                        Number of document chunks initially retrieved from the vector database. Higher values provide more candidates for reranking but may include less relevant content.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="tooltip">
                    📊 Final Chunks: {status['retrieval_settings']['max_chunks']}
                    <span class="tooltiptext">
                        Number of document chunks returned after filtering and reranking. These are the most relevant chunks used to generate the answer.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="tooltip">
                    🎯 Min Similarity: {status['retrieval_settings']['min_similarity_threshold']}
                    <span class="tooltiptext">
                        Minimum similarity threshold for filtering chunks. Only chunks with similarity scores above this threshold are considered relevant.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="tooltip">
                    🔄 Query Expansion: {'✅ Enabled' if status['retrieval_settings']['use_query_expansion'] else '❌ Disabled'}
                    <span class="tooltiptext">
                        Whether to expand the user's query with related terms to improve retrieval. This helps find relevant chunks even when the exact terms don't match.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="tooltip">
                    📈 Reranking: {'✅ Enabled' if status['retrieval_settings']['use_reranking'] else '❌ Disabled'}
                    <span class="tooltiptext">
                        Whether to rerank retrieved chunks based on content relevance. This combines semantic similarity with term overlap and positioning for better relevance.
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Vector store health
        if st.session_state.data_cleared:
            vector_health = "🟡 Reinitialized"
        elif status["vector_store"]["initialized"]:
            vector_health = "🟢 Healthy"
        else:
            vector_health = "🔴 Not Initialized"
        st.markdown(
            f"""
            <div class="tooltip">
                Vector Store: {vector_health}
                <span class="tooltiptext">
                    Status of the vector database (ChromaDB). The vector store stores document embeddings and enables semantic similarity search for retrieving relevant context during question answering.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # LLM health
        llm_health = (
            "🟢 Available" if status["llm"]["available"] else "🔴 Not Available"
        )
        st.markdown(
            f"""
            <div class="tooltip">
                LLM: {llm_health}
                <span class="tooltiptext">
                    Status of the Large Language Model (OpenAI GPT). The LLM generates answers based on retrieved context from the vector store, implementing the 'generation' part of RAG.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        clear_button = st.button(
            "🗑️ Clear All Data",
        )
        if clear_button:
            result = st.session_state.rag_engine.clear_data()
            if result["success"]:
                st.session_state.data_cleared = True
                st.session_state.uploaded_files = []
                st.success(result["message"])
                # Force a rerun to update the UI immediately
                st.rerun()
            else:
                st.error(result["message"])

        reinit_button = st.button(
            "🔄 Reinitialize Vector Store",
        )
        if reinit_button:
            if st.session_state.rag_engine.vector_store.force_reinitialize():
                st.success("Vector store reinitialized successfully")
                st.rerun()
            else:
                st.error("Failed to reinitialize vector store")

        export_button = st.button(
            "📤 Export Conversation",
        )
        if export_button:
            export_text = st.session_state.rag_engine.export_conversation()
            st.download_button(
                label="Download Conversation",
                data=export_text,
                file_name=f"rag_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
            )

    # Main content area
    st.header("💬 Chat Interface")

    # Chat input
    question = st.text_input(
        "Ask a question about your documents:",
        placeholder="What would you like to know about your documents?",
        key="question_input",
    )

    # Ask button
    ask_button = st.button(
        "Ask",
        type="primary",
    )
    if ask_button or question:
        if question:
            with st.spinner("Processing your question..."):
                # Update current time
                st.session_state.current_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Get answer
                result = st.session_state.rag_engine.ask_question(question)

                if result["success"]:
                    # Display answer
                    st.markdown(
                        '<div class="chat-message assistant-message">',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"**🤖 Assistant:** {result['answer']}")

                    # Show sources
                    if result["sources"]:
                        st.markdown('<div class="source-info">', unsafe_allow_html=True)
                        st.markdown(
                            f"**Sources:** {', '.join([os.path.basename(source) for source in result['sources']])}"
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

                    # Show query variations
                    if result.get("query_variations"):
                        st.markdown("**🔍 Query Variations:**")
                        with st.expander(
                            "Click to see how RAG Fusion expanded your question for better retrieval"
                        ):
                            for i, variation in enumerate(
                                result["query_variations"], 1
                            ):
                                variation_style = (
                                    "background-color: #e3f2fd; border-left: 4px solid #2196f3;"
                                    if i == 1
                                    else "background-color: #f3e5f5; border-left: 4px solid #9c27b0;"
                                )
                                st.markdown(
                                    f"""
                                    <div style="{variation_style} padding: 0.75rem; border-radius: 0.25rem; margin: 0.25rem 0; font-family: 'Courier New', monospace; color: #333333;">
                                        <strong>Variation {i}:</strong> {variation}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            st.markdown(
                                f"""
                                <div style="background-color: #fff3; padding: 0.75rem; border-radius: 0.25em; margin: 0.5rem 0; font-size: 0.9em; color: #e65100;">
                                    <strong>💡 How it works:</strong> RAG Fusion uses an LLM to generate multiple ways to ask your question, then searches with all variations and combines the results for better retrieval. This helps find relevant content even when the exact terms don't match.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Show source chunks
                    if result.get("source_chunks"):
                        st.markdown("**📄 Source Chunks:**")
                        with st.expander(
                            "Click to view the document chunks used to generate this answer"
                        ):
                            for chunk in result["source_chunks"]:
                                # Get similarity score if available
                                similarity_info = ""
                                if "similarity_percentage" in chunk:
                                    similarity_info = f'<div class="tooltip"><span class="similarity-score">Similarity: {chunk["similarity_percentage"]}%</span><span class="tooltiptext">Similarity scores show how well each document chunk matches your question. Higher percentages indicate better matches. Scores are calculated using cosine similarity between your question and the document chunks.</span></div>'

                                st.markdown(
                                    f"""
                                    <div class="source-chunks">
                                        <strong>Chunk {chunk['chunk_id']}</strong> (from {os.path.basename(chunk['source'])}) {similarity_info}
                                        <div class="chunk-content">{chunk['content']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    st.markdown("</div>", unsafe_allow_html=True)

                else:
                    st.error(f"Error: {result['answer']}")

    # Display conversation history
    st.subheader("📚 Conversation History")
    history = st.session_state.rag_engine.get_conversation_history()

    if history:
        for i, entry in enumerate(reversed(history), 1):
            # User question
            st.markdown(
                '<div class="chat-message user-message">', unsafe_allow_html=True
            )
            st.markdown(f"**👤 You:** {entry['question']}")
            st.markdown(f"<small>{entry['timestamp']}</small>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Assistant answer
            st.markdown(
                '<div class="chat-message assistant-message">',
                unsafe_allow_html=True,
            )
            st.markdown(f"**🤖 Assistant:** {entry['answer']}")

            if entry["sources"]:
                st.markdown('<div class="source-info">', unsafe_allow_html=True)
                st.markdown(
                    f"**Sources:** {', '.join([os.path.basename(source) for source in entry['sources']])}"
                )
                st.markdown("</div>", unsafe_allow_html=True)

            # Show query variations in conversation history
            if entry.get("query_variations"):
                st.markdown("**🔍 Query Variations:**")
                with st.expander(
                    "Click to see how RAG Fusion expanded this question for better retrieval"
                ):
                    for i, variation in enumerate(entry["query_variations"], 1):
                        variation_style = (
                            "background-color: #e3f2fd; border-left: 4px solid #2196f3;"
                            if i == 1
                            else "background-color: #f3e5f5; border-left: 4px solid #9c27b0;"
                        )
                        st.markdown(
                            f"""
                            <div style="{variation_style} padding: 0.75rem; border-radius: 0.25rem; margin: 0.25rem 0; font-family: 'Courier New', monospace; color: #333333;">
                                <strong>Variation {i}:</strong> {variation}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    st.markdown(
                        f"""
                        <div style="background-color: #fff3; padding: 0.75rem; border-radius: 0.25em; margin: 0.5rem 0; font-size: 0.9em; color: #e65100;">
                            <strong>💡 How it works:</strong> RAG Fusion uses an LLM to generate multiple ways to ask your question, then searches with all variations and combines the results for better retrieval. This helps find relevant content even when the exact terms don't match.
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Show source chunks in conversation history
            if entry.get("source_chunks"):
                st.markdown("**📄 Source Chunks:**")
                with st.expander(
                    "Click to view the document chunks used to generate this answer"
                ):
                    for chunk in entry["source_chunks"]:
                        # Get similarity score if available
                        similarity_info = ""
                        if "similarity_percentage" in chunk:
                            similarity_info = f'<div class="tooltip"><span class="similarity-score">Similarity: {chunk["similarity_percentage"]}%</span><span class="tooltiptext">Similarity scores show how well each document chunk matches your question. Higher percentages indicate better matches. Scores are calculated using cosine similarity between your question and the document chunks.</span></div>'

                        st.markdown(
                            f"""
                            <div class="source-chunks">
                                <strong>Chunk {chunk['chunk_id']}</strong> (from {os.path.basename(chunk['source'])}) {similarity_info}
                                <div class="chunk-content">{chunk['content']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            st.markdown("</div>", unsafe_allow_html=True)
            st.divider()
    else:
        st.info("No conversation history yet. Start by asking a question!")


if __name__ == "__main__":
    main()
