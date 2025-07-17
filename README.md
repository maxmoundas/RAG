# RAG System with Streamlit Frontend

A comprehensive Retrieval-Augmented Generation (RAG) system with a modern Streamlit frontend for document processing, vector search, and AI-powered question answering.

## Features

### Core RAG Functionality
- **Document Processing**: Support for PDF, DOCX, TXT, and MD files
- **Vector Database**: ChromaDB for efficient similarity search and retrieval
- **LLM Integration**: OpenAI GPT models for intelligent answer generation
- **Semantic Search**: Advanced similarity matching with configurable chunk sizes
- **RAG Fusion**: LLM-powered query expansion for enhanced retrieval accuracy

### Advanced UI Features
- **Modern Streamlit Interface**: Clean, responsive web interface with custom styling
- **Real-time Processing**: Live document upload and processing with progress indicators
- **Interactive Chat**: Natural conversation interface with question-answer flow
- **Source Attribution**: Detailed display of which documents and chunks were used for answers
- **Similarity Scores**: Visual similarity percentages showing how well each chunk matches your question
- **Expandable Source Chunks**: Click to view the exact document chunks used for each answer
- **Query Variations Display**: See how RAG Fusion expands your questions for better retrieval

### System Monitoring & Analytics
- **System Status Dashboard**: Real-time monitoring of vector store and LLM health
- **Document Analytics**: Track number of document chunks, conversations, and processing metrics
- **Chunk Configuration Display**: View current chunk size and overlap settings
- **Health Indicators**: Visual status indicators for vector store and LLM availability

### Data Management
- **Conversation History**: Persistent tracking and display of all Q&A interactions
- **Data Export**: Export complete conversation history to text files
- **Data Clearing**: One-click clearing of all documents and conversation data
- **Vector Store Reinitialization**: Manual reinitialization option for troubleshooting

### Advanced Features
- **Tooltip Explanations**: Hover-over tooltips explaining system components and metrics
- **File Upload Management**: Support for multiple file uploads with size tracking
- **Temporary File Handling**: Secure processing of uploaded documents
- **Error Handling**: Comprehensive error messages and recovery suggestions
- **Advanced Retrieval**: Query expansion, similarity filtering, and content-based reranking for better chunk relevance
- **RAG Fusion**: Intelligent query variation generation using LLM for improved retrieval coverage

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

## Usage

### Getting Started
1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, TXT, or MD files
2. **Process Documents**: Click "🔄 Process Documents" to create embeddings and store in vector database
3. **Ask Questions**: Use the chat interface to ask questions about your documents
4. **View Sources**: See which documents and specific chunks were used to generate answers

### Advanced Usage
- **Monitor System Health**: Check the sidebar for real-time system status and analytics
- **View Similarity Scores**: See percentage scores showing how well each document chunk matches your question
- **Explore Source Chunks**: Click expandable sections to view the exact text chunks used for answers
- **Export Conversations**: Download complete conversation history as text files
- **Manage Data**: Clear all data or reinitialize the vector store when needed

### Understanding the Interface

#### Sidebar Features
- **Document Upload**: Drag and drop or select multiple files
- **System Status**: Real-time health indicators for vector store and LLM
- **Analytics Dashboard**: 
  - Document chunks count
  - Total conversations
  - Chunk size and overlap settings
  - Vector store and LLM status
- **Data Management**: Clear data, reinitialize vector store, export conversations

#### Main Chat Interface
- **Question Input**: Natural language questions about your documents
- **Answer Display**: AI-generated responses with source attribution
- **Source Information**: Shows which documents were referenced
- **Similarity Scores**: Percentage scores indicating chunk relevance
- **Expandable Chunks**: Detailed view of source document chunks
- **Query Variations**: Display of how RAG Fusion expanded your question for better retrieval
- **Conversation History**: Complete history of all interactions

## Project Structure

```
RAG/
├── app.py                 # Main Streamlit application with UI
├── config.py              # Configuration settings
├── rag_system/
│   ├── __init__.py
│   ├── document_processor.py  # Document loading and chunking
│   ├── vector_store.py        # ChromaDB vector database operations
│   ├── llm_interface.py       # OpenAI LLM integration
│   ├── rag_fusion.py          # RAG Fusion query expansion system
│   └── rag_engine.py          # Main RAG orchestration engine
├── embeddings_*/           # Vector embeddings storage directories
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## How the RAG System Works

The RAG system follows a classic Retrieval-Augmented Generation architecture with four main components working together. Here's the step-by-step process:

### **1. Document Processing Phase**

**Component**: `DocumentProcessor` class
- **Purpose**: Converts uploaded documents into searchable chunks
- **Process**:
  1. **File Loading**: Supports PDF, DOCX, TXT, and MD files
  2. **Text Extraction**: Extracts raw text from each file
  3. **Chunking**: Splits text into smaller chunks (default 500 characters with 100 character overlap)
  4. **Metadata Addition**: Adds source file information to each chunk

### **2. Vector Storage Phase**

**Component**: `VectorStore` class
- **Purpose**: Creates and manages a searchable vector database
- **Process**:
  1. **Embedding Generation**: Uses HuggingFace's `all-MiniLM-L6-v2` model to convert text chunks into vectors
  2. **Vector Storage**: Stores embeddings in ChromaDB with metadata
  3. **Persistence**: Saves vectors to disk for future use

### **3. Question Processing Phase**

When a user asks a question:

**Step 1: RAG Fusion Query Expansion**
- **Component**: `RAGFusion` class
- **Purpose**: Generates multiple query variations to improve retrieval coverage
- **Process**:
  1. **Original Query**: Takes the user's original question
  2. **LLM-Powered Expansion**: Uses the LLM to generate 5 additional variations considering:
     - Synonyms and related terms
     - Different phrasings and structures
     - Broader and narrower interpretations
     - Technical and non-technical language
     - Different aspects of the same topic
  3. **Query Set**: Creates a set of 6 queries (original + 5 variations)

**Step 2: Multi-Vector Search**
- **Component**: `VectorStore` with `RAGFusion`
- **Purpose**: Searches the vector database with all query variations
- **Process**:
  1. **Parallel Search**: Performs similarity search with each query variation
  2. **Result Collection**: Gathers results from all 6 searches
  3. **Deduplication**: Removes duplicate chunks based on content hash
  4. **Fusion Scoring**: Combines scores using weighted algorithm

**Step 3: Scoring Algorithm**
- **Component**: `RAGFusion._combine_and_rank_results()`
- **Purpose**: Intelligently combines results from multiple queries
- **Process**:
  1. **Score Aggregation**: For each unique chunk, collects all scores from different queries
  2. **Fusion Calculation**: Computes weighted fusion score using:
     - **Max Score** (50% weight): Best similarity score from any query variation
     - **Average Score** (30% weight): Mean of all scores for the chunk
     - **Frequency Bonus** (20% weight): How many queries found this chunk (capped at 3)
  3. **Final Score**: `fusion_score = (max_score × 0.5) + (avg_score × 0.3) + (frequency_bonus × 0.2)`
  4. **Ranking**: Sorts chunks by fusion score in descending order

### **4. Answer Generation Phase**

**Component**: `LLMInterface` class
- **Purpose**: Generates human-like answers using retrieved context
- **Process**:
  1. **Context Preparation**: Combines retrieved document chunks into context
  2. **Prompt Engineering**: Creates a structured prompt with context and question
  3. **LLM Generation**: Uses OpenAI's GPT-4o-mini to generate answer
  4. **Source Tracking**: Tracks which documents were used as sources

### **5. Response Enhancement**

**Additional Features**:
- **Follow-up Questions**: Generates3relevant follow-up questions
- **Source Attribution**: Shows which documents were referenced
- **Similarity Scores**: Displays how relevant each chunk was
- **Conversation History**: Maintains chat history for context
- **Query Variations Display**: Shows all generated query variations for transparency

### **Key Technical Details**

**Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Lightweight, fast, and effective for semantic search
- Runs on CPU for accessibility

**Vector Database**: ChromaDB
- Persistent storage with automatic indexing
- Supports similarity search with scores

**LLM**: OpenAI GPT-4o-mini
  - Configurable temperature (default: 0.7)
  - Structured prompting for consistent responses
  - Used for both answer generation and query expansion

**Chunking Strategy**: RecursiveCharacterTextSplitter
- Splits on natural boundaries (paragraphs, sentences, words)
- Maintains semantic coherence within chunks

**RAG Fusion Configuration**:
- **Query Variations**: 6 total (original + 5 LLM-generated)
- **Fusion Weights**: 50% max score, 30% average score, 20% frequency bonus
- **Fallback**: Simple term extraction if LLM unavailable

### **Data Flow Summary**

```
Documents → Text Extraction → Chunking → Embedding → Vector Store
                                                           ↓
User Question → RAG Fusion → Multiple Query Variations → Similarity Search → Context Retrieval
                                                           ↓
Context + Question → LLM Prompt → Answer Generation → Response
```

This architecture ensures that the system can provide accurate, source-attributed answers based on the specific documents you've uploaded, rather than relying solely on the LLM's pre-trained knowledge. The RAG Fusion component significantly improves retrieval accuracy by exploring multiple ways to interpret and search for the user's question.

## Configuration

### Default Settings
- **Chunk Size**: 500 characters per document chunk
- **Chunk Overlap**: 100 characters overlap between chunks
- **LLM Model**: OpenAI GPT-4o-mini (configurable)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB with persistent storage
- **RAG Fusion**: Enabled by default for enhanced retrieval
- **Retrieval Settings**:
  - **Search Candidates**: 8 initially retrieved
  - **Final Chunks**: 4 chunks returned after filtering/reranking
  - **Min Similarity Threshold**: 0.3 (similarity required)
  - **Query Expansion**: Enabled (expands queries with key terms)
  - **Reranking**: Enabled (combines semantic similarity with content relevance)
  - **Query Variations**: 6 variations per question (original + 5)

### Customization
- Adjust chunk size and overlap in `rag_system/document_processor.py`
- Modify LLM settings in `rag_system/llm_interface.py`
- Configure vector store parameters in `rag_system/vector_store.py`
- Tune retrieval settings in `config.py`:
  - `DEFAULT_K`: Number of initial search candidates
  - `MAX_CHUNKS_TO_RETURN`: Final number of chunks returned
  - `MIN_SIMILARITY_THRESHOLD`: Minimum similarity score for filtering
  - `USE_QUERY_EXPANSION`: Enable/disable query expansion
  - `USE_RERANKING`: Enable/disable content-based reranking

## Features in Detail

### Advanced Retrieval System
The RAG system includes sophisticated retrieval techniques for chunk relevance:

#### **RAG Fusion**
- **LLM-Powered Query Expansion**: Uses an LLM to generate multiple variations of user questions
- **Intelligent Variations**: Creates synonyms, different phrasings, broader/narrower interpretations
- **Fusion Scoring**: Combines results from all variations using weighted scoring (max score, average score, frequency)
- **Enhanced Coverage**: Finds relevant chunks even when exact terms don't match the original question
- **Transparency**: Displays all generated query variations in the interface for user understanding

#### **Query Expansion**
- **Key Term Extraction**: Automatically identifies important terms from user questions
- **Query Variations**: Creates multiple search queries using key terms
- **Broader Coverage**: Finds relevant chunks even when exact terms don't match

#### **Similarity Filtering**
- **Threshold-Based Filtering**: Only includes chunks above minimum similarity threshold
- **Quality Control**: Ensures only relevant chunks are considered for answers
- **Configurable Threshold**: Adjustable minimum similarity score (default: 0.3)

#### **Content-Based Reranking**
- **Term Overlap Analysis**: Calculates how many query terms appear in each chunk
- **Position Weighting**: Gives higher scores to chunks where terms appear earlier
- **Combined Scoring**: Merges semantic similarity with content relevance
- **Weighted Average**: 70% semantic similarity + 30% content relevance

### Similarity Search & Scoring
- **Cosine Similarity**: Advanced similarity calculation between questions and document chunks
- **Percentage Scores**: Visual similarity percentages (0-100%) for each chunk
- **Relevance Ranking**: Automatic ranking of chunks by relevance to your question
- **Tooltip Explanations**: Hover over similarity scores for detailed explanations

### Source Attribution
- **Document Sources**: Shows which files were used for each answer
- **Chunk-Level Detail**: Displays specific document chunks with IDs
- **Content Preview**: Expandable sections showing exact chunk content
- **File Path Display**: Clear indication of source document names

### System Monitoring
- **Health Checks**: Real-time status of vector store and LLM
- **Performance Metrics**: Track document count, conversation count, and processing stats
- **Error Recovery**: Automatic suggestions for common issues
- **Manual Controls**: Options to reinitialize components when needed

### Data Management
- **Session Persistence**: Maintains conversation history and uploaded files
- **Export Functionality**: Download conversations with timestamps and sources
- **Clean Slate**: Complete data clearing for fresh starts
- **File Management**: Secure handling of uploaded documents

## Troubleshooting

### Common Issues
1. **Vector Store Not Ready**: Click "🔄 Reinitialize Vector Store" in sidebar
2. **LLM Not Available**: Check your OpenAI API key in `.env` file
3. **Processing Errors**: Ensure uploaded files are valid PDF, DOCX, TXT, or MD format
4. **Memory Issues**: Clear data periodically if processing large documents

### Performance Tips
- **Chunk Size**: Smaller chunks (500-1000 chars) for precise answers, larger chunks (1000-2000 chars) for context
- **Document Size**: Process documents in batches for better performance
- **Regular Maintenance**: Export conversations and clear data periodically

## License

MIT License - see LICENSE file for details.