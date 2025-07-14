# RAG System with Streamlit Frontend

A comprehensive Retrieval-Augmented Generation (RAG) system with a modern Streamlit frontend for document processing, vector search, and AI-powered question answering.

## Features

### Core RAG Functionality
- **Document Processing**: Support for PDF, DOCX, TXT, and MD files
- **Vector Database**: ChromaDB for efficient similarity search and retrieval
- **LLM Integration**: OpenAI GPT models for intelligent answer generation
- **Semantic Search**: Advanced similarity matching with configurable chunk sizes

### Advanced UI Features
- **Modern Streamlit Interface**: Clean, responsive web interface with custom styling
- **Real-time Processing**: Live document upload and processing with progress indicators
- **Interactive Chat**: Natural conversation interface with question-answer flow
- **Source Attribution**: Detailed display of which documents and chunks were used for answers
- **Similarity Scores**: Visual similarity percentages showing how well each chunk matches your question
- **Expandable Source Chunks**: Click to view the exact document chunks used for each answer

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
│   └── rag_engine.py          # Main RAG orchestration engine
├── embeddings_*/           # Vector embeddings storage directories
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Configuration

### Default Settings
- **Chunk Size**: 1000 characters per document chunk
- **Chunk Overlap**: 200 characters overlap between chunks
- **LLM Model**: OpenAI GPT-3.5-turbo (configurable)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: ChromaDB with persistent storage

### Customization
- Adjust chunk size and overlap in `rag_system/document_processor.py`
- Modify LLM settings in `rag_system/llm_interface.py`
- Configure vector store parameters in `rag_system/vector_store.py`

## Features in Detail

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