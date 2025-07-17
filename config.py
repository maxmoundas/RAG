import os
from typing import Dict, Any


class Config:
    """Configuration settings for the RAG system."""

    # OpenAI Settings
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini-2024-07-18")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # Vector Store Settings
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "embeddings")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    # Document Processing Settings
    CHUNK_SIZE = int(
        os.getenv("CHUNK_SIZE", "1500")
    )
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200")) 

    # Retrieval Settings
    DEFAULT_K = int(os.getenv("DEFAULT_K", "8"))
    MIN_SIMILARITY_THRESHOLD = float(
        os.getenv("MIN_SIMILARITY_THRESHOLD", "0.3")
    )  # Minimum similarity score
    MAX_CHUNKS_TO_RETURN = int(
        os.getenv("MAX_CHUNKS_TO_RETURN", "4")
    )  # Final chunks to return
    USE_QUERY_EXPANSION = os.getenv("USE_QUERY_EXPANSION", "true").lower() == "true"
    USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() == "true"

    # Supported File Types
    SUPPORTED_FILE_TYPES = [".pdf", ".docx", ".txt", ".md"]

    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration and return status."""
        validation_results = {
            "openai_api_key": bool(cls.OPENAI_API_KEY),
            "vector_store_path": os.path.exists(cls.VECTOR_STORE_PATH)
            or not os.path.exists(cls.VECTOR_STORE_PATH),
            "chunk_size": cls.CHUNK_SIZE > 0,
            "chunk_overlap": cls.CHUNK_OVERLAP >= 0,
            "chunk_overlap_valid": cls.CHUNK_OVERLAP < cls.CHUNK_SIZE,
        }

        return validation_results

    @classmethod
    def get_validation_errors(cls) -> list:
        """Get list of validation errors."""
        validation = cls.validate()
        errors = []

        if not validation["openai_api_key"]:
            errors.append("OPENAI_API_KEY not set")

        if not validation["chunk_size"]:
            errors.append("CHUNK_SIZE must be greater than 0")

        if not validation["chunk_overlap"]:
            errors.append("CHUNK_OVERLAP must be non-negative")

        if not validation["chunk_overlap_valid"]:
            errors.append("CHUNK_OVERLAP must be less than CHUNK_SIZE")

        return errors
