import os
from pathlib import Path

class Config:
    # Embedding model
    EMBEDDING_MODEL = "multi-qa-MiniLM-L6-cos-v1"
    
    # ChromaDB settings
    COLLECTION_NAME = "architecture_research_papers"
    PERSIST_DIRECTORY = "./chroma_db"
    
    # JSONL files - update these paths to match your actual file locations
    JSONL_FILES = [
        "D:\\CODING\\Machine Learning\\Architect-RAG-LLM-Assistant\\chunks\\building_codes_chunks.jsonl",
        "D:\\CODING\\Machine Learning\\Architect-RAG-LLM-Assistant\\chunks\\case_studies_chunks.jsonl",
        "D:\\CODING\\Machine Learning\\Architect-RAG-LLM-Assistant\\chunks\\material_guide_chunks.jsonl",
        "D:\\CODING\\Machine Learning\\Architect-RAG-LLM-Assistant\\chunks\\misc_chunks.jsonl"
    ]
    
    # Ollama settings
    OLLAMA_MODEL = "llama2"  # or "mistral", "codellama", etc.
    OLLAMA_BASE_URL = "http://localhost:11434"
    
    # RAG settings
    TOP_K_RESULTS = 5
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

config = Config()