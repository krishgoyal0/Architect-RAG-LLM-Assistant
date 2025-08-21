from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name="multi-qa-MiniLM-L6-cos-v1"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"Embedding model loaded with dimension: {self.dimension}")
    
    def embed_text(self, text):
        """Convert text to embedding vector"""
        if isinstance(text, str):
            return self.model.encode(text).tolist()
        elif isinstance(text, list):
            return self.model.encode(text).tolist()
        else:
            raise ValueError("Input must be string or list of strings")
    
    def embed_query(self, query):
        """Embed a single query"""
        return self.embed_text(query)
    
    def embed_documents(self, documents):
        """Embed multiple documents"""
        return self.embed_text(documents)