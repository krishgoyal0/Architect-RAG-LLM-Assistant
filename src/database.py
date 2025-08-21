import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction
from typing import List, Dict, Any
import json
from tqdm import tqdm

from src.embedding_utils import EmbeddingModel
from src.config import config


# Wrapper so we can plug your EmbeddingModel into Chroma
class CustomEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        return [self.embedding_model.embed_text(t) for t in texts]


class ResearchPaperDatabase:
    def __init__(self):
        # ✅ Persistent client (no more deprecated Settings)
        self.client = chromadb.PersistentClient(path=config.PERSIST_DIRECTORY)

        # Initialize embedding model
        self.embedding_model = EmbeddingModel(config.EMBEDDING_MODEL)

        # Wrap into Chroma-compatible embedding function
        self.embedding_fn = CustomEmbeddingFunction(self.embedding_model)

        # Get or create collection
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """Get existing collection or create a new one"""
        try:
            collection = self.client.get_collection(
                name=config.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            print(f"Loaded existing collection: {config.COLLECTION_NAME}")
        except:
            collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {config.COLLECTION_NAME}")

        return collection

    def load_jsonl_file(self, file_path: str):
        """Load and process a single JSONL file"""
        import os
        documents, metadatas, ids = [], [], []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(tqdm(lines, desc=f"Processing {file_path}")):
                    try:
                        data = json.loads(line.strip())

                        content = data.get('text', '')
                        metadata = data.get('metadata', {})

                        # ✅ Ensure metadata is non-empty (Chroma requirement)
                        if not metadata or not isinstance(metadata, dict) or len(metadata) == 0:
                            metadata = {
                                "source": os.path.basename(file_path),
                                "line": i
                            }

                        if content and len(content.strip()) > 0:
                            documents.append(content)
                            metadatas.append(metadata)
                            ids.append(f"{os.path.basename(file_path)}_{i}")

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {i} in {file_path}: {e}")
                        continue
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return [], [], []

        return documents, metadatas, ids


    def add_documents_from_jsonl(self, jsonl_files: List[str]):
        """Add documents from multiple JSONL files to the database"""
        all_documents, all_metadatas, all_ids = [], [], []

        for file_path in jsonl_files:
            print(f"Processing file: {file_path}")
            documents, metadatas, ids = self.load_jsonl_file(file_path)

            if documents:
                all_documents.extend(documents)
                all_metadatas.extend(metadatas)
                all_ids.extend(ids)
            else:
                print(f"No valid documents found in {file_path}")

        if not all_documents:
            print("No documents to add to the database")
            return

        # Add in batches
        batch_size = 100
        for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding documents to database"):
            batch_docs = all_documents[i:i + batch_size]
            batch_metas = all_metadatas[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]

            self.collection.add(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids
            )

        print(f"Added {len(all_documents)} documents to the database")

    def query_documents(self, query: str, n_results: int = config.TOP_K_RESULTS):
        """Query the database for similar documents"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            print(f"Error querying database: {e}")
            return None

    def get_collection_stats(self):
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_documents": count,
                "collection_name": config.COLLECTION_NAME
            }
        except:
            return {"error": "Collection not available"}

    def persist(self):
        """Persist the database to disk"""
        # self.client.persist()
        print("Database persisted to disk")
