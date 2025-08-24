import ollama
from typing import List, Dict, Any
from src.config import config
from src.database import ResearchPaperDatabase

class RAGPipeline:
    def __init__(self):
        self.db = ResearchPaperDatabase()
        self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using Ollama with retrieved context"""
        
        # Prepare the context
        context_text = "\n\n".join([
            f"Reference {i+1}:\n{doc}" for i, doc in enumerate(context)
        ])
        
        # System prompt for architecture research
        system_prompt = """
        You are an expert in architecture research. Use the provided research paper excerpts to answer the user's question accurately, comprehensively, and in a structured way.  

        Guidelines:
        1. Base your answer strictly on the provided context. If the context doesn’t have relevant info, state it clearly.  
        2. Cite references naturally within the explanation when possible.  
        3. Always provide your answer in the following format:

        ---
        ### QUES
        (repeat the user’s question)

        ### ANSWER (based on provided context)
        (give a clear, structured response with headings/bullets for readability)

        ### EVALUATION
        - ✅ Correct points
        - ❌ Limitations or misalignments
        - 🔑 Missing aspects (if any)

        ### FINAL JUDGMENT
        (Summarize correctness in percentage terms and give an overall conclusion)
        ---

        4. Maintain an academic yet approachable tone — precise but easy to read.  
        5. If the user asks about 'universal' or 'global' regulations but context is jurisdiction-specific, clarify this distinction and, if appropriate, mention widely recognized international standards or principles.  
        """
        
        # User prompt with context
        user_prompt = f"""Based on the following research excerpts, answer this question: {query}

        Research Context:
        {context_text}

        Please provide a comprehensive answer citing relevant sections from the research papers."""

        try:
            response = self.ollama_client.chat(
                model=config.OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            return response['message']['content']
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, user_query: str, n_results: int = config.TOP_K_RESULTS) -> Dict[str, Any]:
        """Complete RAG pipeline: retrieve and generate"""
        
        # Step 1: Query the database
        print("Searching for relevant research papers...")
        results = self.db.query_documents(user_query, n_results)
        
        if not results or not results['documents']:
            return {
                "answer": "No relevant research papers found for your query.",
                "sources": [],
                "context": []
            }
        
        # Extract retrieved documents
        retrieved_docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if 'distances' in results else [0] * len(metadatas)
        
        # Step 2: Generate response
        print("Generating comprehensive answer...")
        answer = self.generate_response(user_query, retrieved_docs)
        
        # Prepare source information
        sources = []
        for i, (metadata, distance) in enumerate(zip(metadatas, distances)):
            source_info = {
                "source_id": i+1,
                "title": metadata.get('title', 'Unknown Title'),
                "authors": metadata.get('authors', []),
                "year": metadata.get('year', 'Unknown'),
                "confidence": f"{1 - distance:.3f}" if distance is not None else "N/A"
            }
            sources.append(source_info)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": retrieved_docs,
            "query": user_query
        }
    
    def initialize_database(self, jsonl_files: List[str]):
        """Initialize the database with research papers"""
        print("Initializing database with research papers...")
        self.db.add_documents_from_jsonl(jsonl_files)
        self.db.persist()
        print("Database initialization complete!")