import argparse
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_pipeline import RAGPipeline
from src.config import config

def main():
    parser = argparse.ArgumentParser(description="Architecture Research Paper RAG System")
    parser.add_argument("--init", action="store_true", help="Initialize database with research papers")
    parser.add_argument("--query", type=str, help="Query to search in research papers")
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    if args.init:
        # Initialize database
        rag.initialize_database(config.JSONL_FILES)
    
    elif args.query:
        # Process single query
        result = rag.query(args.query)
        
        print("\n" + "="*80)
        print("ANSWER:")
        print("="*80)
        print(result["answer"])
        print("\n" + "="*80)
        print("SOURCES:")
        print("="*80)
        for source in result["sources"]:
            print(f"Source {source['source_id']}:")
            print(f"  Title: {source['title']}")
            print(f"  Authors: {', '.join(source['authors']) if source['authors'] else 'Unknown'}")
            print(f"  Year: {source['year']}")
            print(f"  Confidence: {source['confidence']}")
            print()
    
    elif args.interactive:
        # Interactive mode
        print("Architecture Research Paper RAG System")
        print("Type 'quit' to exit, 'stats' to see database statistics")
        print("-" * 60)
        
        while True:
            query = input("\nEnter your research question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'stats':
                stats = rag.db.get_collection_stats()
                print(f"Database Statistics: {stats}")
                continue
            elif not query:
                continue
            
            result = rag.query(query)
            
            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            print(result["answer"])
            print("\n" + "="*80)
            print("TOP SOURCES:")
            print("="*80)
            for source in result["sources"][:3]:  # Show top 3 sources
                print(f"• {source['title']} ({source['year']})")
            print()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()