"""
Main entry point for the academic paper RAG system
"""
import os
import logging
import argparse
from typing import Dict, List, Optional, Any
from pathlib import Path
from dotenv import load_dotenv

from scholar_search import ScholarSearch
from text_extraction import PdfProcessor
from embedding import OpenAIEmbedder
from vector_store import VectorStore
from rag_engine import RAGEngine

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AcademicPaperRAG:
    """Main class for the academic paper RAG system"""
    
    def __init__(self):
        """Initialize all components of the RAG system"""
        self.scholar_search = ScholarSearch()
        self.pdf_processor = PdfProcessor()
        self.embedder = OpenAIEmbedder()
        self.vector_store = VectorStore()
        self.rag_engine = None  # Initialize on demand to save resources
    
    def ingest_papers(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers, download PDFs, extract text, generate embeddings, and store in vector DB
        
        Args:
            query: Search query for Google Scholar
            num_results: Number of papers to ingest
            
        Returns:
            List of processed papers
        """
        logger.info(f"Starting paper ingestion for query: {query}")
        
        # Step 1: Search for papers
        papers = self.scholar_search.search(query, num_results=num_results)
        papers_with_pdfs = self.scholar_search.extract_pdf_links(papers)
        logger.info(f"Found {len(papers_with_pdfs)} papers with PDF links")
        
        # Step 2: Process papers (download PDFs and convert to Markdown)
        processed_papers = []
        for paper in papers_with_pdfs:
            processed_paper = self.pdf_processor.process_paper(paper)
            if processed_paper:
                processed_papers.append(processed_paper)
        
        logger.info(f"Processed {len(processed_papers)} papers")
        
        # Step 3: Generate embeddings
        papers_with_embeddings = self.embedder.embed_papers(processed_papers)
        logger.info(f"Generated embeddings for {len(papers_with_embeddings)} papers")
        
        # Step 4: Store in vector DB
        num_added = self.vector_store.add_papers(papers_with_embeddings)
        logger.info(f"Added {num_added} papers to vector store")
        
        return papers_with_embeddings
    
    def answer_query(self, query: str, similarity_top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a query using RAG
        
        Args:
            query: Query text
            similarity_top_k: Number of similar documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        # Initialize RAG engine if not already initialized
        if self.rag_engine is None:
            logger.info("Initializing RAG engine")
            self.rag_engine = RAGEngine()
        
        # Answer query
        return self.rag_engine.answer_query(query, similarity_top_k=similarity_top_k)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Academic Paper RAG System")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest papers")
    ingest_parser.add_argument("query", help="Search query for Google Scholar")
    ingest_parser.add_argument("--num-results", type=int, default=5, help="Number of papers to ingest")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Answer a query")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=3, help="Number of similar documents to retrieve")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize RAG system
    rag_system = AcademicPaperRAG()
    
    # Run command
    if args.command == "ingest":
        papers = rag_system.ingest_papers(args.query, num_results=args.num_results)
        logger.info(f"Ingested {len(papers)} papers")
        
        # Print paper titles
        for i, paper in enumerate(papers, 1):
            logger.info(f"{i}. {paper.get('title', 'Unknown')}")
    
    elif args.command == "query":
        result = rag_system.answer_query(args.query, similarity_top_k=args.top_k)
        
        # Print answer
        print("\n" + "="*80)
        print(f"Query: {result['query']}")
        print("="*80)
        print(f"Answer: {result['answer']}")
        print("="*80)
        print(f"Sources: {len(result['source_documents'])}")
        
        # Print source documents
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\nSource {i} (score: {doc['score']:.4f}):")
            if 'title' in doc['metadata']:
                print(f"Title: {doc['metadata']['title']}")
            print("-"*40)
            print(doc['text'][:300] + "..." if len(doc['text']) > 300 else doc['text'])
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
