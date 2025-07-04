"""
RAG engine module using LlamaIndex and GPT-40 mini for answering queries
"""
import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import numpy as np
from llama_index.core import Document, VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.vector_stores import SimpleVectorStore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEngine:
    """Class for answering queries using RAG with LlamaIndex and GPT-40 mini"""
    
    def __init__(self, collection_name: str = "academic_papers", model_name: str = "gpt-4o-mini"):
        """
        Initialize the RAG engine
        
        Args:
            collection_name: Name of the collection
            model_name: Name of the OpenAI model to use
        """
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        
        # Get persist directory from environment or use default
        self.persist_directory = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db"))
        self.persist_directory.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initializing RAG engine with model: {model_name}")
        
        # Initialize a simple vector store
        self.vector_store = SimpleVectorStore()
        
        # Load documents from markdown storage if available
        markdown_dir = Path(os.getenv("MARKDOWN_STORAGE_PATH", "./papers"))
        if markdown_dir.exists() and any(markdown_dir.glob("*.md")):
            try:
                logger.info(f"Loading documents from {markdown_dir}")
                documents = SimpleDirectoryReader(input_dir=str(markdown_dir), required_exts=[".md"]).load_data()
                logger.info(f"Loaded {len(documents)} documents")
            except Exception as e:
                logger.error(f"Error loading documents: {str(e)}")
                documents = []
        else:
            documents = []
        
        # Initialize embedding model
        self.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        # Initialize LLM
        self.llm = OpenAI(model=model_name, temperature=0.1)
        
        # Configure global settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        Settings.num_output = 1024  # Reasonable default
        Settings.context_window = 4096  # For GPT-4o-mini
        
        # Initialize vector store index
        if documents:
            # Create index from documents
            self.index = VectorStoreIndex.from_documents(
                documents=documents,
                llm=self.llm,
                embed_model=self.embed_model
            )
            logger.info("Created index from documents")
        else:
            # Create empty index
            self.index = VectorStoreIndex.from_documents(
                documents=[],
                llm=self.llm,
                embed_model=self.embed_model
            )
            logger.info("Created empty index")
        
        logger.info(f"RAG engine initialized with model: {model_name}")
    
    def answer_query(self, query: str, similarity_top_k: int = 3) -> Dict[str, Any]:
        """
        Answer a query using RAG
        
        Args:
            query: Query text
            similarity_top_k: Number of similar documents to retrieve
            
        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing query: {query}")
        
        try:
            # Create query engine
            query_engine = self.index.as_query_engine(
                similarity_top_k=similarity_top_k,
                llm=self.llm
            )
            
            # Get response
            response = query_engine.query(query)
            
            # Format source documents
            source_documents = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    source_documents.append({
                        "text": node.node.text,
                        "score": node.score,
                        "metadata": node.node.metadata
                    })
            
            return {
                "query": query,
                "answer": str(response),
                "source_documents": source_documents
            }
        
        except Exception as e:
            logger.error(f"Error answering query: {str(e)}")
            return {
                "query": query,
                "answer": f"Error: {str(e)}",
                "source_documents": []
            }
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the RAG engine
        
        Returns:
            System prompt text
        """
        return """
        You are an AI assistant specializing in academic research papers.
        Your task is to provide accurate, helpful answers based on the academic papers in your knowledge base.
        
        When answering:
        1. Focus on information from the provided academic papers
        2. Cite the papers you reference in your answer
        3. If the papers don't contain enough information to answer the question, acknowledge the limitations
        4. Be concise but thorough
        5. Use an academic, professional tone
        
        The user will ask questions about academic topics, and you should provide the most relevant information from the papers.
        """


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize RAG engine
        rag_engine = RAGEngine()
        
        # Example query
        query = "What are the main applications of quantum computing?"
        
        # Get answer
        result = rag_engine.answer_query(query)
        
        print(f"Query: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Sources: {len(result['source_documents'])}")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
