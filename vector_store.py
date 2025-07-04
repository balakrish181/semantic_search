"""
Vector store module using FAISS for storing and retrieving paper embeddings
"""
import os
import logging
from typing import Dict, List, Optional, Any, Union
import json
import numpy as np
import faiss
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """Class for storing and retrieving paper embeddings using FAISS"""
    
    def __init__(self, collection_name: str = "academic_papers"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the collection
        """
        # Get persist directory from environment or use default
        self.persist_directory = Path(os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db"))
        self.persist_directory.mkdir(exist_ok=True, parents=True)
        
        # File paths for storing index and metadata
        self.index_file = self.persist_directory / f"{collection_name}_index.faiss"
        self.metadata_file = self.persist_directory / f"{collection_name}_metadata.pkl"
        
        # Initialize FAISS index and metadata storage
        logger.info(f"Initializing FAISS vector store with name: {collection_name}")
        
        # Load existing index and metadata if they exist
        self.papers_metadata = []
        self.paper_ids = []
        
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.papers_metadata = data.get('metadata', [])
                    self.paper_ids = data.get('ids', [])
                logger.info(f"Loaded existing index with {len(self.paper_ids)} papers")
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
    
    def _create_new_index(self):
        """Create a new FAISS index"""
        # We don't know the embedding dimension yet, so we'll create it when adding the first paper
        self.index = None
        self.papers_metadata = []
        self.paper_ids = []
        logger.info("Created new FAISS index")
    
    def add_paper(self, paper: Dict[str, Any]) -> bool:
        """
        Add a paper to the vector store
        
        Args:
            paper: Paper metadata with embedding
            
        Returns:
            True if successful, False otherwise
        """
        if "embedding" not in paper:
            logger.warning(f"No embedding found for paper: {paper.get('title', 'Unknown')}")
            return False
        
        # Generate a unique ID for the paper
        paper_id = paper.get("id", None)
        if not paper_id:
            # Use title as fallback ID
            paper_id = "paper_" + "".join(c if c.isalnum() else "_" for c in paper.get("title", "unknown"))[:50]
        
        # Prepare metadata (exclude large fields)
        metadata = {k: v for k, v in paper.items() if k not in ["embedding"]}
        
        # Convert metadata to JSON-serializable format
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                metadata[key] = json.dumps(value)
        
        try:
            # Convert embedding to numpy array
            embedding = np.array([paper["embedding"]], dtype=np.float32)
            
            # Initialize index if this is the first paper
            if self.index is None:
                dimension = len(paper["embedding"])
                self.index = faiss.IndexFlatL2(dimension)
                logger.info(f"Initialized FAISS index with dimension {dimension}")
            
            # Add to index
            self.index.add(embedding)
            
            # Store metadata
            self.papers_metadata.append(metadata)
            self.paper_ids.append(paper_id)
            
            # Save index and metadata
            self._save_index()
            
            logger.info(f"Added paper to vector store: {paper.get('title')}")
            return True
        except Exception as e:
            logger.error(f"Error adding paper to vector store: {str(e)}")
            return False
            
    def _save_index(self):
        """Save the index and metadata to disk"""
        try:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_file))
            
            # Save metadata
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.papers_metadata,
                    'ids': self.paper_ids
                }, f)
                
            logger.info(f"Saved index with {len(self.paper_ids)} papers")
        except Exception as e:
            logger.error(f"Error saving index: {str(e)}")
            
    
    def add_papers(self, papers: List[Dict[str, Any]]) -> int:
        """
        Add multiple papers to the vector store
        
        Args:
            papers: List of paper metadata with embeddings
            
        Returns:
            Number of papers successfully added
        """
        # Filter papers with embeddings
        papers_with_embeddings = [p for p in papers if "embedding" in p]
        if len(papers_with_embeddings) < len(papers):
            logger.warning(f"Skipping {len(papers) - len(papers_with_embeddings)} papers without embeddings")
        
        if not papers_with_embeddings:
            return 0
        
        success_count = 0
        for paper in papers_with_embeddings:
            if self.add_paper(paper):
                success_count += 1
        
        return success_count
    
    def search(self, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar papers using a query embedding
        
        Args:
            query_embedding: Embedding vector for the query
            n_results: Number of results to return
            
        Returns:
            List of paper metadata for similar papers
        """
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("No papers in the index to search")
                return []
            
            # Limit n_results to the number of papers in the index
            n_results = min(n_results, self.index.ntotal)
            
            # Convert query to numpy array
            query_np = np.array([query_embedding], dtype=np.float32)
            
            # Search for similar papers
            distances, indices = self.index.search(query_np, n_results)
            
            # Format results
            formatted_results = []
            
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.papers_metadata):
                    continue
                    
                # Get metadata for this paper
                metadata = self.papers_metadata[idx]
                
                result = {
                    "id": self.paper_ids[idx],
                    "distance": float(distances[0][i])
                }
                
                # Parse metadata
                for key, value in metadata.items():
                    # Try to parse JSON strings back to objects
                    if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                        try:
                            metadata[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                
                result.update(metadata)
                formatted_results.append(result)
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            return []
    
    def search_by_text(self, query_text: str, embedder, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar papers using a text query
        
        Args:
            query_text: Text query
            embedder: Embedder instance to generate query embedding
            n_results: Number of results to return
            
        Returns:
            List of paper metadata for similar papers
        """
        # Generate embedding for query
        query_embedding = embedder.generate_embedding(query_text)
        
        # Search using the embedding
        return self.search(query_embedding, n_results)
    
    def get_all_papers(self) -> List[Dict[str, Any]]:
        """
        Get all papers from the vector store
        
        Returns:
            List of all paper metadata
        """
        try:
            if not self.papers_metadata:
                return []
                
            # Format results
            formatted_results = []
            for i, metadata in enumerate(self.papers_metadata):
                result = {
                    "id": self.paper_ids[i]
                }
                
                # Parse metadata
                for key, value in metadata.items():
                    # Try to parse JSON strings back to objects
                    if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                        try:
                            metadata[key] = json.loads(value)
                        except json.JSONDecodeError:
                            pass
                
                result.update(metadata)
                formatted_results.append(result)
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error getting papers from vector store: {str(e)}")
            return []


if __name__ == "__main__":
    # Example usage
    from embedding import OpenAIEmbedder
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Initialize embedder
    embedder = SPECTER2Embedder()
    
    # Example paper
    paper = {
        "title": "Quantum Computing: An Overview",
        "snippet": "This paper provides an overview of quantum computing principles and applications.",
        "markdown_content": """
        # Quantum Computing: An Overview
        
        ## Abstract
        This paper provides an overview of quantum computing principles and applications.
        We discuss qubits, quantum gates, and quantum algorithms.
        
        ## Introduction
        Quantum computing is an emerging field that leverages quantum mechanics to perform computations.
        Unlike classical computers that use bits, quantum computers use quantum bits or qubits.
        """
    }
    
    # Generate embedding for the paper
    paper_with_embedding = embedder.embed_paper(paper)
    
    # Add paper to vector store
    vector_store.add_paper(paper_with_embedding)
    
    # Search for similar papers
    results = vector_store.search_by_text("quantum computing algorithms", embedder)
    
    print(f"Found {len(results)} similar papers")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title')} (distance: {result.get('distance'):.4f})")
