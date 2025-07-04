"""
Embedding module for generating OpenAI embeddings for academic papers
"""
import os
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIEmbedder:
    """Class to generate OpenAI embeddings for academic papers"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the OpenAI embedder
        
        Args:
            model_name: OpenAI embedding model name
        """
        logger.info(f"Initializing OpenAI embedder with model: {model_name}")
        
        # Get API key
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY in .env file.")
        
        # Set model name
        self.model_name = model_name
        
        # Set API endpoint
        self.api_url = "https://api.openai.com/v1/embeddings"
    
    def _call_openai_api(self, text_batch: List[str]) -> List[List[float]]:
        """
        Call OpenAI API to generate embeddings
        
        Args:
            text_batch: Batch of texts to embed
            
        Returns:
            List of embeddings
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "input": text_batch,
            "model": self.model_name
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # If rate limited, wait and retry once
            if hasattr(e, "response") and e.response.status_code == 429:
                logger.info("Rate limited, waiting 5 seconds and retrying...")
                time.sleep(5)
                try:
                    response = requests.post(self.api_url, headers=headers, json=data)
                    response.raise_for_status()
                    
                    result = response.json()
                    embeddings = [item["embedding"] for item in result["data"]]
                    
                    return embeddings
                except Exception as retry_e:
                    logger.error(f"Error on retry: {str(retry_e)}")
            
            # Return empty embeddings as fallback
            return [[0.0] * 1536] * len(text_batch)
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        # Call OpenAI API with a single text
        embeddings = self._call_openai_api([text])
        
        # Return the first (and only) embedding
        return embeddings[0] if embeddings else [0.0] * 1536
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # Process in batches of 20 to avoid API limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self._call_openai_api(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Sleep briefly between large batches to avoid rate limits
            if len(texts) > batch_size and i + batch_size < len(texts):
                time.sleep(0.5)
        
        return all_embeddings
    
    def embed_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate embeddings for a paper
        
        Args:
            paper: Paper metadata including markdown content
            
        Returns:
            Paper metadata with embeddings
        """
        if "markdown_content" not in paper:
            logger.warning(f"No markdown content found for paper: {paper.get('title', 'Unknown')}")
            return paper
        
        # Generate embedding for the paper
        paper["embedding"] = self.generate_embedding(paper["markdown_content"])
        
        return paper
    
    def embed_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple papers
        
        Args:
            papers: List of paper metadata including markdown content
            
        Returns:
            List of paper metadata with embeddings
        """
        papers_with_content = [p for p in papers if "markdown_content" in p]
        if len(papers_with_content) < len(papers):
            logger.warning(f"Skipping {len(papers) - len(papers_with_content)} papers without markdown content")
        
        if not papers_with_content:
            return papers
        
        # Extract markdown content
        texts = [p["markdown_content"] for p in papers_with_content]
        
        # Generate embeddings
        embeddings = self.generate_embeddings_batch(texts)
        
        # Add embeddings to papers
        for i, paper in enumerate(papers_with_content):
            paper["embedding"] = embeddings[i]
        
        return papers


if __name__ == "__main__":
    # Example usage
    embedder = OpenAIEmbedder()
    
    # Example text
    text = """
    Quantum computing is an emerging field that leverages quantum mechanics to perform computations.
    Unlike classical computers that use bits, quantum computers use quantum bits or qubits.
    This allows quantum computers to potentially solve certain problems much faster than classical computers.
    """
    
    # Generate embedding
    embedding = embedder.generate_embedding(text)
    
    print(f"Generated embedding with {len(embedding)} dimensions")
    print(f"First 5 values: {embedding[:5]}")
