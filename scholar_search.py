"""
Google Scholar search module using SerpApi
"""
import os
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from serpapi import GoogleSearch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScholarSearch:
    """Class to search Google Scholar and extract paper metadata and PDF links"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with SerpApi key"""
        self.api_key = api_key or os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            raise ValueError("SerpApi API key is required. Set SERPAPI_API_KEY in .env file.")
    
    def search(self, query: str, num_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Google Scholar for academic papers
        
        Args:
            query: Search query string
            num_results: Maximum number of results to return
            
        Returns:
            List of paper metadata including title, authors, abstract, and PDF link
        """
        logger.info(f"Searching Google Scholar for: {query}")
        
        params = {
            "engine": "google_scholar",
            "q": query,
            "api_key": self.api_key,
            "num": num_results,
            "as_vis": 1,  # Include citations and patents
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                logger.error(f"SerpApi error: {results['error']}")
                return []
            
            if "organic_results" not in results:
                logger.warning("No results found")
                return []
            
            papers = []
            for result in results["organic_results"]:
                paper = {
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "snippet": result.get("snippet", ""),
                    "publication_info": result.get("publication_info", {}),
                    "resources": []
                }
                
                # Extract PDF links if available
                if "resources" in result:
                    for resource in result["resources"]:
                        if resource.get("file_format") == "PDF":
                            paper["resources"].append({
                                "title": resource.get("title", ""),
                                "link": resource.get("link", "")
                            })
                
                papers.append(paper)
            
            logger.info(f"Found {len(papers)} papers")
            return papers
        
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {str(e)}")
            return []
    
    def extract_pdf_links(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract PDF links from paper metadata
        
        Args:
            papers: List of paper metadata from search()
            
        Returns:
            List of papers with PDF links
        """
        papers_with_pdfs = []
        
        for paper in papers:
            if paper.get("resources"):
                # Paper already has PDF resources
                papers_with_pdfs.append(paper)
            else:
                # Try to find PDF link in the paper's link
                # This is a simplified approach - in a real implementation,
                # you might want to scrape the paper's page to find the PDF link
                papers_with_pdfs.append(paper)
        
        return papers_with_pdfs


if __name__ == "__main__":
    # Example usage
    search = ScholarSearch()
    papers = search.search("quantum computing algorithms", num_results=5)
    papers_with_pdfs = search.extract_pdf_links(papers)
    
    for i, paper in enumerate(papers_with_pdfs, 1):
        print(f"{i}. {paper['title']}")
        if paper.get("resources"):
            for resource in paper["resources"]:
                print(f"   PDF: {resource['link']}")
        print()
