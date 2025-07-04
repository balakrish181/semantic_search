import logging
import time
import json
import os
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

def extract_text_from_pdf(input_doc_path):
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.ocr_options.lang = ["es"]
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.AUTO
    )

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    conv_result = doc_converter.convert(input_doc_path)
    return conv_result.document.export_to_markdown()

class PdfProcessor:
    """Class to handle PDF downloading and text extraction"""
    
    def __init__(self):
        """Initialize with storage paths from environment variables"""
        self.pdf_dir = Path(os.getenv("PDF_STORAGE_PATH", "./pdfs"))
        self.markdown_dir = Path(os.getenv("MARKDOWN_STORAGE_PATH", "./papers"))
        
        # Create directories if they don't exist
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
    
    def process_paper(self, paper: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a paper by downloading its PDF and extracting text
        
        Args:
            paper: Paper metadata including PDF URL
            
        Returns:
            Updated paper metadata with paths to PDF and Markdown files,
            or None if processing failed
        """
        # Extract PDF URL from paper metadata
        pdf_url = None
        if paper.get("resources"):
            for resource in paper["resources"]:
                if resource.get("link"):
                    pdf_url = resource["link"]
                    break
        
        if not pdf_url:
            _log.warning(f"No PDF URL found for paper: {paper.get('title', 'Unknown')}")
            return None
        
        # Generate a safe filename from the paper title
        safe_title = "".join(c if c.isalnum() else "_" for c in paper.get("title", "unknown"))[:100]
        pdf_filename = f"{safe_title}.pdf"
        pdf_path = self.pdf_dir / pdf_filename
        
        # Download the PDF
        downloaded_path = self.download_pdf(pdf_url, str(pdf_path))
        if not downloaded_path:
            return None
        
        # Extract text from the PDF
        markdown_text = extract_text_from_pdf(downloaded_path)
        if not markdown_text:
            return None
        
        # Save the Markdown text
        markdown_path = self.markdown_dir / f"{safe_title}.md"
        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_text)
        
        # Update the paper metadata
        paper["pdf_path"] = str(pdf_path)
        paper["markdown_path"] = str(markdown_path)
        paper["markdown_content"] = markdown_text
        
        return paper
    
    def download_pdf(self, url: str, output_path: str) -> Optional[str]:
        """
        Download a PDF from a URL
        
        Args:
            url: URL of the PDF
            output_path: Path to save the PDF
            
        Returns:
            Path to the downloaded PDF, or None if download failed
        """
        try:
            _log.info(f"Downloading PDF from {url}")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            _log.info(f"PDF downloaded to {output_path}")
            return output_path
        
        except Exception as e:
            _log.error(f"Error downloading PDF: {str(e)}")
            return None


def main():
    """Example usage of the PDF processor"""
    from scholar_search import ScholarSearch
    
    # Search for papers
    search = ScholarSearch()
    papers = search.search("quantum computing algorithms", num_results=2)
    papers_with_pdfs = search.extract_pdf_links(papers)
    
    # Process the papers
    processor = PdfProcessor()
    processed_papers = []
    
    for paper in papers_with_pdfs:
        start_time = time.time()
        processed_paper = processor.process_paper(paper)
        if processed_paper:
            processed_papers.append(processed_paper)
            end_time = time.time() - start_time
            _log.info(f"Paper processed in {end_time:.2f} seconds: {paper.get('title')}")
    
    _log.info(f"Processed {len(processed_papers)} papers")

if __name__ == "__main__":
    main()