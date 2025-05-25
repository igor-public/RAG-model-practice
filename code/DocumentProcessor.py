import RAGConfig
from RAGConfig import RAGSystemError
import os 
import logging
from langchain.schema import Document
from typing import List
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(
    level=logging.INFO,
    #format='%(name)s - %(message)s'
    format='%(name)s - %(levelname)s - %(message)s'

)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
    
    def load_document(self, path: str) -> List[Document]:
        """Load document from path with error handling"""
        try:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
            
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Document not found: {abs_path}")
            
            loader = PyMuPDFLoader(abs_path)
            docs = loader.load()
            
            if not docs:
                raise RAGSystemError(f"No content loaded from {path}")
            
            logger.info(f"Successfully loaded {len(docs)} pages from {path}")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading document {path}: {str(e)}")
            raise RAGSystemError(f"Failed to load document: {str(e)}")
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        """Split documents into chunks with metadata preservation"""
        try:
            chunks = self.text_splitter.split_documents(docs)
            
            # Enhance metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'text': chunk.page_content  # For Pinecone metadata
                })
            
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise RAGSystemError(f"Failed to split documents: {str(e)}")
