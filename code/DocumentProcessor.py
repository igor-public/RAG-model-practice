import os 
import logging
from langchain.schema import Document
from typing import List
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from code.RAGConfig import RAGConfig, RAGSystemException


# Initialize logging    

logging.basicConfig(
    
    level=logging.INFO,
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
    
        try:
            abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
            
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Document not found: {abs_path}")
            
            # Using a particular loader (PyMuPDFLoader from langchain_community)
            
            logger.debug(f"Loading document from {abs_path}")
            
            loader = PyMuPDFLoader(abs_path)
            docs = loader.load()
            
            if not docs:
                raise RAGSystemException(f"No content loaded from {path}")
            
            logger.info(f"Successfully loaded {len(docs)} pages from {path}")
            return docs
            
        except Exception as e:
            logger.error(f"Error loading document {path}: {str(e)}")
            raise RAGSystemException(f"Failed to load document: {str(e)}")
    
    def split_documents(self, docs: List[Document]) -> List[Document]:
        
        if not docs:
            raise RAGSystemException("No documents provided for splitting")
        
        try:
            chunks = self.text_splitter.split_documents(docs)
            
            # Improve metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    'chunk_id': i,
                    'chunk_size': len(chunk.page_content),
                    'text': chunk.page_content  # For Pinecone metadata
                })
            
            logger.debug(f"Split documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise RAGSystemException(f"Failed to split documents: {str(e)}")
