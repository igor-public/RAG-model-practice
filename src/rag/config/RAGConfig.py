from dataclasses import dataclass
from pathlib import Path
import yaml

from rag.config.AWSBedrockConfig import AWSBedrockConfig
from rag.config.PineconeConfig import PineconeConfig
from rag.config.DocumentProcessingConfig import DocumentProcessingConfig
from rag.config.EmbeddingModelConfig import EmbeddingModelConfig


@dataclass
class RAGConfig:
    aws_bedrock: AWSBedrockConfig
    pinecone: PineconeConfig
    document_processing: DocumentProcessingConfig
    embedding_model: EmbeddingModelConfig



    @staticmethod
    def load_from_yaml(path: str) -> "RAGConfig":
        
        p = Path (path)
        
        try:
            data = yaml.safe_load(p.read_text())
        except FileNotFoundError:
            raise RAGSystemException(f"Could not find config file at {p}")
        
        return RAGConfig(
            aws_bedrock=AWSBedrockConfig(**data["aws_bedrock"]),
            pinecone=PineconeConfig(**data["pinecone"]),
            document_processing=DocumentProcessingConfig(**data["document_processing"]),
            embedding_model=EmbeddingModelConfig(**data["embedding_model"]),
        )
        
class RAGSystemException(Exception):
    """Custom exception for RAG system errors"""

    pass