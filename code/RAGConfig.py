from dataclasses import dataclass

@dataclass
class RAGConfig:
    # AWS Bedrock Config
    model_id: str = "mistral.mistral-large-2407-v1:0"
    model_temperature: float = 0.4
    max_tokens: int = 2048
    model_aws_region: str = "us-west-2"
    model_runtime: str = "bedrock-runtime"
    
    # Pinecone Config
    index_name: str = "ia-index"
    embedding_dim: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    
    # Document Processing Config
    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k_results: int = 3
    similarity_threshold: float = 0.7
    
    # Embedding Model Config
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

class RAGSystemError(Exception):
    """Custom exception for RAG system errors"""
    pass