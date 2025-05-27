from dataclasses import dataclass


@dataclass
class RAGConfig:

    # AWS Bedrock 

    model_id: str = "mistral.mistral-large-2407-v1:0"
    model_temperature: float = 0.4
    max_tokens: int = 2048
    model_aws_region: str = "us-west-2"
    model_runtime: str = "bedrock-runtime"

    # Pinecone related
    index_name: str = "ia-index"
    embedding_dim: int = 384
    metric: str = "cosine"
    cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Document Processing 
    chunk_size: int = 1024
    chunk_overlap: int = 128
    top_k_results: int = 3
    similarity_threshold: float = 0.7

    # Embedding Model 
    embedding_model_name: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )


class RAGSystemException(Exception):
    """Custom exception for RAG system errors"""

    pass
