from pydantic import BaseSettings

class Settings(BaseSettings):
    model_id: str
    model_temperature: float
    max_tokens: int
    aws_region: str
    model_runtime: str

    index_name: str
    embedding_dim: int
    metric: str
    cloud: str
    region: str

    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    similarity_threshold: float

    embedding_model_name: str

    class Config:
        env_file = ".env"   # for PINECONE_KEY, etc.

