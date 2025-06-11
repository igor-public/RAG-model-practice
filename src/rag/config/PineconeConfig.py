from dataclasses import dataclass

@dataclass
class PineconeConfig:
    index_name: str
    embedding_dim: int
    metric: str
    cloud: str
    pinecone_region: str