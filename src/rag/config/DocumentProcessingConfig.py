from dataclasses import dataclass

@dataclass
class DocumentProcessingConfig:
    chunk_size: int
    chunk_overlap: int
    top_k_results: int
    similarity_threshold: float