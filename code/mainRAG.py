import logging
import os
import BedrockManager
import PineconeManager
from dotenv import find_dotenv, load_dotenv
from RAGConfig import RAGConfig


logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Environment variables. Requires PINECONE_KEY (free tier)

load_dotenv(find_dotenv("local.env"))
PINECONE_KEY = os.getenv("PINECONE_KEY")

logger.debug(PINECONE_KEY)


# --- Config ---
MODEL_ID = RAGConfig.model_id
MODEL_TEMPERATURE = RAGConfig.model_temperature
MAX_TOKENS = RAGConfig.max_tokens
MODEL_REGION = RAGConfig.model_aws_region
MODEL_RUNTIME = RAGConfig.model_runtime

INDEX_NAME = RAGConfig.index_name
EMBEDDING_DIM = RAGConfig.embedding_dim
METRIC = RAGConfig.metric
CLOUD = RAGConfig.cloud
PINECONE_REGION = RAGConfig.pinecone_region

model_prompt = "Explain the theory of relativity in simple terms."
SEARCH_QUESTION = "What is the key advantages of LoRA?"
query_prompt = "what is the adapter Layers and Inference Latency"


if __name__ == "__main__":
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_document",
                "description": "Search through the uploaded document to find relevant information. Use this when you need specific information that might be in the document collection.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find relevant documents",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    pc = PineconeManager(api_key=PINECONE_KEY)
    prompt = f"Q: {SEARCH_QUESTION}\n\nAnswer:"

    BedrockManager.getModelResponse_stream(
        prompt, TOOLS, MODEL_ID, MAX_TOKENS, MODEL_TEMPERATURE, pc.client
    )
