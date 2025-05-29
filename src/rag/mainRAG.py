import logging
import os
from rag.BedrockManager import BedrockManager
from rag.PineconeManager import PineconeManager 
from dotenv import find_dotenv, load_dotenv
from rag.RAGConfig import RAGConfig
 

""" with open("/config/default.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f) """

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

    prompt = f"Q: {SEARCH_QUESTION}\n\n"

    messages = [
        {
            "role": "system",
            "content": (
                "You can call the function `search_document` when information "
                "is likely stored in the vector database."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    #pc = PineconeManager(RAGConfig)
    mgr = BedrockManager(RAGConfig)

    mgr.get_model_response_stream(TOOLS, messages)

