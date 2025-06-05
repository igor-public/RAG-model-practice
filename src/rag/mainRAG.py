import logging
import os
from rag.BedrockManager import BedrockManager
from dotenv import find_dotenv, load_dotenv
from rag.RAGConfig import RAGConfig


""" with open("/config/default.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f) """

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


# Environment variables. Requires PINECONE_KEY (free tier)

load_dotenv(find_dotenv("local.env"))
PINECONE_KEY = os.getenv("PINECONE_KEY")

logger.debug(PINECONE_KEY)


# model_prompt = "Explain the theory of relativity in simple terms."
SEARCH_QUESTION = "What is the weather in Munich today?"
# query_prompt = "what is the adapter Layers and Inference Latency"


if __name__ == "__main__":
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "search_document",
                    "description": "Search documents about LoRA and machine‐learning technical details.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        }
                    },
                }
            }
        ],
        # Force the model to only call a tool when explicitly prompted by the system‐prompt logic above
        "toolChoice": {"auto": {}},
    }

    prompt = f"Q: {SEARCH_QUESTION}\n\n"

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    mgr = BedrockManager(RAGConfig)

    mgr.get_model_response_stream(tool_config, messages)
