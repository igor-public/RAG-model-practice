import logging
import os
from rag.BedrockManager import BedrockManager
from rag.PineconeManager import PineconeManager
from dotenv import find_dotenv, load_dotenv
from rag.RAGConfig import RAGConfig


""" with open("/config/default.yaml", "r") as f:
    cfg_dict = yaml.safe_load(f) """

logging.basicConfig(
    level=logging.DEBUG,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


# Environment variables. Requires PINECONE_KEY (free tier)

load_dotenv(find_dotenv("local.env"))
PINECONE_KEY = os.getenv("PINECONE_KEY")

logger.debug(PINECONE_KEY)


model_prompt = "Explain the theory of relativity in simple terms."
SEARCH_QUESTION = "What is the key advantages of LoRA?"
query_prompt = "what is the adapter Layers and Inference Latency"


if __name__ == "__main__":
    tool_config = {
        "tools": [
            {
                "toolSpec": {
                    "name": "search_document",
                    "description": "Search through the uploaded document for relevant passages",
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
        "toolChoice": {"auto": {}},
    }
    
    tool_config_no_tool = {
        "tools": [
            {
                "toolSpec": {
                    "name": "search_document",
                    "description": "Search through the uploaded document for relevant passages",
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
        "toolChoice": {"none": {}},
    }

    prompt = f"Q: {SEARCH_QUESTION}\n\n"
    
    system_prompt = "You are a helpful assistant that answers questions concisely."

    system = [{"text": system_prompt}]

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    # pc = PineconeManager(RAGConfig)
    mgr = BedrockManager(RAGConfig)

    mgr.get_model_response_stream(tool_config, messages)
