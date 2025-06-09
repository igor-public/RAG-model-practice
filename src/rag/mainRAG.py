import logging
import os
import yaml
from rag.BedrockManager import BedrockManager
from dotenv import find_dotenv, load_dotenv
from rag.config.RAGConfig import RAGConfig


logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# Environment variables. Requires PINECONE_KEY (free tier)

load_dotenv(find_dotenv("local.env"))
PINECONE_KEY = os.getenv("PINECONE_KEY")

logger.debug(PINECONE_KEY)


# model_prompt = "Explain the theory of relativity in simple terms."
SEARCH_QUESTION = "What is the weather in Munich today?"
# query_prompt = "what is the adapter Layers and Inference Latency"


if __name__ == "__main__":
    
    
    with open("rag/resources/tools.yaml") as yaml_file:
        tool_config = yaml.safe_load(yaml_file)
    
    prompt = f"Q: {SEARCH_QUESTION}\n\n"

    messages = [{"role": "user", "content": [{"text": prompt}]}]

    mgr = BedrockManager(RAGConfig.load_from_yaml("rag/resources/rag_default.yaml"), prompt_path="rag/resources/system-prompt.yaml")

    mgr.get_model_response_stream(tool_config, messages)
