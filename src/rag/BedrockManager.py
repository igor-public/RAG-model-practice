import boto3
import logging
import yaml
from pathlib import Path
from functools import wraps
from typing import Dict, Any
from rag.PineconeManager import PineconeManager
import json
from rag.config.RAGConfig import RAGConfig, RAGSystemException
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

#Loading resource function

def load_system_prompt(path: Path) -> str:

    p = Path (path)

    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, str):
        raise ValueError(f"Expected system prompt YAML to be a single string, but got {type(raw)}")
    return raw.strip()

# Decorator to handle exceptions in BedrockManager methods

def safe(fn):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except RAGSystemException:
                raise
            except Exception as exc:
                logger.error("%s failed: %s", fn.__name__, exc, exc_info=True)
                raise RAGSystemException(str(exc)) from exc

        return _wrapper

# BedrockManager class
class BedrockManager:
    
    def __init__(self, config: RAGConfig, prompt_path: Path = Path("resources/system-prompt.yaml")):
        self.config = config
        
        #Bedrock client 
        
        self.session = boto3.Session()
        try:
            self.bedrock = self.session.client(
                self.config.aws_bedrock.model_runtime,
                region_name=self.config.aws_bedrock.model_aws_region,
            )
        except Exception as e:
            raise RAGSystemException(f"Could not initialize Bedrock client: {e}") from e
        
        
        #Creating a splitter for the Pinecone index.
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.document_processing.chunk_size, 
            chunk_overlap=config.document_processing.chunk_overlap,
        )
        
        #System prompt loading
        
        try:
            self.system_prompt = load_system_prompt(prompt_path)
        except Exception as e:
            raise RAGSystemException(f"Failed to load system prompt from {prompt_path}: {e}") from e
        
        self.system = [
            {"text": self.system_prompt}
        ]
        
        # Build inferenceConfig from RAGConfig constants
        self.inference_config: Dict[str, Any] = {
            "maxTokens": self.config.aws_bedrock.max_tokens,
            "temperature": self.config.aws_bedrock.model_temperature,
            "topP": 1.0,
        }

    #Streaming response from the model with tools. Wrapping function which handles tool calls with a no stram call first and and processes the response using the streaming response.
    #This is a workaround for the current limitation of the Bedrock API, which does not support streaming responses with tools.
    
    """
        1. Calls Bedrock .converse(...) with tools enabled (no streaming) to see if there's a toolUse.
        2. If toolUse == "search_document", run Pinecone search, append the results to messages.
        3. Call Bedrock .converse_stream(...) (streaming mode) to get the final answer.
        Returns the concatenated string from the stream.
        Raises:
            ValueError if messages or tool_config is missing/invalid.
            RAGSystemException if any call fails internally.
    """

    @safe
    def get_model_response_stream(self, toolConfig, messages) -> str:
        
        if not messages:
            raise ValueError("Messages missing")

        if not toolConfig:
            raise ValueError("tools configuration is missing")


        # Initiating the first call

        logger.debug(
            f"Invoking model {self.config.aws_bedrock.model_id} with TOOLS and no streaming ..."
        )

        try:
            response = self.bedrock.converse(
            modelId=self.config.aws_bedrock.model_id,
            system=self.system,
            messages=messages,
            inferenceConfig=self.inference_config,
            toolConfig=toolConfig,
            )
        
        except Exception as e:
            raise RAGSystemException(f"Failed to invoke model: {e}") from e

        logger.debug(f"\n First response: {json.dumps(response, indent=2)}")

        tool_calls = []

        # Check if the response contains tool calls

        for block in response.get("output", {}).get("message", {}).get("content", []):
            tool_use = block.get("toolUse")

            if tool_use:
                tool_calls.append(
                    {
                        "tool_use_id": tool_use["toolUseId"],
                        "name": tool_use["name"],
                        "input": tool_use["input"],
                    }
                )

        # If the assistant asked to use "search_document", do the Pinecone query, update the`messages`
        
        logger.debug(f"Tool calls: {tool_calls}")

    
        if any(tc.get("name") == "search_document" for tc in tool_calls):
            
            # Create a PineconeManager instance only when needed.
            pc = PineconeManager(self.config)
               
            for tc in tool_calls:
                if tc.get("name") == "search_document":
                    query_text = tc.get("input", {}).get("query", {})
                    
                    #Search in the docs 
                    try:
                        search_result = pc.search(query_text)
                    except Exception as e:
                        logger.error(f"Pinecone search failed for query '{query_text}': {e}", exc_info=True)
                        search_result = f"[Error during search: {e}]"
                    
                    logger.debug(
                        f"Search result for query '{query_text}':\n{search_result}"
                    )
                    
                    # Append the search result to the messages
                    messages.append(
                        {"role": "assistant", "content": [{"text": search_result}]}
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "text": "please summarise the search results in three bullet points"
                                }
                            ],
                        }
                    )
                    logger.info("\n\n Processing search results...\n")

       

        # ---- second pass, with streaming, no tools --------------------------------------------------

        logger.info(
            f"\n\n {self.config.aws_bedrock.model_id} called again with streaming and no tools ..."
        )
        
        logger.debug("Final messages payload: %r", messages)

        final_response = self.bedrock.converse_stream(
            modelId=self.config.aws_bedrock.model_id,
            system=self.system,
            messages=messages,
            inferenceConfig=self.inference_config,
        )

        full_response = ""
        
        for chunk in final_response["stream"]:
            text = chunk.get("contentBlockDelta", {}).get("delta", {}).get("text", "")
            if text:
                print(text, end="", flush=True)
                full_response += text
            if chunk.get("messageStop", {}).get("stopReason") == "stop":
                break
        
        return full_response
