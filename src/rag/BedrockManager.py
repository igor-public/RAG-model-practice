import boto3
import logging
from functools import wraps
from rag.PineconeManager import PineconeManager
import json
from rag.RAGConfig import RAGConfig, RAGSystemException
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

# system = [{"text": [{"text": "system"}]}]

system_prompt = "You can call the function `search_document` when needed."

system = [
    {"text": system_prompt},
]

inferenceConfig = {
    "maxTokens": RAGConfig.max_tokens,
    "temperature": RAGConfig.model_temperature,
    "topP": 1.0,
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
    "toolChoice": {"auto": {}},
}


class BedrockManager:
    @staticmethod
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

    def __init__(self, config: RAGConfig):
        self.config = config
        self.session = boto3.Session()
        self.bedrock = self.session.client(
            RAGConfig.model_runtime, region_name=RAGConfig.model_aws_region
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap
        )

    @safe
    def get_model_response_stream(self, toolConfig, messages):
        pc = PineconeManager(self.config)

        if not messages:
            raise ValueError("Messages missing")

        if not toolConfig:
            raise ValueError("tools configuration is missing")

        logger.debug(
            f"Invoking model {self.config.model_id} with TOOLS and no streaming first..."
        )

        response = self.bedrock.converse(
            modelId=self.config.model_id,
            system=system,
            messages=messages,
            inferenceConfig=inferenceConfig,
            toolConfig=toolConfig,
        )

        full_response = ""
        tool_calls = []

        logger.debug(f"\n First response: {json.dumps(response, indent=2)}")
        
        """   sample 
        
         {
            {
            "ResponseMetadata": {
                "RequestId": "6d7c65da-dfb2-4a00-92c3-563866c4e836",
                "HTTPStatusCode": 200,
                "HTTPHeaders": {
                "date": "Fri, 30 May 2025 15:01:44 GMT",
                "content-type": "application/json",
                "content-length": "298",
                "connection": "keep-alive",
                "x-amzn-requestid": "6d7c65da-dfb2-4a00-92c3-563866c4e836"
                },
                "RetryAttempts": 0
            },
            "output": {
                "message": {
                "role": "assistant",
                "content": [
                    {
                    "toolUse": {
                        "toolUseId": "tooluse_b8oV3HLkTGaNl1TAk88vfQ",
                        "name": "search_document",
                        "input": {
                        "query": "key advantages of LoRA"
                        }
                    }
                    }
                ]
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 94,
                "outputTokens": 25,
                "totalTokens": 119
            },
            "metrics": {
                "latencyMs": 1002
            }
            } """

        for block in response.get("output", {}).get("message", {}).get("content", []):
            tool_use = block.get("toolUse")
            
            if tool_use:
                tool_calls.append({
                "tool_use_id": tool_use["toolUseId"],
                "name":        tool_use["name"],
                "input":       tool_use["input"],
            })

        #tool_calls = response.get("output", {}).get("message", {}).get("content", [])
        
        logger.debug(f"Tool calls: {tool_calls}")
        
        for tc in tool_calls:
            if tc.get("name") == "search_document":
                args = tc.get("input", {})
                search_result = pc.search(args["query"])
                logger.debug(
                        f"Search result for query '{args['query']}':\n{search_result}"
                    )
                
                messages.append({
                    "role": "assistant", 
                     "content": [ 
                          {"text": search_result}
                          ]
                         })
                messages.append({
                    "role": "user", 
                    "content": 
                    [
                        {"text": "please summarise the search results in three bullet points"}
                        ]
                    })

     
        logger.info("\n\n Processing search results...\n")
            
            
            
        # ---- second pass --------------------------------------------------

        logger.info(f"\n\n {self.config.model_id} called again with streaming and no tools ...")
        logger.debug(f"messages: {messages}")   

        final_response = self.bedrock.converse_stream(
                modelId=self.config.model_id,
                system=system,
                messages=messages,
                inferenceConfig=inferenceConfig         
            )

        full_response = ""
        for chunk in final_response["stream"]:
            text = chunk.get("contentBlockDelta", {}).get("delta", {}).get("text", "")
            if text:
                print(text, end="", flush=True)
                full_response += text
            if chunk.get("messageStop", {}).get("stopReason") == "stop":
                break
        print()
        return full_response
