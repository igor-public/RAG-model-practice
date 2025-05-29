import boto3
import logging
from functools import wraps
from typing import List, Dict, Any
from code.PineconeManager import PineconeManager
import json
from code.RAGConfig import RAGConfig, RAGSystemException
from langchain.text_splitter import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


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
    def get_model_response_stream(
        self,
        tools: List[Dict[str, Any]],
        messages: List[Dict[str, Any]],
    ):
        pc = PineconeManager(self.config)

        if not messages:
            raise ValueError("Messages missing")

        if not tools:
            raise ValueError("tools missing")

        logger.debug(
            f"Invoking model {self.config.model_id} with response streaming..."
        )

        response = self.bedrock.invoke_model_with_response_stream(
            modelId=self.config.model_id,
            body=json.dumps(
                {
                    "messages": messages,
                    "max_tokens": self.config.max_tokens,
                    "tool_choice": "auto",
                    "tools": tools,
                    "temperature": self.config.model_temperature,
                    "stream": True,
                    "top_p": 1,
                }
            ),
            contentType="application/json",
            accept="application/json",
        )

        full_response = ""
        tool_calls = []

        logger.debug(f"\n {self.config.model_id} called now ...")

        stream = response.get("body")

        for event in stream:
            chunk = event.get("chunk")
            if not chunk:
                continue

            data = json.loads(chunk["bytes"])
            choice = data["choices"][0]

            token = (
                choice.get("delta", {}).get("content")
                or choice.get("message", {}).get("content", "")
                or ""
            )

            if token:
                print(token, end="", flush=True)
                full_response += token

            tc = (
                choice.get("delta", {}).get("tool_calls")
                or choice.get("message", {}).get("tool_calls")
                or []
            )

            if tc:
                tool_calls.extend(tc)

            if choice.get("finish_reason") in ("stop", "tool_calls"):
                break

        if not tool_calls:
            logger.info("No calls for tools where seen")
            print()
            return full_response
            return

        # ---- run search_document tool -----------------------------------
        if tc and all(item in tool_calls for item in tc):
            logger.info("\n Tools initiated, processing...\n")

            for tc in tool_calls:
                if tc["function"]["name"] == "search_document":
                    args = json.loads(tc["function"]["arguments"])
                    search_result = pc.search(args["query"])
                    logger.debug(
                        f"Search result for query '{args['query']}':\n{search_result}"
                    )

                    messages.extend(
                        [
                            {"role": "assistant", "tool_calls": [tc]},
                            {
                                "role": "tool",
                                "tool_call_id": tc["id"],
                                "content": search_result,
                            },
                        ]
                    )

            logger.debug("\n\n Processing search results...\n")

            # ---- second pass --------------------------------------------------

            final_response = self.bedrock.invoke_model_with_response_stream(
                modelId=self.config.model_id,
                body=json.dumps(
                    {
                        "messages": messages,
                        "max_tokens": self.config.max_tokens,
                        "temperature": self.config.model_temperature,
                        "stream": True,
                    }
                ),
                contentType="application/json",
                accept="application/json",
            )

            for event in final_response["body"]:
                chunk = event.get("chunk")
                if not chunk:
                    continue

                data = json.loads(chunk["bytes"])
                choice = data["choices"][0]

                token = (
                    choice.get("delta", {}).get("content")
                    or choice.get("message", {}).get("content", "")
                    or ""
                )

                if token:
                    print(token, end="", flush=True)
                    full_response += token
                if choice.get("finish_reason") == "stop":
                    break
            print()
            return full_response
