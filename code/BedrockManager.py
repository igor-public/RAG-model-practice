import boto3
import logging
from langchain_core.utils import safe
from RAGConfig import MODEL_REGION, MODEL_RUNTIME, MODEL_ID, MAX_TOKENS, TOOLS, MODEL_TEMPERATURE   
import json
import PineconeManager

logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format="%(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

@safe
def init_bedrock():
    session = boto3.Session()
    return session.client(MODEL_RUNTIME, region_name=MODEL_REGION)


@safe
def getModelResponse_stream( bedrock, TOOLS: list, MODEL_ID, messages, MAX_TOKENS, MODEL_TEMPERATURE):
    
    """
    Invoke a model with response streaming.
    
    :param bedrock: The Bedrock client.
    :param MODEL_ID: The ID of the model to invoke.
    :param messages: The messages to send to the model.
    :param MAX_TOKENS: The maximum number of tokens to generate.
    :param TOOLS: The tools available for the model.
    :param MODEL_TEMPERATURE: The temperature setting for the model.
    :return: A response stream from the model invocation.
    """
    
    if not bedrock:
        raise ValueError("Bedrock client is not initialized.")

    if not MODEL_ID:
        raise ValueError("Model ID is required.")

    if not messages:
        raise ValueError("Messages are required for model invocation.")

    if MAX_TOKENS <= 0:
        raise ValueError("MAX_TOKENS must be a positive integer.")

    if not TOOLS:
        raise ValueError("TOOLS must be provided for the model invocation.")

    if MODEL_TEMPERATURE < 0 or MODEL_TEMPERATURE > 1:
        raise ValueError("MODEL_TEMPERATURE must be between 0 and 1.")

    logger.info(f"Invoking model {MODEL_ID} with response streaming...")
    
    # Invoke the model with response streaming
        
    response = bedrock.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        body=json.dumps(
            {
                "messages": messages,
                "max_tokens": MAX_TOKENS,
                "tool_choice": "auto",
                "tools": TOOLS,
                "temperature": MODEL_TEMPERATURE,
                "stream": True,
                "top_p": 1,
            }
        ),
        contentType="application/json",
        accept="application/json",
    )

    full_response = ""
    tool_calls = []

    logger.debug(f"\n {MODEL_ID} called now ...")

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
        logger.debug("No calls for tools where seen")
        return

    if tc and all(item in tool_calls for item in tc):

        logger.debug("\n Tools initiated, processing...\n")

        for tc in tool_calls:
            if tc["function"]["name"] == "search_document":
                args = json.loads(tc["function"]["arguments"])
                search_result = PineconeManager.search(args["query"])
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

        logger.debug("\n\nProcessing search results...\n")

        final_response = bedrock.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps(
                {
                    "messages": messages,
                    "max_tokens": MAX_TOKENS,
                    "temperature": MODEL_TEMPERATURE,
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
            if choice.get("finish_reason") == "stop":
                break
        print()
