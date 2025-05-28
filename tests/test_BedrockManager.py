# tests/test_bedrock_manager.py

from code.BedrockManager import BedrockManager
from code.RAGConfig import RAGConfig


def testInitializeBedrockManager():
    # -- given
    cfg = RAGConfig()

    """     bedrock = MagicMock()
    pinecone = MagicMock() """

    # -- when
    mgr = BedrockManager(cfg)

    # -- then
    assert mgr.config == cfg
    assert mgr.session is not None
    assert mgr.text_splitter is not None
    assert isinstance(mgr.text_splitter, type(mgr.text_splitter))


def testGetModelResponse_stream():
    # -- given
    cfg = RAGConfig()

    mgr = BedrockManager(cfg)

    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "search_document",
                "description": "Vector‑search the document collection and return relevant chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search terms"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]

    _messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that answers questions concisely.",
        },
        {
            "role": "user",
            "content": "Explain adapter layers and inference latency in one sentence.",
        },
    ]

    # -- when
    response = mgr.get_model_response_stream(
        tools=TOOLS,
        model_id=cfg.model_id,
        messages=_messages,
        max_tokens=cfg.max_tokens,
        model_temperature=cfg.model_temperature,
    )

    # -- then
    assert response is not None
    assert isinstance(response, str)  # Assuming the response is a string
    assert "Adapter layers" in response  # Check if the log message is present


""" def testGetModelResponse_stream_with_invalid_params():
    # -- given
    cfg = RAGConfig()
    mgr = BedrockManager(cfg)
    
    # -- when
    try:
        mgr.getModelResponse_stream(
            bedrock=None,
            TOOLS=None,
            MODEL_ID=None,
            messages=None,
            MAX_TOKENS=-1,  # Invalid value
            MODEL_TEMPERATURE=2.0  # Invalid value
        )
    except ValueError as e:
        # -- then
        assert str(e) == "MAX_TOKENS must be a positive integer."   # or "MODEL_TEMPERATURE must be between 0 and 1."
    else:       
        assert False, "Expected ValueError was not raised"                    """
