import boto3
import json
import os
import logging
from functools import wraps
from typing import List, Dict, Any

from code.RAGConfig import RAGConfig, RAGSystemException  # <- import the custom error
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

###############################################################################
# Logging setup
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s: %(levelname)s – %(message)s",
)
logger = logging.getLogger(__name__)

###############################################################################
# Helper – decorator for uniform exception handling
###############################################################################

def safe(fn):
    """Wrap a function so that any exception is logged and re‑raised as
    ``RAGSystemError``. Keeps all call‑sites concise and consistent."""

    @wraps(fn)
    def _wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RAGSystemError:
            raise                                     # already wrapped
        except Exception as exc:                      # noqa: BLE001   (ruff)
            logger.error("%s failed: %s", fn.__name__, exc, exc_info=True)
            raise RAGSystemError(str(exc)) from exc

    return _wrapper

###############################################################################
# Environment & config  -------------------------------------------------------
###############################################################################
load_dotenv(find_dotenv("local.env"))
PINECONE_KEY = os.getenv("PINECONE_KEY") or ""
if not PINECONE_KEY:
    raise RAGSystemError("PINECONE_KEY not found in environment")

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

###############################################################################
# Low‑level initialisation helpers  -------------------------------------------
###############################################################################


@safe
def init_bedrock():
    session = boto3.Session()
    return session.client(MODEL_RUNTIME, region_name=MODEL_REGION)


@safe
def init_pinecone() -> Pinecone:
    return Pinecone(api_key=PINECONE_KEY)


@safe
def get_index(pc: Pinecone, index_name: str):
    index_stats = pc.Index(index_name).describe_index_stats()
    logger.info("Index '%s' is found and has %s vectors", index_name, index_stats.total_vector_count)
    return pc.Index(index_name)


###############################################################################
# Document loading & splitting  ----------------------------------------------
###############################################################################


@safe
def load_document(path: str):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
    loader = PyMuPDFLoader(abs_path)
    return loader.load()


@safe
def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return splitter.split_documents(docs)


###############################################################################
# Embeddings (lazy‑load once)  -------------------------------------------------
###############################################################################

logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

embedding_model = HuggingFaceEmbeddings(
    model_name=RAGConfig.embedding_model_name,
)

###############################################################################
# Pinecone index helpers  ------------------------------------------------------
###############################################################################


@safe
def create_index(pc: Pinecone):
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=PINECONE_REGION),
    )
    index_stats = pc.Index(INDEX_NAME).describe_index_stats()
    logger.info("Index created with %s vectors", index_stats.total_vector_count)
    return pc.Index(INDEX_NAME)


@safe
def delete_index(pc: Pinecone, index_name: str):
    if index_name in [idx.name for idx in pc.list_indexes()]:
        pc.delete_index(index_name)
        logger.info("Index '%s' deleted", index_name)
    else:
        logger.warning("Index '%s' does not exist", index_name)


@safe
def upload_to_pinecone(ind: Pinecone.Index, chunks):
    vectors = embedding_model.embed_documents([doc.page_content for doc in chunks])
    to_upsert: List[Dict[str, Any]] = []
    for i, (vec, chunk) in enumerate(zip(vectors, chunks)):
        to_upsert.append(
            {
                "id": f"doc-{i}",
                "values": vec,
                "metadata": {**chunk.metadata, "text": chunk.page_content},
            }
        )
    ind.upsert(vectors=to_upsert)
    logger.info("Upserted %s vectors to index '%s'", len(to_upsert), INDEX_NAME)


###############################################################################
# Vector search  --------------------------------------------------------------
###############################################################################


@safe
def search_index(index, query: str) -> str:
    vector = embedding_model.embed_query(query)
    response = index.query(
        vector=vector, top_k=10, include_metadata=True, include_values=False
    )
    if not response.get("matches"):
        return "No relevant documents found."
    context_lines: List[str] = []
    for match in response["matches"]:
        text = match.get("metadata", {}).get("text", "")
        score = match["score"]
        context_lines.append(f"[Relevance: {score:.3f}] {text}")
    return "\n\n".join(context_lines)


###############################################################################
# Tool‑function wrapper  -------------------------------------------------------
###############################################################################


@safe
def search_document_tool(query: str) -> str:
    pc = init_pinecone()
    index = get_index(pc, INDEX_NAME)
    return search_index(index, query)


###############################################################################
# Mistral chat with tools  -----------------------------------------------------
###############################################################################


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


@safe
def call_mistral(prompt: str):
    bedrock = init_bedrock()

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

    tool_calls: List[Dict[str, Any]] = []

    # ── first LLM pass (collect potential tool calls) ──────────────────────
    for event in response["body"]:
        chunk = event.get("chunk")
        if not chunk:
            continue
        data = json.loads(chunk["bytes"])
        choice = data["choices"][0]
        if content := (
            choice.get("delta", {}).get("content")
            or choice.get("message", {}).get("content", "")
        ):
            print(content, end="", flush=True)
        tc = (
            choice.get("delta", {}).get("tool_calls")
            or choice.get("message", {}).get("tool_calls")
            or []
        )
        tool_calls.extend(tc)
        if choice.get("finish_reason") in ("stop", "tool_calls"):
            break

    if not tool_calls:
        return  # normal reply already printed

    # ── execute tool calls ────────────────────────────────────────────────
    for call in tool_calls:
        if call["function"]["name"] == "search_document":
            args = json.loads(call["function"]["arguments"])
            result = search_document_tool(args["query"])
            messages.extend(
                [
                    {"role": "assistant", "tool_calls": [call]},
                    {"role": "tool", "tool_call_id": call["id"], "content": result},
                ]
            )

    # ── second LLM pass with tool results ─────────────────────────────────
    final_resp = bedrock.invoke_model_with_response_stream(
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

    for event in final_resp["body"]:
        chunk = event.get("chunk")
        if not chunk:
            continue
        data = json.loads(chunk["bytes"])
        choice = data["choices"][0]
        if content := (
            choice.get("delta", {}).get("content")
            or choice.get("message", {}).get("content", "")
        ):
            print(content, end="", flush=True)
        if choice.get("finish_reason") == "stop":
            break
    print()


###############################################################################
# Entry‑point  ----------------------------------------------------------------
###############################################################################

if __name__ == "__main__":
    prompt = "Q: What are the key advantages of LoRA?\n\nAnswer:"
    print(prompt)
    call_mistral(prompt)
