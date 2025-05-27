import boto3
import json
import os
import logging, pprint
from functools import wraps
from code.RAGConfig import RAGConfig, RAGSystemException
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import BedrockManager
from PineconeManager import PineconeMager, Pinecone
from DocumentProcessor import DocumentProcessor

logging.basicConfig(
    level=logging.INFO,
    # format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format="%(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def safe(fn):

    """
    All exceptions are logged and raised again as ``RAGSystemException``. 
    """

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

# Environment variables. Requires PINECONE_KEY (free tier)

load_dotenv(find_dotenv("local.env"))
PINECONE_KEY = os.getenv("PINECONE_KEY")

logger.debug(PINECONE_KEY)


# --- Config ---
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

model_prompt = "Explain the theory of relativity in simple terms."
SEARCH_QUESTION = "What is the key advantages of LoRA?"
query_prompt = "what is the adapter Layers and Inference Latency"

# RAG-model-practice/code/test.py

def init_bedrock():
    session = boto3.Session()
    return session.client(MODEL_RUNTIME, region_name=MODEL_REGION)


def init_pinecone() -> Pinecone:
    return PineconeMager.init(api_key=PINECONE_KEY)


def get_index(pc: Pinecone, index_name: str):
    index_stats = pc.Index(index_name).describe_index_stats()
    logger.info(
        f"Index '{index_name}' is found and has {index_stats.total_vector_count} vectors"
    )
    return pc.Index(index_name)


# --- Load and process document ---
def load_document(path: str):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
    loader = PyMuPDFLoader(abs_path)
    return loader.load()


def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return splitter.split_documents(docs)


logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# --- Pinecone index ---

def create_index(pc: Pinecone):
    index = pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=PINECONE_REGION),
    )
    index_stats = pc.Index(INDEX_NAME).describe_index_stats()
    logger.info(
        f"Index has been created and populated with '{index_stats.total_vector_count}' vectors"
    )
    return pc.Index(INDEX_NAME)


def delete_index(pc: Pinecone, index_name: str):
    if index_name in [idx.name for idx in pc.list_indexes()]:
        pc.delete_index(index_name)
        logger.info(f"Index '{index_name}' deleted.")
    else:
        logger.warning(f"Index '{index_name}' does not exist.")


def upload_to_pinecone(ind: Pinecone.Index, chunks):
    
    logger.info(f"Uploading '{len(chunks)}' chunks to Pinecone index:  '{INDEX_NAME}'")
    vectors = embedding_model.embed_documents([doc.page_content for doc in chunks])
    to_upsert = [
        {
            "id": f"doc-{i}",
            "values": vectors[i],
            "metadata": chunks[i].metadata,
            "metadata": {
                **chunks[i].metadata,
                "text": chunks[i].page_content,  # Store the actual text
            },
        }
        for i in range(len(chunks))
    ]
    logger.info(
        f"Uploading '{len(to_upsert)}' vectors to Pinecone index:  '{INDEX_NAME}'"
    )
    ind.upsert(vectors=to_upsert)


def search_index(index, query: str) -> str:

    logger.info(f"Tool called - searching for: '{query}'")

    vector = embedding_model.embed_query(query)

    response = index.query(
        vector=vector, top_k=10, include_metadata=True, include_values=True
    )

    if not response.get("matches"):
        return "No relevant documents found."

    context_pieces = []

    for match in response["matches"]:
        text = match.get("metadata", {}).get("text", "")
        score = match["score"]
        context_pieces.append(f"[Relevance: {score:.3f}] {text}")

    return "\n\n".join(context_pieces)


def search_document(query: str) -> str:

    pc = init_pinecone()
    index = get_index(pc, INDEX_NAME)

    return search_index(index, query)


def display_results(response):

    matches = response.get("matches", [])
    logger.info(f"Found {len(matches)} matching documents")

    for match in response.get("matches", []):
        logger.info(f"\nScore: {match['score']:.4f}")
        logger.info(f"Text: {match.get('metadata', {}).get('text', '[No text found]')}")
        logger.debug(f"Metadata: {match.get('metadata')}")


# Define available tools for Mistral
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



# Streaming delivery



if __name__ == "__main__":

    # docs = load_document("sample.pdf")
    # chunks = split_document(docs)

    # pc = init_pinecone()
    # index = get_index(pc,INDEX_NAME)
    # delete_index(pc, INDEX_NAME)
    # index = create_index(pc)
    # upload_to_pinecone(index, chunks)

    # response = search_index(index, SEARCH_QUESTION)
    # display_results(response)

    # context = "\n".join([match['metadata']['text'] for match in response['matches']])
    # print(f"the conext: {context}")


    pc = PineconeMager(api_key=PINECONE_KEY)
    prompt = f"Q: {SEARCH_QUESTION}\n\nAnswer:"
    
    BedrockManager.getModelResponse_stream(prompt)
