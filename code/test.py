import boto3
import json
import os 
from dotenv import load_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec


# --- Load environment ---
load_dotenv()
PINECONE_KEY = os.getenv("PINECONE_KEY")


# --- Config ---
MODEL_ID = "mistral.mistral-large-2407-v1:0"
MODEL_TEMPERATURE = 1
MAX_TOKENS = 2048
REGION = "us-west-2"
RUNTIME = "bedrock-runtime"

INDEX_NAME = "ia-index"
EMBEDDING_DIM = 384
METRIC = "cosine"
CLOUD = "aws"
PINECONE_REGION = "us-east-1"


model_prompt = "Explain the theory of relativity in simple terms."
query_prompt="what is the adapter Layers and Inference Latency"

#RAG-model-practice/code/test.py    

def init_bedrock():
    session = boto3.Session()
    return session.client(RUNTIME, region_name=REGION)

def init_pinecone() -> Pinecone:
    return Pinecone(api_key=PINECONE_KEY)

# --- Load and process document ---
def load_document(path: str):
    abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", path))
    loader = PyMuPDFLoader(abs_path)
    return loader.load()

def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    return splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# --- Pinecone index ---
def create_index(pc: Pinecone):
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=PINECONE_REGION)
    )
    return pc.Index(INDEX_NAME)

def delete_index(pc: Pinecone, index_name: str):
    if index_name in [idx.name for idx in pc.list_indexes()]:
        pc.delete_index(index_name)
        print(f"Index '{index_name}' deleted.")
    else:
        print(f"Index '{index_name}' does not exist.")

def upload_to_pinecone(index, chunks):
    vectors = embedding_model.embed_documents([doc.page_content for doc in chunks])
    to_upsert = [
        {
            "id": f"doc-{i}",
            "values": vectors[i],
            "metadata": chunks[i].metadata
        }
        for i in range(len(chunks))
    ]
    index.upsert(vectors=to_upsert)

def search_index(index, query: str):
    vector = embedding_model.embed_query(query)
    return index.query(vector=vector, top_k=3, include_metadata=True, include_values=True)

def display_results(response):
    print("\nTop matching documents:")
    for match in response.get('matches', []):
        print(f"\nScore: {match['score']:.4f}")
        print(f"Text: {match.get('metadata', {}).get('text', '[No text found]')}")
        print(f"Metadata: {match.get('metadata')}")


def call_mistral(prompt: str):
    bedrock = init_bedrock()
    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": MODEL_TEMPERATURE,
            "top_p": 1
        }),
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response['body'].read())
    print(json.dumps(result, indent=2))

# Streaming delivery

"""
 response = bedrock.invoke_model_with_response_stream(
        modelId=modelID,
        body=json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": model_temperature,
            "top_p": 1
        }),
        contentType="application/json",
        accept="application/json"

stream = response.get('body')
if stream:
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            chunk_obj = json.loads(chunk.get('bytes').decode())
            content = (chunk_obj.get("choices", [{}])[0].get("message", {}).get("content", ""))
            if content: print(content, end="", flush=True)
"""
if __name__ == "__main__":
    docs = load_document("example.pdf")
    chunks = split_document(docs)

    pc = init_pinecone()
    delete_index(pc, INDEX_NAME)  
      
    index = create_index(pc)
    
    upload_to_pinecone(index, chunks)
    response = search_index(index, "What are adapter layers?")
    display_results(response)
    call_mistral("Explain adapter layers in simple terms.")