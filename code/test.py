import boto3
import json
import os 
import logging
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(
    level=logging.INFO,
    #format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    format='%(name)s: %(message)s'

)
logger = logging.getLogger(__name__)


# --- Load environment ---
load_dotenv(find_dotenv("local.env"))


PINECONE_KEY = os.getenv("PINECONE_KEY")

logger.debug(PINECONE_KEY)



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

def get_index(pc: Pinecone, index_name: str):
    index_stats = pc.Index(index_name).describe_index_stats()
    logger.info(f"Index '{index_name}' is found and has {index_stats.total_vector_count} vectors")
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

# embedding_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# )

# --- Pinecone index ---
def create_index(pc: Pinecone):
    index = pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIM,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=PINECONE_REGION)
    )
    index_stats = pc.Index(INDEX_NAME).describe_index_stats()
    logger.info(f"Index has been created and populated with '{index_stats.total_vector_count}' vectors")
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
                "text": chunks[i].page_content  # Store the actual text
            }   
        }
    for i in range(len(chunks))
    ]
    logger.info(f"Uploading '{len(to_upsert)}' vectors to Pinecone index:  '{INDEX_NAME}'")
    ind.upsert(vectors=to_upsert)

def search_index(index, query: str):

    logger.info(f"Query : '{query}'")

    vector = embedding_model.embed_query(query)
    
    logger.debug(f"Vector:  '{vector}'")
    
    return index.query(vector=vector, top_k=3, include_metadata=True, include_values=True)

def display_results(response):
    matches = response.get('matches', [])
    logger.info(f"Found {len(matches)} matching documents")
    
    for match in response.get('matches', []):
        logger.info(f"\nScore: {match['score']:.4f}")
        logger.info(f"Text: {match.get('metadata', {}).get('text', '[No text found]')}")
        logger.debug(f"Metadata: {match.get('metadata')}")


def call_mistral(prompt: str):

    print (prompt)

    bedrock = init_bedrock()
    
    response = bedrock.invoke_model_with_response_stream(
        modelId=MODEL_ID,
        body=json.dumps({
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": MODEL_TEMPERATURE,
            "stream": True,
            "top_p": 1
        }),
        contentType="application/json",
        accept="application/json"
    )

    print("returning the stream ")


    stream = response["body"]

    print(stream)
    
    for event in stream:         # event-stream iterator
       
        chunk = event.get("chunk")

    
        print(chunk)

        if not chunk:
            continue                       # ping / keep-alive / error entries

        data = json.loads(chunk["bytes"])

        token = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
        
        # If you ever switch to the text-completion endpoint, use:
        # token = data.get("outputs", [{}])[0].get("text", "")
        
        if token:
            print(token, end="", flush=True)

        # graceful stop once the model says it’s done
        if data.get("choices", [{}])[0].get("finish_reason") == "stop":
            break


    #result = json.loads(response['body'].read())
    #answer = result["choices"][0]["message"]["content"]
    #logger.info(answer)

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
            if content: logger.info(content, end="", flush=True)
"""

if __name__ == "__main__":
    # docs = load_document("sample.pdf")
    # chunks = split_document(docs)

    '''
    
    pc = init_pinecone()
   
    index = get_index(pc,INDEX_NAME)
    
    # delete_index(pc, INDEX_NAME)  
      
    # index = create_index(pc)
    
    # upload_to_pinecone(index, chunks)

    
    response = search_index(index, "What are the adapter layers?")
    # display_results(response)

    context = "\n".join([match['metadata']['text'] for match in response['matches']])
    print(f"the conext: {context}")
    '''

    #prompt = f"Context: {context}\n\nQuestion: {query_prompt}\n\nAnswer:"
    
    prompt = query_prompt
    print (prompt)
    call_mistral(prompt)

    