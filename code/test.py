import boto3
import json
import os 

PINECONDE_KEY=${PINECONDE_KEY}

#bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")  # use correct region

modelID = "mistral.mistral-large-2407-v1:0"
model_temperature = 1
max_tokens = 2048



prompt = "Explain the theory of relativity in simple terms."
query_prompt="what is the adapter Layers and Inference Latency"

'''
session=boto3.Session()

bedrock = session.client('bedrock-runtime', 'us-west-2')

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
)

stream = response.get('body')
if stream:
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            chunk_obj = json.loads(chunk.get('bytes').decode())
            content = (chunk_obj.get("choices", [{}])[0].get("message", {}).get("content", ""))
            if content: print(content, end="", flush=True)


'''



# non stream delivery

'''

response = bedrock.invoke_model(
    modelId=modelID,  
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": model_temperature,
        "top_p": 1
    })
)

result = json.loads(response['body'].read())

print(json.dumps(result, indent=2))

'''

#vectores


# loading the document

from langchain_community.document_loaders.pdf import PyMuPDFLoader

project_root = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(project_root, "..", "sample.pdf")

loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

#print(docs) 


# splitting the document


from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
chunks = splitter.split_documents(docs)

#for chunk in chunks:
#    print(chunk)




# embedding model setup from HF

from langchain_community.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# creating indexing within Pinecone

from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=PINECONDE_KEY)

index_name = "ia-index"
dimension_ = 384  # ✅ for multilingual-MiniLM-L12-v2
metric_ = "cosine"
cloud_ = "aws"
region_ = "us-east-1"



pc.create_index(
    name=index_name,
    dimension=dimension_, # Replace with your model dimensions
    metric=metric_, # Replace with your model metric
    spec=ServerlessSpec(
        cloud=cloud_,
        region=region_
    )
)

index = pc.Index(index_name)

#  embed the chunks
vectors = embedding_model.embed_documents([doc.page_content for doc in chunks])

#  format and upsert
to_upsert = [
    {
        "id": f"doc-{i}",
        "values": vectors[i],
        "metadata": chunks[i].metadata,
        "sparce_values": chunks[i].page_content  
    }
    for i in range(len(chunks))
]

index.upsert(vectors=to_upsert)




## search for the vector

pc = Pinecone(api_key=PINECONDE_KEY)

index = pc.Index(index_name)


# Step 1: Embed the query prompt
query_vector = embedding_model.embed_query(query_prompt)

# Step 2: Query the Pinecone index
query_response = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True,
    include_values=True
)

# Step 3: Print results
print("\nTop matching documents:")
for match in query_response['matches']:
    score = match['score']
    metadata = match.get('metadata', {})
    text = metadata.get('text', '[No text found in metadata]')
    print(f"\nScore: {score:.4f}")
    print(f"Text: {text}")
    print(f"Metadata: {metadata}")
