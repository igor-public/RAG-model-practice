# RAG Model Practice

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline 
The key features: 

- **AWS Bedrock** (via boto3) for LLM inference. One needs the AWS KEY stored locally. 
- **Pinecone** for vector storage & search  
- **LangChain Community** PDF loader & text splitter  
- **Sentence-Transformers** for embeddings
- **stream response** using stream as a response

---

## Requirements

1. **Python** ≥ 3.8  
2. A working **virtual environment** (recommended):  
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # on Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   > If you don’t have a `requirements.txt`, you’ll need at least:
   > ```text
   > boto3
   > python-dotenv
   > langchain-community
   > pinecone-client
   > sentence-transformers
   > ```

4. Amazon key:
   ```bash
   user@machine:~/.aws
 
    -rw-r--r--  1 USERID USERID   43 May 26 13:13 config
    -rw-r--r--  1 USERID USERID  116 May 26 13:13 credentials

---

##  Pinecone Account

You must have a **Pinecone** account and an active API key:

1. Sign up / log in at https://app.pinecone.io  
2. Create or select an **API key** in your project  
3. Copy the key (it looks like `XXXXXYYYYYZZZZZ`)

---

## ⚙ Environment Variables

Create a file named `.env` or `local.env` in the project root with:

```dotenv
PINECONE_KEY="XXXXXYYYYYZZZZZ"
```

> **Tip:** Do **not** commit this file to version control. Add `.env`/`local.env` to your `.gitignore`.

---

## 📁 Project Structure

```
.
├── code/
│   ├── RAGConfig.py
│   ├── PineconeManager.py
│   ├── BedrockManager.py
│   ├── DocumentManager.py
│   └── mainRAG.py         ← entry-point
├── sample.pdf             ← example document
├── local.env              ← your Pinecone key
├── requirements.txt
└── tests/                 ← pytest unit tests
```

---

## ▶ Running the Example

Once your environment is set up and `local.env` is configured:

```bash
python -m code.mainRAG
```

This will:

1. Load `sample.pdf`  
2. Split it into chunks & embed them  
3. Upsert vectors into your Pinecone index  
4. Prompt the RAG system with a sample query  
5. Stream the response back to your console

---

## Running Tests

This project includes unit tests for each component. From the project root:

```bash
pytest -q
```

Use `-s` if you want to see print/log output:

```bash
pytest -q -s
```

---

## Next Steps

- Point `mainRAG.py` at your own PDFs  
- Tweak `RAGConfig` for different LLMs or indexing parameters  
- Extend `TOOLS` to add custom retrieval or external APIs  

