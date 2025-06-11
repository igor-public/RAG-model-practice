# RAG Model Practice

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline.

Key features:

- **AWS Bedrock** (via boto3) for LLM inference. Requires AWS credentials locally.
- **Pinecone** for vector storage & search  
- **LangChain Community** PDF loader & text splitter  
- **Sentence-Transformers** for embeddings
- **Streaming response** capability

---

## 🚀 Installation

**From PyPI:**

```bash
pip install RAG-model-practice
```

---

## Requirements

- **Python** ≥ 3.12
- **AWS credentials** (`~/.aws/credentials`)
- **Pinecone API key** (see below)

---

## ⚙️ Environment Setup

1. **Environment Variables**

   Create a `.env` or `local.env` file in the project root:

   ```dotenv
   PINECONE_KEY="XXXXXYYYYYZZZZZ"
   ```

   *Do **not** commit this file. Add `.env`/`local.env` to `.gitignore`.*

2. **AWS credentials**

   ```bash
   user@machine:~/.aws
   # Ensure both config and credentials files exist and are valid
   ```

---

## 📦 Project Structure

```
.
├── src/rag/
│   ├── RAGConfig.py
│   ├── PineconeManager.py
│   ├── BedrockManager.py
│   ├── DocumentManager.py
│   └── mainRAG.py       # Entry point
├── sample.pdf
├── local.env
└── tests/
```

---

## ▶ Usage

After installation and configuration:

```bash
rag-run
```

Or, from source:

```bash
poetry run rag-run
```

This will:

1. Load `sample.pdf`  
2. Chunk & embed its content  
3. Upsert vectors into Pinecone  
4. Prompt the RAG system with a sample query  
5. Stream the LLM response to your console

---

## 🧪 Running Tests

Run unit tests with:

```bash
pytest -q
```

Or, to see print/log output:

```bash
pytest -q -s
```

---

## Next Steps

- Use your own PDFs in `mainRAG.py`
- Edit `RAGConfig` for different LLMs, chunking, or vector store settings
- Extend with custom tools or APIs as needed

---

## 📦 PyPI Package

You can always find the latest release on [PyPI](https://pypi.org/project/RAG-model-practice/):

[![PyPI version](https://img.shields.io/pypi/v/RAG-model-practice.svg?style=flat-square)](https://pypi.org/project/RAG-model-practice/)

Install with:

```bash
pip install RAG-model-practice
```

