import time
import logging
import os


from typing import List, Any
from langchain.schema import Document
from rag.RAGConfig import RAGSystemException, RAGConfig
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import find_dotenv, load_dotenv

logging.basicConfig(
    level=logging.INFO,
    # format='%(name)s - %(message)s'
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PineconeManager:
    def __init__(self, config: RAGConfig):
        
        self.config = config

        load_dotenv(find_dotenv("local.env"))
        api_key = os.getenv("PINECONE_KEY") or ""
        if not api_key:
            raise RAGSystemException("PINECONE_KEY not found in environment")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name=RAGConfig.embedding_model_name,
        )
        self.api_key = api_key
        self._client = None
        self._index = None

    @property
    def client(self) -> Pinecone:
        # Lazy load - Pinecone client
        if self._client is None:
            try:
                self._client = Pinecone(api_key=self.api_key)
                logger.info("Initialized Pinecone client")
            except Exception as e:
                logger.error(f"Error initializing Pinecone: {str(e)}")
                raise RAGSystemException(f"Failed to initialize Pinecone: {str(e)}")

        return self._client

    def ensure_index(self) -> Any:
        try:
            existing_indexes = [idx.name for idx in self.client.list_indexes()]

            if self.config.index_name not in existing_indexes:
                logger.info(f"Creating index: {self.config.index_name}")
                self.client.create_index(
                    name=self.config.index_name,
                    dimension=self.config.embedding_dim,
                    metric=self.config.metric,
                    spec=ServerlessSpec(
                        cloud=self.config.cloud, region=self.config.pinecone_region
                    ),
                )
                # Wait for index to be ready
                time.sleep(10)

            self._index = self.client.Index(self.config.index_name)
            logger.info(f"Index {self.config.index_name} ready")
            return self._index

        except Exception as e:
            logger.error(f"Error with index operations: {str(e)}")
            raise RAGSystemException(f"Failed to ensure index: {str(e)}")

    # Delete vectors if present

    def delete_index(self) -> None:
        try:
            existing_indexes = [idx.name for idx in self.client.list_indexes()]

            if self.config.index_name in existing_indexes:
                self.client.delete_index(self.config.index_name)
                logger.info(f"Deleted index: {self.config.index_name}")
            else:
                logger.info(f"Index {self.config.index_name} does not exist")

        except Exception as e:
            logger.error(f"Error deleting index: {str(e)}")
            raise RAGSystemException(f"Failed to delete index: {str(e)}")

    # Uploading vectors to Pinecone

    def upsert_vectors(
        self, vectors: List[List[float]], chunks: List[Document]
    ) -> None:
        """Upsert vectors to Pinecone with batch processing"""
        try:
            if len(vectors) != len(chunks):
                raise ValueError("Mismatch between vectors and chunks count")

            index = self.ensure_index()

            # Prepare vectors

            to_upsert = [
                {
                    "id": f"doc-{i}",
                    "values": vectors[i],
                    "metadata": {
                        **chunks[i].metadata,
                        "text": chunks[i].page_content[
                            :1000
                        ],  # Limit text size for metadata
                    },
                }
                for i in range(len(vectors))
            ]

            # Batch upsert
            batch_size = 100
            for i in range(0, len(to_upsert), batch_size):
                batch = to_upsert[i : i + batch_size]
                index.upsert(vectors=batch)
                logger.info(
                    f"Upserted batch {i // batch_size + 1}/{(len(to_upsert) - 1) // batch_size + 1}"
                )

            logger.info(f"Successfully upserted {len(vectors)} vectors")

        except Exception as e:
            logger.error(f"Error upserting vectors: {str(e)}")
            raise RAGSystemException(f"Failed to upsert vectors: {str(e)}")

    # Searching vectors in Pinecone, returning text and metadata
    def search(self, query) -> str:
        try:
            if not self._index:
                self._index = self.client.Index(self.config.index_name)

            query_vector = self.embedding_model.embed_query(query)

            logger.info(f"Searching for: {query}")
            
            logger.debug(f"Searching for vector: {query_vector} with top_k={self.config.top_k_results} and similarity_threshold={self.config.similarity_threshold}")
            
            response = self._index.query(
                vector=query_vector,
                top_k=self.config.top_k_results,
                include_metadata=True,
                include_values=False,
            )

            logger.debug(f"Response vector: {response}")   
            
            # Filter by similarity threshold. Create a new dict only with "matches" in it.
            
            filtered_matches = [
                match
                for match in response.get("matches", [])
                if match["score"] >= self.config.similarity_threshold
            ]
            
            if not filtered_matches:
                
                logger.info("No relevant documents found after filtering")
                logger.info(f"The threshold is: {self.config.similarity_threshold}")
                logger.info(f"The highest ranking finding is: {response['matches'][0]['score'] if response['matches'] else 'N/A'}")
                logger.info(f"The top_k_results is: {self.config.top_k_results}")
                
                return "No relevant documents found."
                        
            logger.info(f"Found {len(filtered_matches)} relevant matches")
            
            context_lines: List[str] = []
            
            for match in filtered_matches:
                text = match.get("metadata", {}).get("text", "")
                score = match["score"]
                context_lines.append(f"[Relevance: {score:.3f}] {text}")
            
            logger.debug(f"Context lines: {context_lines}")
            
            return "\n\n".join(context_lines)

        except Exception as e:
            logger.error(f"Error searching vectors: {str(e)}")
            raise RAGSystemException(f"Failed to search vectors: {str(e)}")