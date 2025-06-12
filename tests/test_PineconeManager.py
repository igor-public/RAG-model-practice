from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import logging

from src.rag.config.RAGConfig import RAGConfig
from src.rag.PineconeManager import PineconeManager

logging.basicConfig(
    level=logging.INFO,
    # format='%(name)s - %(message)s'
    format="%(name)s - %(levelname)s - %(message)s",
)

def fake_index(name="ia-index"):
    
    idx = MagicMock()
    idx.name = name
    idx.describe_index_stats.return_value = SimpleNamespace(total_vector_count=0)
    return idx


@pytest.fixture()
def pc(monkeypatch):
    """
    Patch the Pinecone class so that every
    `Pinecone(api_key=...)` call returns the same MagicMock client.
    """
    mock_client = MagicMock()
    mock_client.list_indexes.return_value = []        # default: no indexes yet
    mock_client.Index.return_value = fake_index()

    mock_pc_cls = MagicMock(return_value=mock_client)
    monkeypatch.setattr("rag.PineconeManager.Pinecone", mock_pc_cls)
    return mock_client   # handed to the test for assertions


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_ensure_index_creates_and_returns_index(pc):
    cfg = RAGConfig()
    mgr = PineconeManager(cfg)

    ix = mgr.ensure_index()

    pc.create_index = pc.create_index  # noqa: B018  (explicit access)
    pc.create_index.assert_called_once()               # index created
    pc.Index.assert_called_with(cfg.index_name)        # handle returned
    assert ix is pc.Index.return_value


""" 
if __name__ == "__main__":
    
    pc = PineconeManager(RAGConfig()) 
    pc.ensure_index()
    pc.search("What is the Low-Rank Adaptation?")   """     