from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from code.RAGConfig import RAGConfig
from code.PineconeManager import PineconeManager

def fake_index(name="ia-index"):
    """Return a MagicMock that looks like a pinecone Index object."""
    idx = MagicMock()
    idx.name = name
    # describe_index_stats returns obj with .total_vector_count attr
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
    monkeypatch.setattr("code.PineconeManager.Pinecone", mock_pc_cls)
    return mock_client   # handed to the test for assertions


# ----------------------------------------------------------------------------
# tests
# ----------------------------------------------------------------------------
def test_ensure_index_creates_and_returns_index(pc):
    cfg = RAGConfig()
    mgr = PineconeManager(cfg, api_key="dummy")

    ix = mgr.ensure_index()

    pc.create_index = pc.create_index  # noqa: B018  (explicit access)
    pc.create_index.assert_called_once()               # index created
    pc.Index.assert_called_with(cfg.index_name)        # handle returned
    assert ix is pc.Index.return_value


def test_delete_index_when_absent(pc):
    cfg = RAGConfig()
    mgr = PineconeManager(cfg, api_key="dummy")

    # list_indexes → []   =>  should *not* call delete_index
    mgr.delete_index()
    pc.delete_index.assert_not_called()


def test_search_filters_by_threshold(pc):
    cfg = RAGConfig(similarity_threshold=0.8)
    mgr = PineconeManager(cfg, api_key="dummy")

    # fake query result with one good, one bad match
    pc.Index.return_value.query.return_value = {
        "matches": [
            {"id": "good", "score": 0.85, "metadata": {}},
            {"id": "bad",  "score": 0.50, "metadata": {}},
        ]
    }
    res = mgr.search([0.1, 0.2, 0.3])

    assert [m["id"] for m in res["matches"]] == ["good"]


def test_upsert_vector_length_mismatch_raises(pc):
    cfg = RAGConfig()
    mgr = PineconeManager(cfg, api_key="dummy")

    with pytest.raises(ValueError, match="Mismatch between vectors"):
        # 2 vectors, 1 chunk  -> should raise before hitting network
        mgr.upsert_vectors([[0.1], [0.2]], [MagicMock()])
