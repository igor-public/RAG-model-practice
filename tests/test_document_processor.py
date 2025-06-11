import os
import pytest
from types import SimpleNamespace
from langchain.schema import Document
import src.rag.DocumentProcessor as dm   
from src.rag.RAGConfig import RAGConfig, RAGSystemException


@pytest.fixture
def cfg(tmp_path):
    # use small chunks for testing, overlap ≤ chunk size
    return RAGConfig(chunk_size=10, chunk_overlap=2)


@pytest.fixture(autouse=True)
def patch_loader(monkeypatch):
   
    def fake_loader_cls(path):
        return SimpleNamespace(load=lambda: ["page1", "page2"])
    
    monkeypatch.setattr(dm, "PyMuPDFLoader", fake_loader_cls)
    yield


def test_load_document_file_not_found(cfg, tmp_path, monkeypatch):
    proc = dm.DocumentProcessor(cfg)
    missing = str(tmp_path / "no.pdf")

    # simulate missing file
    monkeypatch.setattr(os.path, "exists", lambda p: False)

    with pytest.raises(RAGSystemException) as exc:
        proc.load_document(missing)
    assert "Failed to load document" in str(exc.value)


def test_load_document_empty_pages(cfg, tmp_path, monkeypatch):
    proc = dm.DocumentProcessor(cfg)
    test_pdf = str(tmp_path / "doc.pdf")

    # file exists, but loader returns empty list
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    monkeypatch.setattr(dm, "PyMuPDFLoader",
                        lambda path: SimpleNamespace(load=lambda: []))

    with pytest.raises(RAGSystemException) as exc:
        proc.load_document(test_pdf)
    assert "No content loaded" in str(exc.value)


def test_load_document_success(cfg, tmp_path, monkeypatch):
    proc = dm.DocumentProcessor(cfg)
    test_pdf = str(tmp_path / "doc.pdf")

    # file exists, loader returns dummy list
    monkeypatch.setattr(os.path, "exists", lambda p: True)
    # Our patch_loader fixture makes load() return ["page1","page2"]

    docs = proc.load_document(test_pdf)
    assert docs == ["page1", "page2"]


def test_split_documents_empty(cfg):
    proc = dm.DocumentProcessor(cfg)
    with pytest.raises(RAGSystemException) as exc:
        proc.split_documents([])
    assert "No documents provided for splitting" in str(exc.value)


def test_split_documents_success(cfg, monkeypatch):
    proc = dm.DocumentProcessor(cfg)
    # prepare fake Document instances
    docs = [Document(page_content="hello world", metadata={}),
            Document(page_content="foo bar", metadata={"foo": "bar"})]

    # monkeypatch the splitter to return two chunks per doc
    fake_chunks = [
        Document(page_content="chunk1", metadata={}),
        Document(page_content="chunk2", metadata={}),
    ]
    monkeypatch.setattr(proc, "text_splitter",
                        SimpleNamespace(split_documents=lambda d: fake_chunks))

    chunks = proc.split_documents(docs)
    # we should get exactly fake_chunks back, but with augmented metadata
    assert len(chunks) == 2
    for idx, ch in enumerate(chunks):
        assert ch.metadata["chunk_id"] == idx
        assert ch.metadata["chunk_size"] == len(ch.page_content)
        assert ch.metadata["text"] == ch.page_content