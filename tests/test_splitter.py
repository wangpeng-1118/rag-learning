
import pytest

from rag.loader import Document
from rag.splitter import split_text


def test_split_text_with_overlap():
    doc = Document(
        page_content="abcdefghijklmnopqrst",
        metadata={"file_name": "test.txt"},
    )

    chunks = split_text(doc, chunk_size=10, overlap=2)

    assert [chunk.page_content for chunk in chunks] == [
        "abcdefghij",
        "ijklmnopqr",
        "qrst",
    ]

    assert all(
        chunk.metadata["file_name"] == "test.txt"
        for chunk in chunks
    )


def test_split_text_invalid_parameters():
    doc = Document(page_content="hello world", metadata={})

    with pytest.raises(ValueError):
        split_text(doc, chunk_size=0)

    with pytest.raises(ValueError):
        split_text(doc, chunk_size=10, overlap=10)
