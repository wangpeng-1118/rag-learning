from rag.loader import Document

def split_text(
    document: Document,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Document]:
    """ Split a Document into multiple text chunks with source information"""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap could not less than 0")

    if overlap >= chunk_size:
        raise ValueError("overlap must less than chunk_size")

    text = document.page_content
    chunks = []
    start = 0

    while start < len(text):
        end =min(start + chunk_size, len(text))
        chunk_content = text[start:end]

        metadata = {
            **document.metadata,
            "chunk_start": start,
            "chunk_end": end,
        }

        chunks.append(
            Document(
                page_content=chunk_content,
                metadata=metadata,
            )
        )

        if end == len(text):
            break

        start = end - overlap

    return chunks

    