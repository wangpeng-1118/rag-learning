
from rag.loader import load_txt
from rag.splitter import split_text
from rag.embeddings import EmbeddingModel
from rag.retriever import retrieve


def main():
    document = load_txt("data/documents/rag_intro.txt")

    chunks = split_text(
        document,
        chunk_size=200,
        overlap=30,
    )

    embedding_model = EmbeddingModel()

    texts = [chunk.page_content for chunk in chunks]
    chunk_vectors = embedding_model.encode(texts)

    query = "How does RAG retrieve relevant information?"

    results = retrieve(
        query=query,
        chunks=chunks,
        chunk_vectors=chunk_vectors,
        embedding_model=embedding_model,
        top_k=3,
    )

    print(f"\n用户问题：{query}")
    print("\n检索结果：")

    for rank, result in enumerate(results, start=1):
        print(f"\n排名：{rank}")
        print(f"相似度：{result['score']:.4f}")
        print(f"文本：{result['document'].page_content}")
        print(f"来源：{result['document'].metadata['file_name']}")


if __name__ == "__main__":
    main()
