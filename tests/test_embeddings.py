
from rag.embeddings import EmbeddingModel


def test_embedding_shape():
    model = EmbeddingModel()

    texts = [
        "RAG retrieves relevant documents.",
        "RAG 检索相关文档。",
    ]

    vectors = model.encode(texts)

    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    print(vectors.shape)


def test_embedding_normalization():
    model = EmbeddingModel()

    vectors = model.encode(["This is a test sentence."])

    norms = (vectors ** 2).sum(axis=1) ** 0.5

    assert abs(norms[0] - 1.0) < 1e-5

if __name__ == "__main__":
    test_embedding_shape()