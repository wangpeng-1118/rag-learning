
import numpy as np

from rag.retriever import cosine_similarity


def test_cosine_similarity():
    query = np.array([1.0, 0.0])

    documents = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ])

    scores = cosine_similarity(query, documents)

    assert np.allclose(scores, [1.0, 0.0, -1.0])


