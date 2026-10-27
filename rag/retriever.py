import numpy as np

from rag.loader import Document

def cosine_similarity(query_vector, document_vectors):
    """ 计算查询向量与多个文档向量的余弦相似度"""
    query_vector = np.asarray(query_vector)
    document_vectors = np.asarray(document_vectors)

    if query_vector.ndim != 1:
        raise ValueError("query_vector must be a 1D array")

    if document_vectors.ndim != 2:
        raise ValueError("document_vectors must be a 2D array")

    if query_vector.shape[0] != document_vectors.shape[1]:
        raise ValueError("query and document vector dimensions must match")

    if not np.all(np.isfinite(query_vector)):
        raise ValueError("query_vector contains invalid values")

    if not np.all(np.isfinite(document_vectors)):
        raise ValueError("document_vectors contain invalid values")

    # 计算向量的模长（长度）
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)

    denominator = document_norms * query_norm

    # 避免向量长度为0时发生除零错误
    denominator = np.where(denominator == 0, 1e-12, denominator)

    # np.dot()两个向量的点积
    return np.dot(document_vectors, query_vector) / denominator

def retrieve(
        query: str,
        chunks: list[Document],
        chunk_vectors: np.ndarray,
        embedding_model,
        top_k: int = 3,
) -> list[dict]:
    """ 根据用户的问题，检索最相关的文本块"""

    if not query.strip():
        raise ValueError("query must not be empty")
    
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if len(chunks) != len(chunk_vectors):
        raise ValueError("the number of chunks must match the number of vectors")

    if not chunks:
        return []

    query_vector = embedding_model.encode([query])[0]

    scores = cosine_similarity(query_vector, chunk_vectors)

    # np.argsort()默认按照从小到大的顺序排序，返回的是元素的索引
    # 后面是python的切片处理
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append({
            "document": chunks[index],
            "score": float(scores[index]),
        })

    return results

