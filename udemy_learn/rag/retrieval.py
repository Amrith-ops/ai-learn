import math
from collections.abc import Mapping, Sequence




def calculate_cosine_similarity(
    vec1: Sequence[float], vec2: Sequence[float]
) -> float:
    """Measure the angle-based similarity between two embedding vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Embedding vectors must have the same number of dimensions.")

    dot_product = sum(left * right for left, right in zip(vec1, vec2))
    norm_vec1 = math.sqrt(sum(value * value for value in vec1))
    norm_vec2 = math.sqrt(sum(value * value for value in vec2))
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


def rank_documents_by_similarity(
    query_embedding: Sequence[float],
    document_embeddings: Mapping[str, Sequence[float]],
) -> list[tuple[str, float]]:
    """Rank document IDs by cosine similarity, most similar first."""
    similarities = []
    for doc_id, doc_embedding in document_embeddings.items():
        similarity = calculate_cosine_similarity(query_embedding, doc_embedding)
        similarities.append((doc_id, similarity))

    # This reusable ranking stage is ready for the future RAG flow, not yet
    # connected to the assistant's current exact-word document search.
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities
