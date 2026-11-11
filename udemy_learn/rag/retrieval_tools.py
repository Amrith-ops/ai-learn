from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from udemy_learn.rag.embeddings import get_embedding_model_response


class EmbeddedDocument(TypedDict):
    embedding: list[float]
    content: str


def calculate_cosine_similarity(
    vec1: Sequence[float], vec2: Sequence[float]
) -> float:
    """Measure the angle-based similarity between two embedding vectors."""
    if len(vec1) != len(vec2):
        raise ValueError("Embedding vectors must have the same number of dimensions.")

    dot_product = sum(left * right for left, right in zip(vec1, vec2))
    norm_vec1 = sum(value * value for value in vec1) ** 0.5
    norm_vec2 = sum(value * value for value in vec2) ** 0.5
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


def rank_documents_by_similarity(
    query_embedding: Sequence[float],
    document_embeddings: Mapping[str, EmbeddedDocument],
) -> list[tuple[str, float]]:
    """Return file names ordered by cosine similarity, highest first."""
    similarities = [
        (
            file_name,
            calculate_cosine_similarity(query_embedding, document["embedding"]),
        )
        for file_name, document in document_embeddings.items()
    ]
    similarities.sort(key=lambda item: item[1], reverse=True)
    return similarities


def load_documents_and_create_embeddings(
    data_folder: str | Path,
) -> dict[str, EmbeddedDocument]:
    """Load text files and create one embedding for each full document."""
    data_path = Path(data_folder)
    document_embeddings: dict[str, EmbeddedDocument] = {}

    for file in sorted(data_path.glob("*.txt")):
        content = file.read_text(encoding="utf-8")
        document_embeddings[file.name] = {
            "embedding": get_embedding_model_response(content),
            "content": content,
        }

    return document_embeddings


def user_query_embedding_create(query: str) -> list[float]:
    """Create an embedding vector for the user's query."""
    return get_embedding_model_response(query)
