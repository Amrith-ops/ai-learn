import asyncio
import hashlib
import chromadb
import math
from pathlib import Path
from typing import Any

from udemy_learn.utils.embeddings import get_embedding_model_response


def _content_id(text: str) -> str:
    normalized_text = " ".join(text.split())
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def create_chromadb_client():

    path = (
        Path(__file__).resolve().parent
        / "data"
        / "chromadb"
        / "semantic_chroma.db"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(path))
    collection = client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
    return client,collection


async def upsert_chunks_to_chromadb(
    chunk_list_store: list[dict[str, Any]],
    collection: Any,
) -> int:
    """Upsert by normalized text; duplicate content uses the latest source metadata."""
    records_by_id: dict[
        str,
        tuple[str, list[float], dict[str, str]],
    ] = {}

    for chunk in chunk_list_store:
        text = chunk.get("text")
        source = chunk.get("source")
        chunk_id = chunk.get("chunk_id")
        embedding = chunk.get("embedding")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Each chunk must have non-empty 'text'.")
        if not isinstance(source, str) or not source.strip():
            raise ValueError("Each chunk must have a non-empty 'source'.")
        if not isinstance(chunk_id, (str, int)) or not str(chunk_id).strip():
            raise ValueError("Each chunk must have a non-empty 'chunk_id'.")
        if (
            not isinstance(embedding, list)
            or not embedding
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in embedding
            )
        ):
            raise ValueError("Each chunk must have a non-empty numeric 'embedding'.")

        content_id = _content_id(text)
        records_by_id[content_id] = (
            text,
            embedding,
            {"source": source, "chunk_id": str(chunk_id)},
        )

    if not records_by_id:
        return 0

    await asyncio.to_thread(
        collection.upsert,
        ids=list(records_by_id),
        documents=[record[0] for record in records_by_id.values()],
        embeddings=[record[1] for record in records_by_id.values()],
        metadatas=[record[2] for record in records_by_id.values()],
    )
    return len(records_by_id)


async def get_top_5_chunks_from_chromadb(
    query: str,
    collection: Any,
) -> list[dict[str, Any]]:
    """Return up to five nearest chunks with their source and distance."""
    if not query.strip():
        raise ValueError("query must not be empty.")

    count = await asyncio.to_thread(collection.count)
    if count == 0:
        return []

    query_embedding = await asyncio.to_thread(
        get_embedding_model_response,
        query,
    )
    result = await asyncio.to_thread(
        collection.query,
        query_embeddings=[query_embedding],
        n_results=min(5, count),
        include=["documents", "metadatas", "distances"],
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {
            "id": chunk_id,
            "text": document,
            **metadata,
            "distance": distance,
        }
        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
        )
    ]
    