import asyncio
from udemy_learn.utils.embeddings import get_embedding_model_response

def chunk_text_from_documents(
        text: str,
        source: str,
        chunk_size = 200,
        overlap = 50
):
    cursor_start = 0
    cursor_end = chunk_size
    chunk_list_store = []
    chunk_id = 0
    while (cursor_start <= len(text)):
        text_chunk = " ".join(text[cursor_start: cursor_end])
        chunk_list_store.append(text_chunk)
        cursor_start = cursor_end - overlap
        cursor_end = cursor_start + chunk_size
        chunk_id += 1
        chunk_list_store.append(
            {
            "text": text_chunk,
            "source": source,
            "chunk_id": chunk_id
            }
        )
    return chunk_list_store

async def add_embedding_chunk_dict(chunk_list_store: list[dict[str, any]]):
    for chunk in chunk_list_store:
        try:
            embedding = await asyncio.to_thread(
                get_embedding_model_response,
                chunk["text"]
            )
            chunk["embedding"] = embedding
        except RuntimeError as err:
            chunk["embedding"] = None
    return chunk_list_store
