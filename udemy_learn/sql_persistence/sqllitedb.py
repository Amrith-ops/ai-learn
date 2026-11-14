import sqlite3
import json
import asyncio
from pathlib import Path
from udemy_learn.sql_persistence.embeddings import get_embedding_model_response
from udemy_learn.rag.retrieval_tools import calculate_cosine_similarity

DB_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sqlitedb"
    / "agent_memory.db"
)


def initialize_db() -> None:
    """Create the database schema if needed."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_request TEXT,
                    assistant_response TEXT,
                    embedding TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        connection.close()


def get_db_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def create_embedding_userreq_res(user_request:str, assistant_response:str):
    total_text = f"""
                User request : {user_request}\n
                Assistant response : {assistant_response}
                """
    total_text_embedding = get_embedding_model_response(total_text)
    return total_text_embedding

async def save_new_responses( user_request: str,
    assistant_response: str):
    total_text_embedding = await asyncio.to_thread(create_embedding_userreq_res(user_request, assistant_response))
    insert_conversation_db(user_request,assistant_response, json.dumps(total_text_embedding))

def insert_conversation_db(
    user_request: str,
    assistant_response: str,
    embedding_response: str,
) -> None:
    connection = get_db_connection()
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    user_request,
                    assistant_response,
                    embedding
                ) VALUES (?, ?, ?)
                """,
                (user_request, assistant_response, embedding_response),
            )
    finally:
        connection.close()


def fetch_conversations_db() -> list[sqlite3.Row]:
    """Return all saved conversations in insertion order."""
    connection = get_db_connection()
    connection.row_factory = sqlite3.Row
    try:
        total_results = connection.execute(
            """
            SELECT id, user_request, assistant_response, embedding, created_at
            FROM conversations
            ORDER BY id
            """
        ).fetchall()
    finally:
        connection.close()
    return total_results

def pick_high_similar_records(
    user_input: str,
    threshold: float = 0.7,
) -> list[dict[str, str | float | None]]:
    """Return up to three prior conversations relevant to the user request."""
    total_results = fetch_conversations_db()
    records_with_embeddings = [
        record
        for record in total_results
        if record["embedding"] is not None
        and record["user_request"] is not None
        and record["assistant_response"] is not None
    ]
    if not records_with_embeddings:
        return []

    query_embedding = get_embedding_model_response(user_input)
    matched_documents: list[dict[str, str | float | None]] = []
    for record in records_with_embeddings:
        embedding = record["embedding"]
        user_request = record["user_request"]
        assistant_response = record["assistant_response"]
        embedding_db = json.loads(embedding)
        cosine_score = calculate_cosine_similarity(query_embedding, embedding_db)
        if cosine_score >= threshold:
            matched_documents.append(
                {
                    "user_request": user_request,
                    "assistant_response": assistant_response,
                    "score": cosine_score,
                    "created_at": record["created_at"],
                }
            )
    return sorted(matched_documents, key=lambda record: record["score"], reverse=True)[:3]
       
       
     





