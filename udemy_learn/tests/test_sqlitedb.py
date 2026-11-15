import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from udemy_learn.sql_persistence import sqllitedb


class SQLiteDatabaseTests(unittest.TestCase):
    def test_initialize_db_creates_schema_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "nested" / "agent_memory.db"
            with patch.object(sqllitedb, "DB_PATH", database_path):
                sqllitedb.initialize_db()
                first_connection = sqllitedb.get_db_connection()
                try:
                    first_columns = {
                        row[1]
                        for row in first_connection.execute(
                            "PRAGMA table_info(conversations)"
                        )
                    }
                finally:
                    first_connection.close()

                sqllitedb.initialize_db()
                second_connection = sqllitedb.get_db_connection()
                try:
                    second_columns = {
                        row[1]
                        for row in second_connection.execute(
                            "PRAGMA table_info(conversations)"
                        )
                    }
                finally:
                    second_connection.close()

            expected_columns = {
                "id",
                "user_request",
                "assistant_response",
                "embedding",
                "created_at",
            }
            self.assertEqual(first_columns, expected_columns)
            self.assertEqual(second_columns, expected_columns)
            self.assertTrue(database_path.is_file())

    def test_insert_and_fetch_conversations_close_connections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "agent_memory.db"
            with patch.object(sqllitedb, "DB_PATH", database_path):
                sqllitedb.initialize_db()

                opened_connections: list[sqlite3.Connection] = []
                original_get_connection = sqllitedb.get_db_connection

                def track_connection() -> sqlite3.Connection:
                    connection = original_get_connection()
                    opened_connections.append(connection)
                    return connection

                with patch.object(
                    sqllitedb,
                    "get_db_connection",
                    side_effect=track_connection,
                ):
                    sqllitedb.insert_conversation_db(
                        "What is SQLite?",
                        "SQLite is an embedded database.",
                        "[0.1, 0.2]",
                    )
                    sqllitedb.insert_conversation_db(
                        "Where is it stored?",
                        "In a local database file.",
                        "[0.3, 0.4]",
                    )
                    records = sqllitedb.fetch_conversations_db()

                self.assertEqual(len(opened_connections), 3)
                for connection in opened_connections:
                    with self.assertRaises(sqlite3.ProgrammingError):
                        connection.execute("SELECT 1")

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["user_request"], "What is SQLite?")
            self.assertEqual(
                records[1]["assistant_response"],
                "In a local database file.",
            )

    def test_picks_only_similar_conversation_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "agent_memory.db"
            with patch.object(sqllitedb, "DB_PATH", database_path):
                sqllitedb.initialize_db()
                sqllitedb.insert_conversation_db(
                    "Related request",
                    "Related response",
                    json.dumps([1.0, 0.0]),
                )
                sqllitedb.insert_conversation_db(
                    "Unrelated request",
                    "Unrelated response",
                    json.dumps([0.0, 1.0]),
                )

                with patch.object(
                    sqllitedb,
                    "get_embedding_model_response",
                    return_value=[1.0, 0.0],
                ):
                    records = sqllitedb.pick_high_similar_records(
                        "A related request"
                    )

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["user_request"], "Related request")
            self.assertEqual(records[0]["assistant_response"], "Related response")
            self.assertGreaterEqual(records[0]["score"], 0.7)

    def test_empty_memory_does_not_request_a_query_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "agent_memory.db"
            with patch.object(sqllitedb, "DB_PATH", database_path):
                sqllitedb.initialize_db()
                with patch.object(
                    sqllitedb,
                    "get_embedding_model_response",
                    side_effect=AssertionError("No stored embedding should be queried."),
                ) as get_embedding:
                    records = sqllitedb.pick_high_similar_records(
                        "A new request"
                    )

        self.assertEqual(records, [])
        get_embedding.assert_not_called()


class SaveConversationTests(unittest.IsolatedAsyncioTestCase):
    async def test_creates_embedding_and_inserts_without_blocking_event_loop(self) -> None:
        with (
            patch.object(
                sqllitedb,
                "create_embedding_userreq_res",
                return_value=[0.1, 0.2],
            ) as create_embedding,
            patch.object(sqllitedb, "insert_conversation_db") as insert_record,
        ):
            await sqllitedb.save_new_responses("Question", "Answer")

        create_embedding.assert_called_once_with("Question", "Answer")
        insert_record.assert_called_once_with(
            "Question",
            "Answer",
            json.dumps([0.1, 0.2]),
        )


if __name__ == "__main__":
    unittest.main()
