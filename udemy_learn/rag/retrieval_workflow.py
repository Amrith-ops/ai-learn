from pathlib import Path

from udemy_learn.rag.retrieval_tools import (
    rank_documents_by_similarity,
    load_documents_and_create_embeddings,
    user_query_embedding_create
)

"""
Create a workflow by tying all functions from retrieval.py together.
    1. Load all documents from the data folder and create embeddings for them.
    2. Create an embedding for the user query.
    3. Rank the documents based on their similarity to the user query.
    4. Return the ranked list of documents with their similarity scores.
    5. Pick and return the most relevant document content based on the highest similarity score.
"""

def run_retrieval_workflow(data_folder: str, user_query: str) :
    # Step 1: Convert user query into an embedding vector
    query_embedding = user_query_embedding_create(user_query)
    document_embeddings = load_documents_and_create_embeddings(data_folder)
    if not document_embeddings:
        return None

    ranked_documents = rank_documents_by_similarity(query_embedding, document_embeddings)
    file_name, similarity = ranked_documents[0]
    return file_name, document_embeddings[file_name]["content"], similarity