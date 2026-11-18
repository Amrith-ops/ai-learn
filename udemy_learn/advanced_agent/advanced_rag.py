"""
Workflow looks like:
1. open the document
2. chunk the document
3. convert each chunk to embedding and metadata
4. insert the embedding into chromadb
5. fetch the top 5 semantic documents
6. add the semantic score to each document dictionary
7. calculate the keyword score (skip for now)
8. add re-ranking document strategy pick top 3 documents
"""