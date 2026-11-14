import json
from urllib.request import Request, urlopen


OLLAMA_EMBEDDING_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "nomic-embed-text"


def get_embedding_model_response(input_text: str) -> list[float]:
    """Return one vector from the local Ollama embedding endpoint."""
    payload = {
        "model": EMBEDDING_MODEL,
        "input": [input_text],
    }
    request = Request(
        OLLAMA_EMBEDDING_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError("Ollama returned no embeddings for the input text.")
    return embeddings[0]