# Gemini Assistant and RAG Learning Project

This project is intentionally organized around the problems encountered while
building the assistant. Each stage demonstrates one approach and its
limitations before moving toward the next one.

## Run the assistant

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m udemy_learn
```

Set `GEMINI_API_KEY` in the environment or in a `.env` file at the repository
root. `API_KEY` is also accepted for compatibility. The model defaults to
`gemini-3.5-flash-lite`; set `MODEL` to use a different Gemini model.

Install the application dependencies in the selected Python environment if
they are not installed yet:

```powershell
.\.venv\Scripts\python.exe -m pip install google-genai python-dotenv
```

## Current learning stages

1. **Conversation history** — the conversation loop stores user and assistant
   turns and sends the full history on each Gemini request.
2. **Manual local tool routing** — `router/tool_router.py` uses Python keyword
   checks to select a time, dice, password, or summarize tool. This is
   intentionally not Gemini function calling yet. Basic tool results are
   displayed locally; a summarize request loads the selected file and sends
   its full contents to Gemini with a summarize instruction.
3. **Exact-word document search** — `tools/files.py` contains
   `build_retriever_rag`, a learning prototype that scans text files and returns
   the first file containing the query as a case-insensitive substring. It
   cannot find related wording by meaning and does not compare all matching
   documents for relevance. This prototype is separate from the current chat
   route.
4. **RAG building blocks** — `rag/embeddings.py` requests vectors from the
   local Ollama embedding model; `rag/retrieval.py` calculates cosine similarity
   and ranks document vectors. These are currently building blocks, not yet
   connected to file loading or the conversation loop. The next step is to
   embed document chunks and the query, retrieve the best-matching text, and
   pass that text to Gemini.

## Module responsibilities

- `__main__.py` is the package entry point.
- `conversation.py` owns terminal input/output, conversation history, and the
  current manually selected tool flow.
- `assistant.py` adapts the app's role/content history to Gemini's request
  format.
- `config/config.py` loads environment-based settings when the app starts.
- `tools/basics.py` and `tools/files.py` contain local tools.
- `rag/` contains the separate embedding and similarity-search experiments.

The module boundaries demonstrate separation of concerns: terminal flow,
provider communication, configuration, local tools, and retrieval can be
understood and tested independently.

## Tests

Run all tests from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\udemy_learn\tests -v
```

The embedding integration test requires Ollama to be running with the
`nomic-embed-text` model pulled:

```powershell
ollama pull nomic-embed-text
```
