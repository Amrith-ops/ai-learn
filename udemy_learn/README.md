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
.\.venv\Scripts\python.exe -m pip install google-genai python-dotenv "mcp<2"
```

## Run the MCP tool server

The MCP server exposes the existing time, dice, and password tools over
Streamable HTTP. Start it from the repository root:

```powershell
.\.venv\Scripts\python.exe -m udemy_learn.mcp_server.server
```

The local server listens at `http://127.0.0.1:8000/mcp`. Configure an
MCP-compatible agent host with that URL to discover and call the tools. This
default local endpoint is for development; a server made available beyond the
local machine should be deployed behind appropriate authentication and
network-access controls.

## Use the MCP server from the agent

Keep the server running in one terminal. The agent-side workflow in
`mcp_client/mcp_workflow.py` connects to its HTTP URL, discovers tools and their
input schemas, asks Gemini to choose a tool, then calls the selected tool over
the same MCP session. Call `run_mcp_workflow()` from an async application to
use this flow. In the terminal app, the MCP result is sent back to Gemini with
the original request so it can produce a natural-language answer. If no tool
is selected, Gemini answers the original request directly.

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
4. **Semantic retrieval and RAG handoff** — for messages that do not match a
   local tool or explicit summarize command, `rag/retrieval.py` runs the
   workflow. `rag/retrieval_tools.py` embeds each text file and the query with
   Ollama, then ranks documents by cosine similarity. The top document's text
   and the original question are passed to Gemini as context. Summarize
   commands still use the earlier direct file-reading path. This prototype
   embeds whole files on every query; chunking and caching can be learned next.
5. **Request classification and agent loop** — `request_classifier.py` asks
   Gemini whether a request needs a single tool step or multiple steps.
   `conversation.py` routes single-step requests through the MCP workflow and
   multi-step requests through `agent_loop/agent_loop_workflow.py`, which
   tracks actions and observations until the planner returns a final answer.

## Module responsibilities

- `__main__.py` is the package entry point.
- `conversation.py` owns terminal input/output, conversation history, and the
  current manually selected tool flow.
- `assistant.py` adapts the app's role/content history to Gemini's request
  format.
- `config/config.py` loads environment-based settings when the app starts.
- `tools/basics.py` and `tools/files.py` contain local tools.
- `mcp_server/server.py` exposes selected utility tools through the MCP
  Streamable HTTP transport.
- `mcp_client/client.py` manages the MCP connection, tool discovery, and remote
  calls; `mcp_client/mcp_workflow.py` asks Gemini to choose a tool and supplies
  its result to the conversation.
- `request_classifier.py` classifies requests as single-step or multi-step.
- `agent_loop/` contains the planner, MCP tool execution, and state management
  for multi-step requests.
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
