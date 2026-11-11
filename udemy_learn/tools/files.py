from pathlib import Path


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"


def read_text_file(file_name: str) -> str:
    file_path = Path(file_name)
    if not file_path.is_absolute():
        if file_path.parts and file_path.parts[0].casefold() == "data":
            file_path = PROJECT_DIRECTORY / file_path
        else:
            file_path = DATA_DIRECTORY / file_path

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"Error: File not found {file_path}"


def parse_summarize_file_name(user_input: str) -> str | None:
    parts = user_input.strip().split(maxsplit=1)
    if len(parts) != 2 or parts[0].casefold() not in {"summarize", "summary"}:
        return None

    file_name = parts[1].strip().strip('"')
    return file_name or None


def get_summarize_file_content(user_input: str) -> str:
    file_name = parse_summarize_file_name(user_input)
    if file_name is None:
        return (
            'Error: file name is missing or invalid. Please provide a valid file '
            'name after "summarize" or "summary".'
        )

    return f"Summarize the following content:\n{read_text_file(file_name)}"


def build_retriever_rag(query_from_user: str) -> str:
    """Demonstrate exact-word retrieval before the semantic RAG milestone."""
    for file in sorted(DATA_DIRECTORY.glob("*.txt")):
        print(f"File: {file.name}")
        content = read_text_file(str(file))
        if query_from_user.casefold() in content.casefold():
            return f"Found in {file.name}:\n{content}"
    return "No relevant content found."
