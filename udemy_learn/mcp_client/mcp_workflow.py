from udemy_learn.mcp_client.client import connect
from udemy_learn.assistant import generate_assistant_reply
from google.genai import Client

"""
First the python code calls the MCP client to call MCP server to get the list of tools available
Then it prepares a prompt for the LLM model to select a tool from the list of tools available
LLM model looks at the prompt and user input and select a tool from the list of tools available and returns the tool name to 
the python code

Then the python code calls the MCP client to call MCP server to execute the tool selected by the LLM model and returns the result to the user

"""

def build_tool_descriptions(tools):
    descriptions = ""
    for tool in tools:
        descriptions += f"""
                        Tool: {tool.name}
                        Description : {tool.description}
                        ----------------------------------------
                        """
    return descriptions

async def run_mcp_workflow(
    client: Client,
    user_input: str,
    model: str,
):
    # uses the MCP client to get the list of tools available from the MCP server
    mcp_client = await connect()
    tools = await mcp_client.list_tools()
    # Build prompt using the fetched tool list, this list of tools in prompt is sent to LLM along with user input
    tool_prompt_description = build_tool_descriptions(tools)
    response = generate_assistant_reply(
        client,
        model, 
        messages= [{"role":"user","content":f"{user_input}"}],
        system_instruction = f"""
                You are an AI planner that selects tool based on the user input.
                You have access to following tools:
                {tool_prompt_description}
                Return the response with exact tool name 
                If there are no tools match then return None
                """
    )


