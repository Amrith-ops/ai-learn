What does the workflow looks like:
1. User sends the request
2. Call MCP client using python program to discover which tools are available
3. Make a prompt which has list of tools and it's description
4. Create a agent state with default values
5. Send the prompt to planner agent and ask it to return only next one action on the basis of past actions and past observations
6. Fetch the response and store the response in action field in the agent state
7. Call the tool through mcp client and get tool response , store this tool response in 
the observations
8. Increment the current step