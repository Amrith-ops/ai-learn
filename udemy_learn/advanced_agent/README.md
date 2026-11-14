In this advanced agent module i improvise on what i have built on agent_loop
In the agent loop the planner agent is presnt to take the user input and tools available
to decide which action/tool need to be invoked next based on the previous action and 
observation recorded for that action.
In this module, we now look at the task decomposition i.e when a request arrives from
user consider it as a broader goal and break the goal into small tasks, and send pending task to planner agent and ask planner agent to pick which action needs to be executed next
In the planner agent prompt now = user request + tasks list + tool list available +
previous actions took + observations list (which contains what is current task , actions history and observation or result history) + result list

So the flow now looks like:
User request --> Task Decomposer agent --> parse the tasks and store it in task list
--> Send task list to planner agent --> planner agent tells which task needs to be done
and what is tool for that task --> execute the action/tool recommended for that task
--> store the observation/result returned by the response --> check if next action is
needed --> if needed run the loop --> if tasks finished break out of the loop --> AnswerAgent --> this takes the observations, intermediate results, tasks created, actions executed and prepare the result into user understandable natural language

Now my question, why task needs to be prepared upfront , why do we need to have a task decomposer agent what value it will add , it's just giving english tasks which later turn out to functions/actions so what's the deal here?