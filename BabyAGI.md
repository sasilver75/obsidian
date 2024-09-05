April 2023 (~1 month after GPT-4 and AutoGPT)
Github Repository: [Link](https://github.com/yoheinakajima/babyagi)

Note: Similar to [[AutoGPT]], in that you set a goal and it autonomously iterates to try to accomplish the objective.
Open-Source autonomous Agent project

An example of an AI-powered task management system that uses OpenAI and vector databases like Chroma/Weaviate to create/prioritize/execute tasks.

Main idea: It creates tasks based on the result of previous tasks and a predefined objective. Script uses LM to create new tasks base on objective, and Chroma/Weaviate to store/retrieve task results for context.

1. Pull the first task from the task list
2. Send the task to the execution agent, which uses OpenAI's API to complete the task based on context.
3. Enrich the result and store it in Chroma/Weaviate.
4. Create new tasks and reprioritize the task list based on the objective and result of the previous task.