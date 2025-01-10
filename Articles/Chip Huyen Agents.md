January 7, 2025
[[Chip Huyen]]

-----

The section starts with an overview of agents and then continue with two aspects that determine the capabilities of an agent: 
1. Tools
2. Planning

# Agent Overview
The term ==agent== has been used in many different engineering contexts, including but not limited to a software agent, intelligent agent, user agent, conversational agent, and reinforcement learning agent. So what is an agent?

==Agent==: Anything that an be viewed as perceiving its environment through sensors and acting upon that environment through actuators.
- The environment is defined by its use case.
- The set of actions an AI agent can perform is augmented by the tools it has access to.

There's a strong dependency between an agent's environment and its set of tools. The environment determines what tools an agent can potentially use. For example, if the environment is a chess game, the only possible actions for an agent are the valid chess moves.

![[Pasted image 20250109143955.png|600]]

An AI agent is meant to accomplish tasks typically provided by the users. In an AI agent, AI is the brain that processes the task, plans a sequence of actions to achieve this task, and determines whether the task has been accomplished.

Given the query: "Project the sales revenue for Fruity Fedora over the next three months"
1. Reason about how to accomplish this task -- it might decide that it needs the sales numbers from the last five years
2. Invoke SQL query generation to generate the query to get sales numbers from the last 5 years.
3. Invoke SQL query execution to execute this query.
4. Reason about the tool outputs (outputs from the SQL query execution) and how they help with sales prediction.
	- It might decide that these numbers are insufficient to make a reliable projection, perhaps because of missing values. It then decides that it also needs information about past marketing campaigns.
5. Invoke SQL query generation to generate the queries for past marketing campaigns.
6. Invoke SQL query execution
7. Reasoning that this new information is sufficient to help predict future sales. It then generates a projection.
8. Reason that the task has been successfully completed.

Compared to non-agent use cases, agents typically require more powerful models for two reasons:
1. ==Compound mistakes==: Multiple steps means that 95% accuracy per step over 10 steps means that overall accuracy.
2. ==Higher Stakes==: Agents with tools can perform more impactful tasks with real consequences.

## Tools
- A system doesn't need access to external tools to be an agent. Without external tools, the agent's capabilities would be limited. By itself, a model can typically perform one action -- an LLM can generate text and an image generator can generate images.
- The set of tools an agent has access to is its ==tool inventory==.
- More tools give an agent more capabilities, however the more tools there are 




