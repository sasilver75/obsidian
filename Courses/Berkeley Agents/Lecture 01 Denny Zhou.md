https://www.youtube.com/watch?v=QL-FS_Zcmyo&list=PLS01nW3RtgopsNLeM936V4TNSsvvVglLc&index=6&t=314s

===

![[Pasted image 20241017221242.png]]

Instead of just taking text as inputs and text as outputs, we use a LLM as the key brain for reasoning/planning for agents that let agent observe and take actions in the environments, using external tools and databases/knowledge bases to help the agent perform tasks.

Rich capabilities of LLMs make LLM agents flexible

![[Pasted image 20241017221425.png]]

Through the interaction with complex/diverse environments, they can update their memory... and obtain grounding through this interaction data.

Agents can interact with other agents through multiagent interactions and collaboration...

So why are Agents the next frontier?
- Trial and error process
- Leverage of external tools/revtrieval to extend capabilities
- Dynamic agentic flow... can facilitate solving complex tasks through task decomposition, allocation of subtasks to specialized submodules, multiagent collaboration, etc.

![[Pasted image 20241017221802.png]]
Mentions [[SWE-bench]], [[GAIA]], [[WebArena]] as relevant benchmarks.

To better enable agent deployments there are a number of key challenges to address:
1. Improve reasoning and planning capabilities of agents
2. Embodiment and learning from environment feedback -- how can we recover from mistakes for long-horizon tasks? (Multimodal learning, etc.)
3. Multi-agent learning, theory of mind of other agents?
4. Safety and privacy (LMs are susceptible to adversarial attacks)
5. Human-agent interaction, ethics (how to control agent behavior)

Topics in Course:
1. Model core capabilities (Reasoning, Planning, Multimodal)
2. LLM agent frameworks (Workflow design, tool use, RAG, multi-agent)
3. Applications (software dev, workflow automation, multimodal, enterprise)
4. Safety and ethics 







