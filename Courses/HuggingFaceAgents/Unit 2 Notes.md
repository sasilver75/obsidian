https://huggingface.co/learn/agents-course/en/unit2/introduction

# Introduction to Agentic Frameworks
- ==An agentic framework isn't always needed when building an application around LLMs!==
- Sometimes, predefined ==workflows== are sufficient to fulfill user requests!
- If the approach to build an agent is simple, like a ==chain== of prompts, then using plain code may be enough!
	- Then the developer will have **FULL CONTORL OF THE SYSTEM, WITHOUT ABSTRACTIONS!**

Some of these abstractions are helpful:
- An ==LLM engine== that powers the system.
- A list of ==tools== the agent can access.
- A ==parser for extracting tool calls== from LLM output.
- A ==system prompt== synced with the parser.
- A ==memory system==.
- ==Error logging== and ==retry mechanisms== to control LLM mistakes.
	- We'll see how these get resolved in various frameworks.



# Unit 2.1: smolagents
- ==CodeAgents== are the primary type of agent in `smoleagents`.
	- ==Instead of generating JSON or text, these agents produce Python code to perform actions!==
	- This module explores their purpose, functionality, and how they work, along with hands-on examples to showcase their capabilities.
- ==ToolCallingAgents== are the second type of agent supported by smolagents
	- These agents rely on JSON/text blobs that the system must parse and interpret to execute actions.
- ==Tools== are functions that an LLM can use within an agentic system, and they act as the essential buildling blocks for agent behavior. 
	- In Smolagent, we create tools using the `Tool` class or the `@tool` decorator.. 
	- You'll also learn about the default toolbox, how to share tools, and how to load community tools.
- ==Retrieval Agents== allow models to access knowledge bases, making it possible to search/synthesize/retrieve information from multiple sources.
	- They leverage vector stores for efficient retrieval and implement RAG patterns.
	- This module explores implementation strategies, including fallback mechanisms for robust information retrieval.
- ==Multi-Agent Systems==: By combining agents with different capabilities, you can create more sophisticated solutions. How do we design, implement, and manage multi-agent systems?
- ==Vision Agents== and ==Browser Agents==
	- Vision agents extend traditional agent capabilities by incorporating Vision-Language Models (VLMs), enabling them to process and interpret visual information.
	- We use vision agents to build a browser agent that can browse the web and extract information from it.

Why use `smolagents`
- ==Simplicity==: Minimal code complexity and abstractions, makes framework easy to understand/adopt/extend.
- ==Flexible LLM Support==: Works with any LLM through integration with HF
- ==Code-First Approach==: First-class support for Code Agents, removing need for parsing and simplifying tool calling
- ==HF Hub Integration==: Allows the use of Gradio Spaces as tools, integrates well with Hub

With these advantages in mind, when should we use smolagents?
- When you need a **LIGHTWEIGHT AND MINIMAL SOLUTION**
- When you want to **EXPERIMENT QUICKLY WITHOUT COMPLEX CONFIGURATIONS**
- When your application logic is straightforward.

Code vs JSON Actions
- Unlike other frameworks, `smolagents` focuses on tool calls in code -- this simplifies the execution process, because there's no need to parse the JSON in order to *build* code that calls the tools -- the output can be executed directly.

Agents in `smolagents` operate as ==Multi-Step Agentes==
- Each `MultiStepAgent` performs:
	- One thought
	- One tool call and execution

In addition to using CodeAgent as the primary type of agent, smolagents also supports ToolCallingAgent, which writes tool calls in JSON.

> In `smolagents`, tools are defined using `@tool` decorator wrapping a Python function or the `Tool` class.


Model integration with `smolagent`
- Supports flexible integration, allowing you to use any model that meets certain criteria.
- Several predefined classes to simplify model connections:
	- ==TransformerModel==: Implements a local `transformers` pipeline
	- ==HfApiModel==: Supports serverless inference calls through HF infra, or a growing number of third parties
	- ==LiteLLMModel==: Leverages LiteLLM for lightweight model interactions
	- ==OpenAIServerModel==: Connects to any service that offers an OpenAI API interface.
	- ==AzureOpenAIServerModel==: Supports integration with any Azure OpenAI deployment.


![[Pasted image 20250418230847.png]]
The streamlined approach of CodeAgents, which generate Python tool calls to perform actions (efficient, expressive, accurate)
- The streamlined approach reduces the number of required actions, simplifying complex operations, and enables reuse of existing code functions.
- smolagents has a lightweight framework for building code agents, implemented in approximately 1,000 lines of code.


#### Why Code Agents?
- Writing actions in code rather than JSON offers several key advantages:
	- ==Composability==: Easily combine and reuse actions.
	- ==Object Management==: Work directly with complex structures like images.
	- ==Generality==: Express any computationally possible tasks
	- ==Natural for LLMS==: LLMs are trained on high-quality code

![[Pasted image 20250418231138.png]]

CodeAgent.run(): Follows the [[ReAct]] framework from Unit 1...
- The main abstraction in `smolagents` is `MultiStepAgent`, which serves as the core building block.
- CodeAgent is a special instance of `MultiStepAgent`.

A CodeAgent performs actions through a cycle of steps, with existing variables and knowledge being incorporated into the agent's context, which is kept in an ==execution log==
1. System prompt is stored in a `SystemPromptStep`, and the user query is logged in a `TaskStep`
2. Then, the following while loop is executed
	1. Method `agent.write_memory_to_messages()` writes the agent's logs into a series of LLM-readable chat messages.
	2. These messages are sent to a Model, which generates a completion.
	3. The completion is parsed to extract the action, which, in our case, should be a code snippet (it's a CodeAgent!).
	4. The action is executed
	5. The results are logged into memory in an `ActionStep`.


Let's see some examlples!




# Unit 2.2: LlamaIndex
- 

# Unit 2.3: LangGraph
- 