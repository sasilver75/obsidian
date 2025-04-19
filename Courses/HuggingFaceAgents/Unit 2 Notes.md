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


Let's see some examples!

.. in Code...

## Tool-Calling Agents in Smolagents
- A tool is a function that an LLM can use in an agentic system.
- So it cannot be only a function. It should be a class.
- So at core, the tool is a class that wraps a function with metadata that helps the LLM understand how to use it.

```python
from smolagents import Tool

class HFModelDownloadsTool(Tool):
    name = "model_download_counter"
    description = """
    This is a tool that returns the most downloaded model of a given task on the Hugging Face Hub.
    It returns the name of the checkpoint."""
    inputs = {
        "task": {
            "type": "string",
            "description": "the task category (such as text-classification, depth-estimation, etc)",
        }
    }
    output_type = "string"

    def forward(self, task: str):
        from huggingface_hub import list_models

        model = next(iter(list_models(filter=task, sort="downloads", direction=-1)))
        return model.id


```

Above, we're creating a custom tool that Subclasses `Tool` to inherit useful methods.
This child class also defines:
- An attribute ==name==: The name of the tool.
- An attribute ==description==: Used to populate agent's system prompt.
- An attribute ==inputs==, a dict mapping parameters to dicts with keys "type" and "description"
- An attribute ==output_type== attribute, which specifies the output type.
- A ==forward==  method, which contains the inference code to be executed.

That's all you need to do to build a tool to use it with an agent in smolagents!
- Note that `tool()` decorator is the recommended way to define simple tools, but sometimes you need more than this.

We can share our tools to the Hub too!

```python
model_downloads_tool = HFModelDownloadsTool() # An instance of our custom tool class we created above

model_downloads_tool.push_to_hub("{your_username}/hf-model-downloads", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")
```

For the push to work, you need:
- All methods are self-contained, using variables that come from their args.
- As per the above point, ==all imports should be defined directly within the tool's function==, else you will get an error when trying to call save() or push_to_hub() with a custom tool.
- If you subclass the __init\__ method, you can give it no other argument than self.
	- This is because arguments set during a specific tool instance's initialization are hard to track, which prevents from sharing them properly to the hub....blah blah.

Then you can load the tool with `load_tool()` or create it from `from_hub()` and pass it to the `tool` parmeter in our agent.

```python
from smolagents import load_tool, CodeAgent

model_download_tool = load_tool(
    "{your_username}/hf-model-downloads",
    trust_remote_code=True
)
```


We can also directly import a Gradio Space from the hub as a tool, using the `Tool.from_space()` method!
For instance, let's import the Flux.1-dev Space from the Hub and use it to generate an image.

```python
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

image_generation_tool("A sunny beach")
```

Then we can use this tool like any other tool!

```python
from smolagents import CodeAgent, InferenceClientModel

# A class to interact with Hugging Face's Inference Providers for language model interaction
# Note: HfApiModel has been renamed to InferenceClientModel to more closely follow the name of the underlying Inference library
model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")
agent = CodeAgent(tools=[image_generation_tool], model=model)

agent.run(
    "Improve this prompt, then generate an image of it.", additional_args={'user_prompt': 'A rabbit wearing a space suit'}
)
```


### Using LangChain tools in smolagent
- We love Langchain and think it has a a compelling suite of tools!
- We can import a tool form LangChain using the `from_langchain()` method.

Below, let's recreate our search result using a Langchain web search tool.
This tool will need `pip install langchain google-search-results -q` to work properly.
```python
from langchain.agents import load_tools

search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

agent = CodeAgent(tools=[search_tool], model=model)

agent.run("How many more blocks are in BERT base encoder?")
```


We can technically add a tool to the agent.tools dictionary...
```python
from smolagents import InferenceClientModel

model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-32B-Instruct")

agent = CodeAgent(tools=[], model=model, add_base_tools=True)
agent.tools[model_download_tool.name] = model_download_tool
```

And then leverage the new tool:
```python
agent.run(
    "Can you give me the name of the model that has the most downloads in the 'text-to-video' task on the Hugging Face Hub but reverse the letters?"
)
```

Use a collection of tools!
- We can leverage ==tool collections== by using ==ToolCollection==
	- These are a collection of separate tools that all become available in an agent's toolbox.


Tool Collection from a collection in the Hub:
```python
from smolagents import ToolCollection, CodeAgent

image_tool_collection = ToolCollection.from_hub(
    collection_slug="huggingface-tools/diffusion-tools-6630bb19a942c2306a2cdb6f",
    token="<YOUR_HUGGINGFACEHUB_API_TOKEN>"
)
# See here that we can spread the iterable image_tool_collection.tools into the toolas argument of our CodeAgent!
agent = CodeAgent(tools=[*image_tool_collection.tools], model=model, add_base_tools=True)

agent.run("Please draw me a picture of rivers and lakes.")
```


We can also load tools from [[Model Context Protocol]] servers available on glama.ai or smithery.ai!
- Note that MCP servers come with security risks!
	- Malicious servers may execute harmful code on your machine
	- Stdio-based MCP servers will ALWAYS execute code on your machine (that's their intended functionality)
	- SSE-based MCP servers: While the remote MCP servers won't be able to exec code on your machine, still proceed with catution.

We load as follows:
```python
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters

# See that we pass the server parametesr to this StdioServerParameters, for a stdio-based MCP server
server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    agent.run("Please find a remedy for hangover.")
```

And for an [[Server-Sent Event|SSE]]-based MCP server, we can simply pass a dict with parameters to mcp.client.sse.sse_client

```python
from smolagents import ToolCollection, CodeAgent

with ToolCollection.from_mcp({"url": "http://127.0.0.1:8000/sse"}, trust_remote_code=True) as tool_collection:
    agent = CodeAgent(tools=[*tool_collection.tools], add_base_tools=True)
    agent.run("Please find a remedy for hangover.")

### [](https://huggingface.co/docs/smolagents/tutorials/tools#use-mcp-tools-with-mcpclient-directly)
```

We can also use MCP tools with MCPClient directly, which gives us more control over the connection and tool management.

For stdio-based MCP server:
```python
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
import os

server_parameters = StdioServerParameters(
    command="uvx",  # Using uvx ensures dependencies are available
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

with MCPClient(server_parameters) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("Please find the latest research on COVID-19 treatment.")
```


For SSE-based MCP serverS:
```python
from smolagents import MCPClient, CodeAgent

with MCPClient({"url": "http://127.0.0.1:8000/sse"}) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("Please find a remedy for hangover.")
```


We can also connecto MULTIPLE MCP servers by passing a LIST of server parameters:
```python
from smolagents import MCPClient, CodeAgent
from mcp import StdioServerParameters
import os

server_params1 = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

server_params2 = {"url": "http://127.0.0.1:8000/sse"}

with MCPClient([server_params1, server_params2]) as tools:
    agent = CodeAgent(tools=tools, model=model, add_base_tools=True)
    agent.run("Please analyze the latest research and suggest remedies for headaches.")
```

### Understanding Secure Code Execution
- Multiple papers have shown that ==letting the LLM write its own actions (the tool calls) in code is much better than teh standord format for tool calling, which is across the industry different shades of "writing actions as a JSON of tool names and arguments to use.==


# Unit 2.2: LlamaIndex
- 

# Unit 2.3: LangGraph
- 