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

Code is just a better way to express actions than JSON snippets!
- ==Composability==: Could we next JSON actions within eachother, or define JSON actions to re-use later, like we can with Python functions?
- ==Object management==: Can we store the output of an action like generate_image in JSON?
- ==Generality==: Code is built to express anything a computer can do
- ==Representation in LLM training corpus==: LLMs have examples of many quality actions in training corpusses.

![[Pasted image 20250419131825.png]]

Local code execution:
- By default, ==CodeAgent runs LLM-generated code in YOUR ENVIRONMENT!==
	- This is ==INHERENTLY RISKY!==
		- ==Plain LLM errors==: LLM might attempt to execute potentially dangerous code
		- ==Supply chain attack==: An LLM might attempt to use code outside your supply chain (e.g. random PyPi package that you don't control)
		- ==Prompt Injection==: An agent browsing the web could arrive on an a malicious website that contains harmful instructions, thus injecting an attack into the agent's memory.
		- ==Exploitation of publicly-accessible agents==: Attacks crafting adversarial inputs to exploit the agent's execution capabilities.

On the spectrum of agency, ==Code agents give much higher agency to the LLM, which goes hand-in-hand with higher risk!==

We advise you to keep in mind that NO SOLUTION will be 100% safe!
![[Pasted image 20250419132239.png]]


Our local Python executor:
- To add a FIRST layer of security, ==code execution in `smolagents` is not performed by the vanilla Python interpreter==.
	- We rebuilt a more secure `LocalPythonExecutor` from the ground up.
	- To be precise, this interpreter loads the Abstract Syntax Tree (AST) from the code and executes it operation by operation, making sure to always follow certain rules.
		- ==Imports are disallowed until explicitly authorized by users==
		- ==Access to submodules is disabled by default==, and each must be explicitly authorized in the import list as well
			- You can do `numpy.*` to allow both `numpy` and all subpackages, like `numpy.random` or `numpy.a.b`.
		- The total count of elementary operations processed is capped to ==prevent infinite loops== and resource bloating.
		- Any operation that has not been explicitly defined in our customer interpreter will raise an error.

==No local Python sandbox can ever be completely secure.==
- It's still possible for a determined attacked or maliciously fine-tuend LLM to find vulnerabilitis and potentially harm your environment.
	- For example, if you use `Pillow` to process images, the LLM could generate code that creates THOUSANDS of large image files to fill your hard drive!

==The only way to run LLM-generated code with truly robust security isoluation is to use remote execution options like E2B or Docker==!



### Sandbox Approaches for Secure CodeEx

There are two main approaches to sandboxing code execution in smolagent:

1. ==Running individual code snippets in a sandbox==
	1. The rest of the agentic system is in your local environment; Simpler to set up, using executor_type="e2b"/"docker", but it doesn't support multi-agents and still requires passing state data between your environment and the sandbox.
2. ==Running the entire agentic system in a sandobx==
	1. This approach runs the entire agentic system, including the ==agent, model, and tools== within a sandbox environment.
	2. This provides better isolation but requires more manual setup and might require passing sensitive credentials (like API keys) to the sandbox environment.


##### E2B Setup

> E2B: https://e2b.dev/
> "Run AI-Generated Code Securely in your App. E2B is an open-source runtime for executing AI-generated code in secure cloud sandboxes. Made for agentic and AI use cases."

1. Create an E2B account at e2b.dev
2. Install the required packages

`pip install smolagents[e2b]`

There's a simple way to use an E2B Sandbox:
- Adding `executor_type="e2b"` to the agent initialization as follows:

```python
from smolagentes import InferenceClientModel, CodeAgent

agent = CodeAgent(model=InferenceClientModel(), tools=[], executor_type="e2b")

agent.run("Can you give me a 100th Fibonacci number?")
```

This solution sends the agent state to the server at the start of each `agent.run()`. The *models* are called from the local environment, but the ==generated code will be sent to the sandbox for execution, with only the output being returned==.

This is illustrated in the figure below:
![[Pasted image 20250419134203.png|600]]

==This solution doesn't work well for multi-agent setups, since we don't transfer *secrets* to the remote sandbox (this is a good thing).==
- Hence this solution doesn't work (yet) for more complicated multi-agent setups.

To use multi-agent setups in an E2B sandbox, you need to run your agents *completely* from within E2B

\<Example of this>

#### Docker Setup

Installation:
1. Install Docker on your system
2. Install the required packages

`pip install 'smolagents[docker]'`

Similarly to the E2B run:
```python
from smolagents import InferenceClientModel, CodeAgent

agent = CodeAgent(model=InferenceClientModel(), tools=[], executor_type="docker")

agent.run("Can you give me the 100th Fibonacci number?")
```

To run multi-agent systems, we'll need to setup a custom interpreter in a sandbox.
Here's to setup the Dockerfile:

\<example of dockerfile>
\<example of sandbox manager to run code>



### Best Practices for Sandboxes (E2B and Docker)

- Resource management
	- Set memory and CPU limits
	- Implement execution timeouts
	- Monitor resource usage
- Security
	- Run with minimal privileges
	- Disable unnecessary network access
	- Use environment variables for secrets
- Environment
	- Keep dependencies minimal
	- Use fixed package versions
	- If you use base images, update them regularly
- Cleanup
	- Always ensure proper cleanup of resources, especially for Docker containers, to avoid having dangling containers eating up resources.


### Comparing Security Approaches

Approach 1: running just the code snippets in a sandbox:
- +: Easy to setup (executor_type="docker"/"e2b", no need to transfer API keys, better protection for local env
- -: Doesn't support multi-agent, requires transferring state between your environment and the sandbox, limited to specific code execution

Approach 2: Running the entire agentic system in a sandbox
- +: Support multi agents, complete isolation of the entire system, more flexible for complex agent architectures
- -:  Requires more manual setup, may require transferring sensitive API keys to the sandbox, potentially higher-latency due to more complex operations

Choose the approach that best balances your security needs with your application's requirements.
For most applications with simpler agent architectures, Approach 1 provides a good balance of security and ease of use.
For more complex ones, Approach 2, while harder to setup, offers better security.


## Building a Web Browser with Vision Models


```shell
pip install smolagents selenium helium pillow -q
```

Let's setup:
```python
from io import BytesIO
from time import sleep

import helium # Lighter web automation based on Selenium
from dotenv import load_dotenv # For environment variables
from PIL import Image # Python Image Library
from selenium import webdriver # Browser automation library
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from smolagents import CodeAgent, tool # Our lovely agents framework
from smolagents.agents import ActionStep

# Load environment variables
load_dotenv()
```

Now let's create our core browser interaction ==tools== that allow our agent to navigate and interact with web pages:
```python
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    # driver is helium.start_chrome(headless=False, options=chrome_options) instance created earlier
    
    # finds the elements that contain our text
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")

	# We can't return the n'th option if we have n-m options
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]

	# Scroll so that our n'th element is visible
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    
    result += f"Focused on element {nth_result} of {len(elements)}"

	# Seems like result is a string that ... just has this narration of what happened?
    return result

@tool
def go_back() -> None:
    """Goes back to previous page."""
    # driver is helium.start_chrome(headless=False, options=chrome_options) instance created earlier
    driver.back()

@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    This does not work on cookie consent banners.
    """
    
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
```

Earlier, we'd set up our browser with Chrome and configured screenshot capabilities:

```python
# Configure Chrome options
chrome_options = webdriver.ChromeOptions() 
chrome_options.add_argument("--force-device-scale-factor=1")
chrome_options.add_argument("--window-size=1000,1350")
chrome_options.add_argument("--disable-pdf-viewer")
chrome_options.add_argument("--window-position=0,0")

# Initialize the browser
driver = helium.start_chrome(headless=False, options=chrome_options)

# Set up screenshot callback
def save_screenshot(memory_step: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0) # Let JavaScript animations happen before taking the screenshot <<< This is important
    driver = helium.get_driver()  # I wonder if this the same driven object as above?
    current_step = memory_step.step_number
    if driver is not None:
        for previous_memory_step in agent.memory.steps:  # Remove previous screenshots for lean processing
            if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= current_step - 2:
                previous_memory_step.observations_images = None
        png_bytes = driver.get_screenshot_as_png()  # Take screenshot
        image = Image.open(BytesIO(png_bytes))  # Turn screenshot into a PIPL image
        print(f"Captured a browser screenshot: {image.size} pixels")
        memory_step.observations_images = [image.copy()]  # Create a copy to ensure it persists

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    memory_step.observations = (
        url_info if memory_step.observations is None else memory_step.observations + "\n" + url_info
    )
```


Now let's actually create our Web Automation Agent!
```python
from smolagents import InferenceClientModel

# Initialize the model
model_id = "meta-llama/Llama-3.3-70B-Instruct"  # You can change this to your preferred model
model = InferenceClientModel(model_id=model_id) # Uses HF serverless inference, I think.

# Create the agent
agent = CodeAgent(
    tools=[go_back, close_popups, search_item_ctrl_f], # Our tools we defined
    model=model,
    additional_authorized_imports=["helium"], # <-- Our tools don't require the import, but our step_callback does?
    step_callbacks=[save_screenshot], # <-- See this is where we have this save_screenshot callback we wrote (not a tool)
    max_steps=20, # Limits the number of steps the agent can take
    verbosity_level=2, # Probabl somehow configures the verbosity of the output.
)

# Import helium for the agent
agent.python_executor("from helium import *", agent.state) # Not sure why we have to do this, and why we have to pass state.
# Oh, so it's just so that the agent doesn't have to remember to do this? I guess that's smart. See the prompt start below.
```

This agent needs instructions on how to use Helium for web automation:

```python
helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
We've already ran "from helium import *"
Then you can go to pages!
Code:
```py
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1200) # This will scroll one viewport down
```<end_code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>
"""
```

Now we can run our agent with our task!
```python
search_request = """
Please navigate to https://en.wikipedia.org/wiki/Chicago and give me a sentence containing the word "1992" that mentions a construction accident.
"""

# Wait, so the helium_instructions aren't passed as the system prompt, but just appended to the user prompt?
agent_output = agent.run(search_request + helium_instructions)
print("Final output:")
print(agent_output)
```


We can run different tasks by modifying the request:
```python
github_request = """
I'm trying to find how hard I have to work to get a repo in github.com/trending.
Can you navigate to the profile for the top author of the top trending repo, and give me their total number of commits over the last year?
"""

agent_output = agent.run(github_request + helium_instructions)
print("Final output:")
print(agent_output)
```


### Agent-Related objects

Agents
- Our agents inherit from `MultiStepagent`, meaning they can act in multiple steps, each step consisting of:
	- ==A thought==
	- ==One tool call and execution==
- We provide two types of agentS:
	- CodeAgent (default: Writes tool calls in Python code)
	- ToolCallingAgnet: Writes its tool calls in JsON

## Writing Actions as Code Snippets or JSON Blobs

==ToolCalling Agents== are the second type of agent available in smolai
- Use the built-in tool-calling capabilities of LLM providers to generate tool calls like JSON structures

If Alfred wants to search for catering services and party ideas, a CodeAgent would generate and run python code like:

```python
for query in [
    "Best catering services in Gotham City", 
    "Party theme ideas for superheroes"
]:
    print(web_search(f"Search for: {query}"))
```

Whereas a ToolCalling Agent would instead create a JSON structure:
```python
[
    {"name": "web_search", "arguments": "Best catering services in Gotham City"},
    {"name": "web_search", "arguments": "Party theme ideas for superheroes"}
]
```
And then this JSON blob is used to execute the tool calls (rather than the Code Block being executed, in the CodeAgent case)

The difference is:
- How they structure their actions (JSON objects that specify tool names and arguments, instead of executable code)
- The system then parses these instructions to execute the appropriate tools.

Let's make a web-browsing tool-calling agent:
```python
from smolagents import ToolCallingAgent, DuckDuckGoSearchTool, HfApiModel

agent = ToolCallingAgent(tools=[DuckDuckGoSearchTool()], model=HfApiModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
```
Above: So wee that we initialize the agent same way, just use ToolCallingAgent instead of CodeAgent.

When we examine the agent's trace, instead of seeing Executing parsd code:, you'll see something likeL
```
╭─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ Calling tool: 'web_search' with arguments: {'query': "best music recommendations for a party at Wayne's         │
│ mansion"}                                                                                                       │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```


### Tools

Let's talk about tools now!

To interact with a tool, an LLM needs an interface description with these key components:
- ==Name==: What the tool is called
- ==Tool descriptions==: What the tool does
- ==Input types and descriptions==: What arguments the tool accepts
- ==Output type==: What the tool returns

By using these tools, Alfred can make informed decisions and gather all the information needed to plan the perfect party.

In smolagents, tools can be defined in two wayS:
1. Using the ==@tool decorator== for simple function-based tools
2. Creating a ==subclass of Tool== for more complex functionality

==The @tool decorator is the most recommended way to define simple tools.==
Under the hood, smolagents will parse basic information about the function from Python.

Generating a tool that retrieves the highest-rated catering:

Let's imagine that Alfred has already decided on the menu for the party, butn ow he needs help preparing food for such a large number of guests.

Average can leverage a tool to search for the best catering services in the area.

```python
from smolagents import CodeAgent, HfApiModel, tool

@tool
def catering_service_tool(query: str) -> str:
    """
    This tool returns the highest-rated catering service in Gotham City.

    Args:
        query: A search term for finding catering services.
    """
    # This is just a dopey example of a tool. Let's pretend that it's actually going out and doing something important.
    
    # Example list of catering services and their ratings
    services = {
        "Gotham Catering Co.": 4.9,
        "Wayne Manor Catering": 4.8,
        "Gotham City Events": 4.7,
    }

    # Find the highest rated catering service (simulating search query filtering)
    best_service = max(services, key=services.get)

    return best_service

# then we can create our CodeAgent
agent = CodeAgent(tools=[catering_service_tool], model=HfApiModel())

# Run the agent to find the best catering service
result = agent.run(
    "Can you give me the name of the highest-rated catering service in Gotham City?"
)

print(result)   # Output: Gotham Catering Co.

```


Alternatively, we can define a Tool as a Python class, inheriting from Tool. We need to define:
- ==name==: Tool's name
- ==description==: A description used to populate the agent's system propmt
- ==inputs==: A dictionary with keys type and description, providing information to help the Python interpreter process inputs
- ==output_type==: Specifies the expected output type
- ==forward==: The method containing the inference logic to execute

Let's say we wanted to build a tool that generates creative theme ideas for our party!

```python
from smolagents import Tool, CodeAgent, HfApiModel

# A class-based SmolAgent tool
class SuperheroPartyThemeTool(Tool):

	# Name of the tool
    name = "superhero_party_theme_generator"
    
    # Description of the tool
    description = """
    This tool suggests creative superhero-themed party ideas based on a category.
    It returns a unique party theme idea."""

	# For each argument, the name, type, and description
    inputs = {
        "category": {
            "type": "string",
            "description": "The type of superhero party (e.g., 'classic heroes', 'villain masquerade', 'futuristic Gotham').",
        }
    }

	# Output type
    output_type = "string"

	# The actual function that works
    def forward(self, category: str):
        themes = {
            "classic heroes": "Justice League Gala: Guests come dressed as their favorite DC heroes with themed cocktails like 'The Kryptonite Punch'.",
            "villain masquerade": "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains.",
            "futuristic Gotham": "Neo-Gotham Night: A cyberpunk-style party inspired by Batman Beyond, with neon decorations and futuristic gadgets."
        }

        return themes.get(category.lower(), "Themed party idea not found. Try 'classic heroes', 'villain masquerade', or 'futuristic Gotham'.")

# Instantiate the tool
party_theme_tool = SuperheroPartyThemeTool()
agent = CodeAgent(tools=[party_theme_tool], model=HfApiModel())

# Run the agent to generate a party theme idea
result = agent.run(
    "What would be a good superhero party idea for a 'villain masquerade' theme?"
)

print(result)  # Output: "Gotham Rogues' Ball: A mysterious masquerade where guests dress as classic Batman villains."
```


### Default toolbox

Smolagents come with a set of ==pre-built tools that can be directly injected into your agent==
- This default toolbox includes:
	- ==PythonInterpreterTool==
	- ==FinalAnswerTool==
	- UserInputTool
	- DuckDuckGoSearchTool
	- GoogleSearchTool
	- VisitWebpageTool

Sharing and importing tools
- We can share custom tools on the Hub, as well as connect with ==HF Spaces== and ==LangChain tools==, significantly enhancing Alfred's ability to orchestrate an unforgettable party at Wayne Manor.

Sharing a Tool to the Hub
- Sharing your custom tool with the community is easy. Just upload it to your HuggingFace account using the push_to_hub method.
- We can import tools using the load_Tool function.

```python
# Saving a tool to the Hub
party_theme_tool.push_to_hub("{your_username}/party_theme_tool", token="<YOUR_HUGGINGFACEHUB_API_TOKEN>")


from smolagents import load_tool, CodeAgent, HfApiModel

# Loading a tool from the Hub
image_generation_tool = load_tool(
    "m-ric/text-to-image",
    trust_remote_code=True
)

agent = CodeAgent(
    tools=[image_generation_tool],
    model=HfApiModel()
)
```


You can also import a HuggingFace Space as a tool! 
- This opens up possibilities for integrating with thousand of spaces from the community for tasks from image generation to data analysis.

```python
from smolagents import CodeAgent, HfApiModel, Tool

# Load a tool from HF Space
image_generation_tool = Tool.from_space(
    "black-forest-labs/FLUX.1-schnell",
    name="image_generator",
    description="Generate an image from a prompt"
)

# Create a model
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Create an agent from the model and the tools
agent = CodeAgent(tools=[image_generation_tool], model=model)

# Ask the agent a question! :)
agent.run(
    "Improve this prompt, then generate an image of it.",
    additional_args={'user_prompt': 'A grand superhero-themed party at Wayne Manor, with Alfred overseeing a luxurious gala'}
)
```


Importing a LangChain tool:
```python
from langchain.agents import load_tools
from smolagents import CodeAgent, HfApiModel, Tool

# Load the LancgChain tool. load_tools does some lifting here too, which is interesting.
# "serpapi" is SerpAPI, which is a Google Search API
search_tool = Tool.from_langchain(load_tools(["serpapi"])[0])

# Agent
agent = CodeAgent(tools=[search_tool], model=model)

# Run
agent.run("Search for luxury entertainment ideas for a superhero-themed event, such as live performances and interactive experiences.")
```

Importing a tool collection from an MCP server

```python
import os
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters
from smolagents import HfApiModel

# model
model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct")

# Information to comply with MCP using Stdio
server_parameters = StdioServerParameters(
    command="uvx",
    args=["--quiet", "pubmedmcp@0.1.3"],
    env={"UV_PYTHON": "3.12", **os.environ},
)

# Get the tool collection from the MCP server.
with ToolCollection.from_mcp(server_parameters, trust_remote_code=True) as tool_collection:
	# Given whatever tools the MCP server exposes to your CodeAgetn
    agent = CodeAgent(tools=[*tool_collection.tools], model=model, add_base_tools=True)
    # Heeeeeelp
    agent.run("Please find a remedy for hangover.")
```


Retrieval Agents
- [[Agentic RAG]] extends traditional RAG by combining autonomous agents with dynamic knowledge retrieval.
	- A user's query is passed to a search engine, and the retrieved result are given to the model along with the query.
	- The model then generates a response based on the query and retrieved information.
	- Agentic RAG extends traditional RAG systems by combining autonomous agents with dynamic knowledge retrieval.

Agentic RAG addresses these issues by ==allowing the agent to autonomously formulate search queries, critique retrieved results, and conduct multiple retrieval steps== for a more tailored  and comprehensive output.

Let's build a simple agent that can search the web using DuckDuckGo!
- This agent will retrieve information and synthesize responses to answer queries.

Our agent can now:
- Search for latest superhero party trends
- Refine results to include luxury elements
- Synthesize information into a complete plan

Let's see:
```python
from smolagents import CodeAgent, DuckDuckGoSearchTool, HfApiModel

# Initialize the search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the model
model = HfApiModel()

agent = CodeAgent(
    model=model,
    tools=[search_tool],
)

# Example usage
response = agent.run(
    "Search for luxury superhero-themed party ideas, including decorations, entertainment, and catering."
)
print(response)
```

1. **Analyzes the Request:** Alfred’s agent identifies the key elements of the query—luxury superhero-themed party planning, with focus on decor, entertainment, and catering.
2. **Performs Retrieval:** The agent leverages DuckDuckGo to search for the most relevant and up-to-date information, ensuring it aligns with Alfred’s refined preferences for a luxurious event.
3. **Synthesizes Information:** After gathering the results, the agent processes them into a cohesive, actionable plan for Alfred, covering all aspects of the party.
4. **Stores for Future Reference:** The agent stores the retrieved information for easy access when planning future events, optimizing efficiency in subsequent tasks.

Custom knowledge bases can be invaluable, though!
Let's create a tool that queries a vector database of technical documentation of specialized knowledge.
Using semantic search, the agent can find the most relevant information for Alfred's needs.

A ==vector database== stores numerical representations (embeddings) of text and other data, created by machine learning models.
It enables semantic search by identifying similar meanings in high-dimensional space

Let's use a [[BM25]] retriever to search the knowledge base and return the top results, and ==RecursiveCharacterTextSplitter== to split the documents into smaller chunks for more efficient search.

```python
from langchain.docstore.document import Document # Document objects
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Assumedly used to chunk documents
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever # A retriever fro mLangchain
from smolagents import CodeAgent, HfApiModel

# A class based tool that returns information
class PartyPlanningRetrieverTool(Tool):
    # Name of the tool
    name = "party_planning_retriever"
    
    description = "Uses semantic search to retrieve relevant party planning ideas for Alfred’s superhero-themed party at Wayne Manor."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be a query related to party planning or superhero themes.",
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=5  # Retrieve the top 5 documents
        )

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(
            query,
        )
        return "\nRetrieved ideas:\n" + "".join(
            [
                f"\n\n===== Idea {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )

# Simulate a knowledge base about party planning
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in party_ideas
]

# Split the documents into smaller chunks for more efficient search
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ".", " ", ""],
)
docs_processed = text_splitter.split_documents(source_docs)

# Create the retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)

# Initialize the agent
agent = CodeAgent(tools=[party_planning_retriever], model=HfApiModel())

# Example usage
response = agent.run(
    "Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options."
)

print(response)
```


With agentic RAG, the agent can employ sophisticated strategies, like:
1. ==Query Reformulation==: Instead of using the raw user query, the agent can rephrase the query to optimize it.
2. ==Multi-Step Retrieval==: Can perform multiple searches, using initial results to inform later searches.
3. ==Source Integration==: Can combine information from multiple sources, like web search and local documentation.
4. ==Result Validation==: Retrieved content can be analyze for relevance and accuracy before being included in responses.

Effective agentic RAG systems require careful consideration of several key aspets.
The agent should select between available tools based on the query type and context.

Memory systems help maintain conversation history and avoid repetitive retrievals.

So: ==the `@tool` decorator is recommended for simple function-based tools, while subclasses of `Tool` offer more flexibility for complex functionality or custom metadata==


## Multi-Agent Systems
- Multi-agent systems enable specialized agents to collaborate on complex tasks, improving modularity, scalability, and robustness.
- Instead of relying on a single agent, ==tasks are distributed among agents with distinct capabilities==

A typical setup might include:
- A manager agent for task delegation
- A code interpreter agent for code execution
- A web search agent for information retrieval

((My question at this point: Why not just have a single agent with all of these tools?))


![[Pasted image 20250419153812.png]]
Example of a multi-agent system where a Manager Agent coordinates a Code Interpreter Tool and a Web Search Agent, which in turn utilizes tools to gather information.

A multi-agent system consists of multiple specialized agents working together under the coordination of an ==Orchestrator Agent==

Let's solve a complex task with a multi-agent hierarchy.

Can we build an agent to find a replacement for Batman's Batmobile car, which was stolen? There's old batmobiles left behind on the various movie sets that we could tune up, if we could find them!

Find all Batman filming locations in the world, calculate the time to transfer via boat to there, and represent them on a map, with a color varying by boat transfer time. Also represent some supercar factories with same boat transfer time.

```shell
pip install 'smolagents[litellm]' plotly geopandas shapely kaleido -q
```

\<example>

We can ask our model to add some dedicated planning steps, and add more prompting:
```python
agent.planning_interval = 4

detailed_report = agent.run(f"""
You're an expert analyst. You make comprehensive reports after visiting many websites.
Don't hesitate to search for many queries at once in a for loop.
For each data point that you find, visit the source url to confirm numbers.

{task}
""")

print(detailed_report)
```

The model’s context window is quickly filling up. So **if we ask our agent to combine the results of detailed search with another, it will be slower and quickly ramp up tokens and costs**.

Let's improve the structure by splitting the task between two separate agents!
- ==Each agent is more focused on its ore subtask, thus more performant==
- ==Separating memories reduces the count of input tokens at each step, thus reducing latency and cost.==

The manager agent should have plotting capabilities to write its final report, os let's give it access to additional imports:

```python
model = HfApiModel(
    "Qwen/Qwen2.5-Coder-32B-Instruct", provider="together", max_tokens=8096
)

web_agent = CodeAgent(
    model=model,
    tools=[
        GoogleSearchTool(provider="serper"),
        VisitWebpageTool(),
        calculate_cargo_travel_time,
    ],
    name="web_agent",
    description="Browses the web to find information",
    verbosity_level=0,
    max_steps=10,
)
```

The manager agent will need to do some heavy mental lifting, so let's give it astronger model and add a planning interval to the mix!
```python
from smolagents.utils import encode_image_base64, make_image_url
from smolagents import OpenAIServerModel

def check_reasoning_and_plot(final_answer, agent_memory):
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True

manager_agent = CodeAgent(
    model=HfApiModel("deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096),
    tools=[calculate_cargo_travel_time],
    managed_agents=[web_agent],
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5, # Interval at which the agent will run a planning step.
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],
    max_steps=15,
)
```


### Vision and Browser Agents

Let's say that Alfred suspects that a visitor might be The Joker! But he needs to verify his identity to prevent him from entering the manor.

```python
from PIL import Image
import requests
from io import BytesIO

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg", # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg" # Joker image
]

images = [] # A list of PIL Images
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36" 
    }
    response = requests.get(url,headers=headers)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)
```

...

In this approach, images are dynamically added to the agent’s memory during execution. ==As we know, agents in `smolagents` are based on the `MultiStepAgent` class, which is an abstraction of the ReAct framework==. This class operates in a structured cycle where various variables and knowledge are logged at different stages:
1. **==SystemPromptStep==:** Stores the system prompt.
2. **==TaskStep==:** Logs the user query and any provided input.
3. **==ActionStep==:** Captures logs from the agent’s actions and results.

![[Pasted image 20250419161153.png]]

Let's build our complete example now! 
- Alfred wants to browse for details to have full control over the guest verification process.




# Unit 2.2: **LlamaIndex**
- 

# Unit 2.3: LangGraph
- 