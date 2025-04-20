
LlamaIndex is a complete toolkit for creating LLM-powered agents over your data using indexes and workflows.

We'll focus on three main parts that help build agents in LlamaIndex:
- Components
- Agents and Tools
- Workflows

Let's look at some of these key parts:
- ==Components==: Basic building blocks you use in LlamaIndex: Things like ==prompts, models, and databases==. Often ***help connect your LlamaIndex system*** with other tools and libraries.
- ==Tools==: Tools are components that provide ==specific capabilities like searching, calculating, or accessing external services==. They are the ***building blocks*** that enable agents to perform tasks.
- ==Agents==: Agents are ==autonomous components== that can use tools and make decisions. They coordinate tool usage to accomplish complex cogals.
- ==Workflows==: ==Step-by-step process== that process logic together. Workflows or agentic workflows are a way to structure agentic behavior without the explicit use of agents.

What makes LlamaIndex Special?
- While LlamaIndex doe some things similar to other frameworks like smolagents, it has some key benefits:
	- ==Clear Workflow System==: Use an event-drive and async-first syntax to compose logic.
	- ==Advanced Document Parsing with LlamaParse==: LlamaParse was made specifically for LlamaIndex, so integration is seamless, though it's a paid feature.
	- ==Many Ready-to-Use Components==: LlamaIndex has been around for a while, so ti works with lots of other frameworks! This means it has many tested and reliable components, like LLMs, retrievers, index, more.
	- ==LlamaHub==: A registry of these components, agents, and tools you can use within LlamaIndex.

### Intro to LlamaHub
- A registry of integrations, agents, and tools you can use in LlamaIndex.

Installation commands for LlamaHub generally follow this EASY TO REMEMBER format:
```shell
pip install llama-index-{component-type}-{framework-name}
```

Let's try to install the dependencies for an LLM and embedding component using the HF inference API integration.

```shell
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
```

Let's see how they're used:
```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
    model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
    temperature=0.7,
    max_tokens=100,
    token=hf_token,
)

response = llm.complete("Hello, how are you?")
print(response)
# I am good, how can I help you today?
```


Now let's dive into Components in LlamaIndex

Alfred needs to understand our requests and prepare, find and use relevant information to help complete tasks.
- This is where LlamaIndex's components come in.

While LlamaIndex has many components, let's focus specially on the ==QueryEngine== component.
- It can be used as a RAG tool for an agent.

Let's make a RAG pipeline using components.

There are 5 key stages within RAG, which in turn will be part of most larger applications you'll build:
1. ==Loading==: This refers to getting your data from where it lives (PDFs, DB, API) into your workflow. LlamaHub has 100s of integrations.
2. ==Indexing==: Creating a data structure (e.g. vector embeddings) that allows for querying the data.
3. ==Storing==: Once your data is indexed you will want to store your index and other metadata.
4. ==Querying==: For any given indexing strategy, there are many ways you can utilize LLMs and LlamaIndex data structures to query, including ***sub-queries, multi-step queries, and hybrid strategies***.
5. ==Evaluation==: A critical step in any flow is checking how effective it is relative to other strategies. This is an objective measure of how accurate/faithful/fast our responses to queries are.

Loading and embedding documents

There are ==Three main ways ot load data into LlamaINdex!==
1. ==SimpleDirectoryReader==: A built-in loader for various file types from a local directory.
2. ==LlamaParse==: LlamaIndex's official tool for PDF parsing, available as a managed API.
3. ==LlamaHub==: A registry of hundreds of data-loading libraries to ingest data from any source.

The simplest way to load data is with `SimpleDirectoryReader`.

This versatile component can load various file types from a folder and convert them into `Document` objects that LlamaIndex can work with.
- Let's see how we can use `SimpleDirectoryReader` to load data from a folder.

```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir="path/to/directory")
documents = reader.load_data()
```

After loading our documents, we need to break them into smaller pieces called `Node` objects.
- ==Nodes== are just ==chunks== of text from the original document that's easier for the AI to work with, while it still has references to the original `Document` object.

The ==IngestionPipeline== helps us create these nodes through two key transformations.
1. ==SentenceSplitter== breaks documents down into manageable chunks by *splitting them at natural sentence boundaries!*
2. ==HuggingFaceEmbedding== converts each chunk into numerical embeddings -- vector representations that capture the semantic meaning in a way AI can process effectively.


This process helps us organize our documents in a way that's more useful for searching and analysis.
```python
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# create the pipeline with transformations
pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_overlap=0), # Splits into chunks (we're specifying no chunk overlap)
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"), # Embeds each chunk
    ]
)

nodes = await pipeline.arun(documents=[Document.example()])
```

Now that we've created our pipeline and split/embedded our documents into chunk vectors, we need to INDEX them to make them searchable later!

Since we're using an ingestion pipeline, ==we can directly attach a vector store to a pipeline to populate it!==
We'll use `Chroma` to store our documents.

```shell
pip install llama-index-vector-stores-chroma
```


```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db") # Create the chromaDB instance
chroma_collection = db.get_or_create_collection("alfred") # Create a collection in the db
vector_store = ChromaVectorStore(chroma_collection=chroma_collection) # Create vector store obj pointing to the db collection

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=25, chunk_overlap=0), # Setting a max chunk size, no overlap
        HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"), # Setting the embedding model
    ],
    vector_store=vector_store,  # See that we're adding the vector store at the end!
)
```

This is where vector embeddings come in: By embedding both the query and node in the same vector space, we can find relevant matches.

The ==VectorStoreIndex== handles this for us, using the same embedding model we used during ingestion to ensure consistency.

Let's see how to create this index from our vector store and embeddings:
```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model) # Pass a ChromaVectorStore, Embed model
```

How can we query a VectorStoreIndex with prompts and LLMs?

Before we query our index, we need to convert it into a ==query interface==. The most common conversion options are:
- ==as_retriever==: For basic document retrieval, returning a list of NodeWithscore objects with similarity scores
- ==as_query_engine==: For single question-answer interactions, returning written responses. ==Most popular for agent-like interactions.==
- ==as_chat_engine==: For conversational interactions that maintain memory across multiple messages, returning a written response using chat history and indexed context.

Let's focus on the query engine, since this is an agents course.
- We also need to pass in an LLM to the query engine to use for the response.


```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct") # Make the API ("model"?)

# Take our VectorStoreIndex (which wraps a ChromaVectorStore, which points to a chroma collecition in a chroma db)
query_engine = index.as_query_engine(
    llm=llm, # A reference to our Model
    response_mode="tree_summarize", # Hmmm
)

# Nowe we can query it directly and get a written response from our model.
query_engine.query("What is the meaning of life?")
# The meaning of life is 42
```

Under the hood:
- The query engine doesn't ONLY use the LLM to answer the question; it also uses a ==ResponseSynthesizer== as a strategy to process the response.
- Once again, this is fully customizable but there are three main strategies that work out of the box:
	- ==refine==: Create and refine an answer by sequentially going through each retrieved text chunk. This makes a separate LLM call per Node/retrieved chunk.
	- ==compact== (Default): Similar to refining but concatenating the chunks beforehand, resulting in fewer LLM calls.
	- ==tree_summarize==: Creates a detailed answer by going through each retrieved text chunk and creating a tree structure of the answer 🤔

> There's also some sort of low-level composition API that lets you customize and fine-tune every step of a query process to match our exact needs.

Evaluation and Observability
- ==LlamaIndex provides built-in evaluation tools to assess response quality.==
- These evaluators leverage LLMs to analyze responses across different dimensions. 
	- ((It seems to me that these are sort of like ==Guardrail==-like models, which are in-the-loop of a query.))

The three main evaluators available are:
- ==FaithfulnessEvaluator==: Evaluates the faithfulness of the answer by checking if the answer is supported by the context.
- ==AnswerRelevancyEvaluator==: Evaluate the relevance of the answer by checking if he answer is relevant to the question.
- ==CorrectnessEvaluator==: Evaluate the correctness of the answer by judging if the answer is correct.


```python
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = # from the previous section
llm = # from the previous section

# query index
evaluator = FaithfulnessEvaluator(llm=llm) # We could potentially use a different LLM; doesn't seem like we HAVE to provide a prompt.

# Query our query_engine (which is the query engien version of our VectorStoreIndex, which wraps ChromeVector store, which points to a collection of a db)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response) # Ah, so it's not in the loop of the query_engine.query(...)
eval_result.passing
```

Even without direct evaluation, we can gain insights into how our system is performing through observability tools!

We can do this using ==LlamaTrace!==

```python
import llama_index
import os

PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)
```

Using tools in LlamaIndex
- Defining a clear set of TOOLS is crucial to performace!


There are four main types of TOOLs in LlamaIndex:
- ==FunctionTool==: ==Converts any Python function into a tool that an agent can use==. Automatically figures out how the function works.
- ==Search Engine==: A tool that lets agents use ==query engines==. ==Since agents are built on query engines, they can also use other agents as tools==.
- ==Toolspecs==: ==Sets== of tools created by the ==community==, which often include ==tools for specific services like Gmail==.
- ==Utility Tools==: Special ==tools that help handle large amounts of data from other tools==.


#### Creating a FunctionTool
- A simple way to wrap any Python function and make it available to an agent!
- You can pass either a ==synchronous or asynchronous function== to the tool, along with the optional `name` and `description` parameters.
- The ==name== and ==description== are particularly important, as they help the agent understand when and how to use the tool effectively.

```python
from llama_index.core.tools import FunctionTool # Import FunctionTool from LlamaIndex

# A function that we'd like to make int oa tool
def get_weather(location: str) -> str:
    """Useful for getting the weather for a given location."""
    print(f"Getting weather for {location}")
    return f"The weather in {location} is sunny"

# We create a tool from get_weather... make sure to pass the (optional) name and description
tool = FunctionTool.from_defaults(
    get_weather, # a fn
    name="my_weather_tool",  # tool name
    description="Useful for getting the weather for a given location.", # tool description
)

# Now we can .call(...) the FuncitonTool to nivoke the wrapped get_weather function.
tool.call("New York")
```


#### Creating a QueryEngineTool
- The ==QueryEngine== we defined in the previous units ==can be transformed into a tool== using the ==QueryEngineTool== class.
- Let's see how to create a QueryEngineTool from QueryEngine below!

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5") # Embedding model

db = chromadb.PersistentClient(path="./alfred_chroma_db") # Creating a Chroma DB with filepath location
chroma_collection = db.get_or_create_collection("alfred") # Creating a collection in our ChromaDB
vector_store = ChromaVectorStore(chroma_collection=chroma_collection) # Creating a ChromeVectorStore @ our collection

index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model) # Creating a LI VectorStoreIndex(ChromaVS)

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct") # Model
# Note that QueryEngine takes our VectorStoreIndex (LI object wrapping ChromaDB Index, basically) and adds an LM into the eq.
query_engine = index.as_query_engine(llm=llm) # Query Engine from our LlaamIndex VectorStoreIndex

# We can then take our QueryEngine and turn it into a Tool, giving it a name and description again.
tool = QueryEngineTool.from_defaults(query_engine, name="some useful name", description="some useful description")
```


### Creating ToolSpecs
- ToolSpecs are collections of tools that work together harmoniously. Like a well-organized professional toolkit.
- ==Just a mechanic's toolkit that contains complementary tools that work together for vehicle repairs, a ToolSpec combines related tools for a specific purpose.==
- For example, an accounting agent's ToolSpec ==might elegantly integrate spreadsheet capabilities, email functionality, and calculation tools== to handle financial tasks with precision and efficiency.

Installing the google toolspec
```shell
pip install llama-index-tools-google
```

And loading the toolspec and converting it to a list of tools
```python
from llama_index.tools.google import GmailToolSpec

tool_spec = GmailToolSpec()
tool_spec_list = tool_spec.to_tool_list()

[(tool.metadata.name, tool.metadata.description) for tool in tool_spec_list]
```


### Model Context Protocol (MCP) in LlamaIndex
- LlamaIndex also allows using MCP through a ToolSpec on LlamaHub -- run an MCP server and start using it through the following implementation.

```python
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# We consider there is a mcp server running on 127.0.0.1:8000, or you can use the mcp client to connect to your own mcp server.
mcp_client = BasicMCPClient("http://127.0.0.1:8000/sse")  # Connect to an MCP Server
mcp_tool = McpToolSpec(client=mcp_client) # Create a ToolSpec from the mcp client

# get the agent 
agent = await get_agent(mcp_tool) # ???? WE HAVEN'T LEARNED WHAT THIS IS YET LOL

# create the agent context
agent_context = Context(agent)
```

### Utility Tools
- Oftentimes querying an API can return an excessive amount of data, some of which is irrelevant and might overflow the context window of the LLM or cost too much.

Here are two main utility tools:
- ==OnDemandToolLoader==: This tool turns any existing LlamaIndex data loader into a tool that an agent can use. 
	- During execution, we load data from the data loader, index it, and then query it "on-demand" based on a supplied natural langauge query string.
	- All three steps happen in a single tool call.
- ==LoadAndSearchToolSpec==: Take any existing Tool as input, implements `to_tool_list`, and when that function is called, two tools are returned: A ==loading tool== and then a ==search tool==.
	- The load tool execution would call the underlying tool, and then index the output.
	- The search tool execution would take a query string as input and call the underlying index.
	- ((I don't understand this explanation))





## Using Agents in LlamaIndex
- LlamaIndex supports Three Main Types of reasoning agents:
![[Pasted image 20250419173945.png|800]]

1. ==Function Calling Agents==: These work with AI models that can call specific functions.
2. ==ReAct Agents==: Can work with any AI that does chat or text endpoint and deal with complex reasoning tasks.
3. ==Advanced Custom Agents==: Use more complex methods to deal with more complex tasks and workflows.

((Not clear what these special FunctionCalling AI models are?... Or how they're different from ReACt agents.))

Initializing Agents
- ==To create an agent, we start by providing it with a set of functions/tools that define its capabilities.==
- As of this writing, the agent will automatically use the function calling API (if available) or a standard ReAct agent loop.

```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.tools import FunctionTool

# define sample Tool -- type annotations, function names, and docstrings, are all included in parsed schemas!
def multiply(a: int, b: int) -> int:
    """Multiplies two integers and returns the resulting integer"""
    return a * b

# initialize llm
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# initialize agent
agent = AgentWorkflow.from_tools_or_functions( # Agent Workflow has a model and tools.
    [FunctionTool.from_defaults(multiply)],  # Making a FunctionTool from our function (with appropriate annotations)
    llm=llm # Our model
)
```


==Agents are stateless by default!==
- Remembering past interactions is ==opt-in== using a ==Context== object!
- This might be useful for interactions like a Chatbot, which maintains context across multiple messages or a task manager that needs to track progress over time.

```python
# stateless
response = await agent.run("What is 2 times 2?")

# remembering state
from llama_index.core.workflow import Context

ctx = Context(agent)  # Context wraps our AgentWorkflow, in this case.

# It seems like we pass our context along with our successive agent.run commands.
response = await agent.run("My name is Bob.", ctx=ctx) 
response = await agent.run("What was my name again?", ctx=ctx)
```
Above: ==Notice that agents in LlamaIndex are ASYNC!==


### Creating RAG Agents with QueryEngineTools

AgenticRAG is a powerful way to use agents to answer questions about your data.

We can ==wrap== a QueryEngine as a TOOL for an agent! In doing so, we can define a name and description.

The LLM will use this information to correctly use the tool.
- Let's see how to load in a QueryEngine tool using the QueryEngine we created in the component section.


```python
from llama_index.core.tools import QueryEngineTool

# QueryEngine tool can turn a QueryEngine (which was an interface to a VectorStoreIndex, which wrapped a Chroma VS) into a tool. And then we can hand that tool to some sort of LlamaIndex agent.

# Take our VectorStoreIndex and turn it into a QueryEngine
query_engine = index.as_query_engine(llm=llm, similarity_top_k=3) # as shown in the Components in LlamaIndex section

# Take our QueryEngine and turn it into a Tool using QueryEngineTool.from_defaults
query_engine_tool = QueryEngineTool.from_defaults(
    query_engine=query_engine,
    name="name", # Name
    description="a specific description", # Desc
    return_direct=False, # ?
)

# Now we can create an Agent Workflow using our model and a list of tools. And look, a system prompt!
query_engine_agent = AgentWorkflow.from_tools_or_functions(
    [query_engine_tool],
    llm=llm,
    system_prompt="You are a helpful assistant that has access to a database containing persona descriptions. "
)
```
Above: We had created QueryEngineTools before, but we hadn't seen them used until now.

Creating MultiAgent Systems!
- The AgentWorkflow class also directly supports Multi-Agent Systems!
- We give each agent a Name and a Description
- The system maintains  a single active speaker, with each agent having the ability to hand off to another agent.

==We can narrow the scope of each agent, to increase their general accuracy when responding to user messages.==

==Agents in LlamaIndex can also directly be used as tools for other agents!==

```python
from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

# Define some tools
# Adding tool!
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

# Subtracting tool!
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    return a - b

# Create agent configs
# NOTE: we can use FunctionAgent or ReActAgent here.

# FunctionAgent works for LLMs with a function calling API.
# ReActAgent works for any LLM.
calculator_agent = ReActAgent(
    name="calculator",
    description="Performs basic arithmetic operations",
    system_prompt="You are a calculator assistant. Use your tools for any math operation.",
    tools=[add, subtract],
    llm=llm,
)

query_agent = ReActAgent(
    name="info_lookup",
    description="Looks up information about XYZ",
    system_prompt="Use your tool to query a RAG system to answer information about XYZ",
    tools=[query_engine_tool],
    llm=llm
)

# Create and run the workflow
# NOTE: See here that instead of in the previous one, here we pass AgentWorkflow a list of agents, with a root agent
# c.f. in the previous one, we used AgentWorkflow.frmo_tools_or_functions and a single model/tools.
agent = AgentWorkflow(
    agents=[calculator_agent, query_agent], root_agent="calculator"
)

# Run the system
response = await agent.run(user_msg="Can you add 5 and 3?")
```
Above: I have no idea what they mean when they talk about "LLMs with a function-calling API"



Next, we can look at Creating agentic workflows in LlamaIndex
- A ==workflow== in LlamaIndex provides a structured way to organize our code into sequential and manageable steps.

![[Pasted image 20250419180543.png]]
Workflows have benefits:
- Clear organization of code into discrete steps
- Event-driven architecture for flexible control flow
- Type-safe communication between steps
- Built-in State management
- support for both simple and complex agent interactions

==As you might have guessed, workflows strike a great balance between the autonomy of agents while maintaining control over the overall workflow.==

Creating Workflows

Let's make a basic single-step workflow by ==defining a class that inherits from Workflow== and decorating our functions with ==@step== and adding ==StartEvent== and ==StopEvent==, which are special events that are used to indicate the start and end of the workflow.

```python
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step

# Define a Workflow-subclassing class
class MyWorkflow(Workflow):
	
	# Use the @step decorator for a function that takes a StartEvent and returns a StopEvent
    @step
    async def my_step(self, ev: StartEvent) -> StopEvent:
        # do something here
        return StopEvent(result="Hello, world!")


w = MyWorkflow(timeout=10, verbose=False)
result = await w.run()
```

Now let's talk about connecting multiple steps!

Here, we need to create Custom Events that carry data between steps:

```python
from llama_index.core.workflow import Event

# This is an Event class that we'll use as our intemediate data representation to be passed beetween our two steps.
class ProcessingEvent(Event): # Note that we subclass Event
    intermediate_result: str

# Ntoe that we subclass Workflow
class MultiStepWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> ProcessingEvent: # Note that it returns a ProcessingEvent
        # Process initial data
        return ProcessingEvent(intermediate_result="Step 1 complete")

    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent: # Note that it returns a StopEvent
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)

w = MultiStepWorkflow(timeout=10, verbose=False)
result = await w.run()
result
```

The type hinting is important here, as it ensures that the workflow is executed correctly. Let's complicate things a bit more!

==Loops== and ==Branches==:
- Type hinting is the most powerful part of workflows because it allows us to create branches, loops, and joins to facilitate more complex workflows.
- Let's show an example of creating a ==loop== by using the Union Operator....

==In the example below, we see that the LoopEvent is taken as Input for the step, and can also be returned as an output.==

```python
from llama_index.core.workflow import Event
import random


class ProcessingEvent(Event):
    intermediate_result: str


class LoopEvent(Event):
    loop_output: str


class MultiStepWorkflow(Workflow):
	# See that this Step1 can "Loop" back to itself, I think. 
	# I'm curious though how the workflow "knows" that step_one should be reinvoked with the LoopEvent?
    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

	# See that this one only receives a ProcessingEvents
    @step
    async def step_two(self, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


w = MultiStepWorkflow(verbose=False)
result = await w.run()
result
```

It's not clear to me how it knows that a LoopEvent returned from StepOne should be rerouted to the StepOne step.
Is it because StepOne can take a LoopEvent as input?

So what if we had 8 events, and steps 2 and 8 could self-loop? Do they each have to have a DIFFERENT LoopEvent class (LoopEvent1, LoopEvent2, for instance?)


### Adding State Mangement

State management is useful when you want to keep track of the state of the workflow, so that every step has access to the same state.
- We can do this by using the `Context` type hint on top of a parameter in the step function!


```python
from llama_index.core.workflow import Context, StartEvent, StopEvent

@step
async def query(self, ctx: Context, ev: StartEvent) -> StopEvent:
    # store query in the context
    await ctx.set("query", "What is the capital of France?")

    # do something with context and event
    val = ...

    # retrieve query from the context
    query = await ctx.get("query")

    return StopEvent(result=val)
```


We can also do Multi-Agent Workflows!
- AgentWorkflow uses Workflow Agent to allow you to create a system of one or more agents that can collaborate and hand off tasks to eachotehr based on their specialized capabilities.
- Maybe each agent handles different aspects of a task.

Instead of `llama_index.core.agent`, we import the agent classes from `llama_index.core.agent.workflow.`

Each agent can then:
- Handle the request directly using their tools
- Handoff to another agent better suited for the task
- Return a response to the user

Let's see how to create a multi-agent workflow:

```python
from llama_index.core.agent.workflow import AgentWorkflow, ReActAgent
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Define some tools
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# Get a modle
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# we can pass functions directly without FunctionTool -- the fn/docstring are parsed for the name/description
multiply_agent = ReActAgent(
    name="multiply_agent",
    description="Is able to multiply two integers",
    system_prompt="A helpful assistant that can use a tool to multiply numbers.",
    tools=[multiply],
    llm=llm,
)

addition_agent = ReActAgent(
    name="add_agent",
    description="Is able to add two integers",
    system_prompt="A helpful assistant that can use a tool to add numbers.",
    tools=[add],
    llm=llm,
)

# Create the workflow
workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent",  # It seems that... the multiply agent is the "manager agent"? 
)

# Run the system
response = await workflow.run(user_msg="Can you add 5 and 3?")
```

Before starting the workflow, we can provide an initial state dict that will be available to all agents.
- The state is stored in the state key of the workflow context.
- It will be injected into the state_prompt with augments each new user message.

Let's inject a counter to count function calls by modifying the previous example:
```python
from llama_index.core.workflow import Context

# Define some tools
async def add(ctx: Context, a: int, b: int) -> int:
    """Add two numbers."""
    # update our count
    cur_state = await ctx.get("state") # ctx.state
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a + b

async def multiply(ctx: Context, a: int, b: int) -> int:
    """Multiply two numbers."""
    # update our count
    cur_state = await ctx.get("state") # ctx.state
    cur_state["num_fn_calls"] += 1
    await ctx.set("state", cur_state)

    return a * b

...

workflow = AgentWorkflow(
    agents=[multiply_agent, addition_agent],
    root_agent="multiply_agent"
    initial_state={"num_fn_calls": 0}, # See INITIAL STATE ADDED HERE
    state_prompt="Current state: {state}. User message: {msg}", # See STATE_PROMPT
)

# run the workflow with context
ctx = Context(workflow)
response = await workflow.run(user_msg="Can you add 5 and 3?", ctx=ctx)  # See CONTEXT HERE

# pull out and inspect the state
state = await ctx.get("state")
print(state["num_fn_calls"])
```

Congrats!





