
Agent
> An Agent is a system that leverages an AI model to interact with its environment in order to achieve a user-defined objective. It combines reasoning, planning, and the execution of actions (often via external tools) to fulfill tasks.


Agent parts:
- Brain: AI Model
- Body: Capabilities and tools; things the agent is equipped to do.

Spectrum of Agency:
![[Pasted image 20250417144744.png]]
Above:
- See that a "workflow" where the models are only used to perform actions in a deterministic DAG-like graph... isn't very agentic. So agentic stuff is about who owns the control flow AND who executes the nodes in the DAG.


The design of the ==Tools== that we give to agents is very important and has a great impact on the quality of our agent.
Some tasks will require very specific tools to be crafted, while others may be solved with general-purpose tools like "web search."

Note that ==Actions== aren't the same as tools:
- An ==Action== is a higher-level concept that might involve the use of multiple tools to complete.
- Actions are the steps that agents take (higher-level objectives), while tools are specific functions that the agent can call upon.

Example:
- Virtual Assistant
	- Takes user queries, analyze context, retrieve information from DBs, provide responses or initiate actions (e.g. setting reminders, sending messages, controlling smart devices).
- Customer Service Chatbot
	- Many companies deploy chatbots as agents that interact with customers in natural language.
	- These agents can answer questions, guide users through troubleshooting steps, open issues in internal databases, complete transactions.
- AI NPC in a Video Game
	- Instead of following rigid behaviors, agent NPCs can respond contextually, adapt to player interactions, and generate more nuanced dialogues. This helps create more lifelike, engaging characters that evolve alongside player actions.

To summarize, agents use LLMs as their core reasoning engine to:
- Understand natural language
- Reason and Plan
- Interact with their environment

LLMs
- Encoder
- Decoder
- Seq2Seq/Encoder-Decoder

The objective is to predict the next ==token==, given a sequence of previous tokens.
You can think of a "token" as if it were a "word," but for efficiency reasons, LLMs don't use whole words.

Each LLM has some ==special tokens== specific to the model.
- The LLM uses these tokens to open and close the structured components of its generation.
- To indicate the start or end of a sequence, message, or response, for instance.
- The most important of these is the ==End of Sequence== (EOS) token.

There are a variety of special tokens:
![[Pasted image 20250417150006.png]]
- It's important to appreciate the diversity of these special tokens, and the role they play in text generation for language models.
- If you want to know more about special tokens, you can check out the configuration of the model in its Hub repository.


{More explanation of how transformers work}

Decoding strategies
- Simplest is ==greedy decoding==

There are also options like ==Beam Search==
- Explores multiple candidate sequences to find the one with maximum total score, even if some individual tokens have lower scores.

{Attention}
{Using LLMs}

![[Pasted image 20250417150549.png]]
Above:
- The difference between what you see in the UI and what you actually send in these conversations to models.
- The entire context is formatted using a ==CHAT TEMPLATE== that is specific to the model.
	- These act as a bridge between conversational messages (user and assistant turns) and the specific formatting requirements of your chosen language model.

Chat templates structure the communication between the user and the agent, ensuring that every model, despite its unique special tokens, receives the correctly-formatted prompt.


==System Messages ("System Prompts")==
- Define how the model should behave. These serve as persistent instructions, guiding every subsequent interaction.

```python
system_message = {
    "role": "system",
    "content": "You are a professional customer service agent. Always be polite, clear, and helpful."
}
```

When using ==Agents==, ==System Messages== often give information about ==available tools,== provide ==instructions on how to format actions to take==, and include ==guidelines on how the thought process should be structured.==

![[Pasted image 20250417150904.png|400]]

Conversations: User and Assistant messages:
- Conversations consist of alternating messages between a User and an Assistant (LLM):

```python
conversation = [
    {"role": "user", "content": "I need help with my order"},
    {"role": "assistant", "content": "I'd be happy to help. Could you provide your order number?"},
    {"role": "user", "content": "It's ORDER-123"},
]
```

We always concatenate all messages in the conversation (along with the system prompt) and pass it to the LLM as a single, stand-alone sequence.
The chat template converts all messages inside this Python list into a prompt, which is just a string input that contains all the messages.

For example, the above conversation would be formatted as:
```html
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
I need help with my order<|im_end|>
<|im_start|>assistant
I'd be happy to help. Could you provide your order number?<|im_end|>
<|im_start|>user
It's ORDER-123<|im_end|>
<|im_start|>assistant
```

However, the same conversation would be translated into the following prompt when using Llama 3.2:
```html
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 10 Feb 2025

<|eot_id|><|start_header_id|>user<|end_header_id|>

I need help with my order<|eot_id|><|start_header_id|>assistant<|end_header_id|>

I'd be happy to help. Could you provide your order number?<|eot_id|><|start_header_id|>user<|end_header_id|>

It's ORDER-123<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

So it's interesting that the same conversation:
```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is calculus?"},
    {"role": "assistant", "content": "Calculus is a branch of mathematics..."},
    {"role": "user", "content": "Can you give me an example?"},
]
```
Needs to be translated into various specialized prompts, depending on which model you're using!


Base Models vs Instruct Models
- A ==Base Model== is trained on raw text data to predict the next token.
- An ==Instruct Model== is fine-tuned specifically to follow instructions and engage in conversations.

To make a Base Model into an Instruct Model, we have to ==format our prompts in a consistent way that the model can understand!==
- [[ChatML]] is one such template format that structure conversations with clear role indicators (system, user, role).
- It's important to note that a base model could be finetuned on different chat templates, so when we're using an instruct model, we need to make sure we're using the correct chat template.

In `transformers`, chat templates include Jinja2 code that describes how to transform the ChatML list of JSON messages, as presented in the above examples, into a textual representation of the system-level instructions, user messages and assistant responses that the model can understand.

Here's a simplified version of the `SmolLM2-135M-Instruct`Jinja2 chat template:

```python
{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face
<|im_end|>
{% endif %}
<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}
```
- Above: Prepends a system prompt, if there isn't one in the list of messages.
- Otherwise, turns the list of messages into chat template format.

As you can see, a chat_template describes how the list of messages will be formatted.

The `transformers` library will take care of chat templates for you as part of the tokenization process.
- We just need to structure our messages in the correct way, and the tokenizer objects will take care of the rest.


The easiest way to ensure your LLM correctly-formatted conversation is to use the ==chat_template== from the model's tokenizer.

Given:
```python
messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
```

We can just load the tokenizer and call `apply_chat_template`:
```python
from transformers import AutoTokenizer

# Make the tokenizer
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")

# Turn the conversation (list[message]) into a rendered prompt using the tokenizer!
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```



## What are tools?

A crucial aspect of Agents is using the right Tools.
- By giving the right tools and by properly describing them, you can dramatically increase what your AI can accomplish!

What are AI tools:
- A tool is a function given to the LLM. This function should fulfill a clear objective.
![[Pasted image 20250417152954.png]]

Furthermore, **LLMs predict the completion of a prompt based on their training data**, which means that their internal knowledge only includes events prior to their training. Therefore, if your agent needs up-to-date data you must provide it through some tool.
- This applies to queries like "what's the weather?" and "Who's going to win the game on Saturday?"

==A tool description should contain:==
- A textual description of what the function does.
- A callable (something to perform an action)
- Arguments with typings
- (Optional) Outputs with typings

==How do tools work?:==
- LLMs can only receive text inputs and generate text outputs. ==LLMs themselves have no ways to call tools on their own!==
- We have to tell the LLM about the existence of these tools, and instruct it to generate text-based invocations when needed.
- The ==Agent== itself (which wraps the model), identifies from the LLM's textual output (e.g. `weather_tool("Paris")`) that a tool call is required, ==executes the tool on the LLM's behalf==, and retrieves the actual weather data!
- ==The agent then appends them as a new message, before passing the updated conversation to the LLM again.==
- From the usr's perpsective, it appears as if the LLM directly interacted with the tool, but in reality, it's the *agent* that handled the execution process in the background.

So how do we give tools to an LLM?
- We essentially use the system prompt to provide textual descriptions of available tools to the model:
![[Pasted image 20250417153857.png]]

For this to work, we have to be very precise and accurate about:
1. ==What the tool does==
2. ==What exact inputs it expects==
3. (Optionally, about what it returns, some examples of it in use, etc.)

Let's say we have a simple calculator tool:

```python
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
```

These details about the input and output interfaces (As well as the actual function description) are all important! Let's put them all into a text string:
```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

When we pass the previous string as part of the input to the LLM, the model will recognize it as a tool, and will (hopefully) know what it needs to pass as input and what to expect from the output.

Auto-formatting Tool sections
- The implementation of our tool (written in Python) actually already provided everything we needed!
- We could prove the Python source code as the SPECIFICATION of the tool for the LLM, but ==the way that the tool is implemented should not matter to the LLM==; All that matters should be its name, what it does, and the input/outputs it expects/provides.
- ==We can leverage Python's introspection features to leverage the source code and build a tool description automatically for us.==

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```
- Above: See that we've added the @tool decorator, which seems to add a to_string method to the function.


This generic Tool class can be reused whenever we need to use a tool:
(Note, this isn't the @tool decorator we used above; that tool decorator function RETURNS one of these.)
```python
class Tool:
    """
    A class representing a reusable piece of code (Tool).

    Attributes:
        name (str): Name of the tool.
        description (str): A textual description of what the tool does.
        func (callable): The function this tool wraps.
        arguments (list): A list of argument.
        outputs (str or list): The return type(s) of the wrapped function.
    """
    def __init__(self, name: str, description: str, func: callable, arguments: list, outputs: str):
        self.name = name
        self.description = description
        self.func = func
        self.arguments = arguments
        self.outputs = outputs

    def to_string(self) -> str:
        """
        Return a string representation of the tool,
        including its name, description, arguments, and outputs.
        """
        args_str = ", ".join([
            f"{arg_name}: {arg_type}" for arg_name, arg_type in self.arguments
        ])

        return (
            f"Tool Name: {self.name},"
            f" Description: {self.description},"
            f" Arguments: {args_str},"
            f" Outputs: {self.outputs}"
        )

    def __call__(self, *args, **kwargs):
        """
        Invoke the underlying function (callable) with provided arguments.
        """
        return self.func(*args, **kwargs)
```
Above:
- See that this basically leaves the `__call__` uninterrupted and adds a `to_string` method, which uses the available function propertie sto pretty-print a nice string.

It might seem complicated, but if we go slowly through it, we can see what it does. 

- **`   name`** (_str_): The name of the tool.
- **`description`** (_str_): A brief description of what the tool does.
- **`function`** (_callable_): The function the tool executes.
- **`arguments`** (_list_): The expected input parameters.
- **`outputs`** (_str_ or _list_): The expected outputs of the tool.
- **`__call__()`**: Calls the function when the tool instance is invoked.
- **`to_string()`**: Converts the tool’s attributes into a textual representation.

This can be used like:
```python
calculator_tool = Tool(
    "calculator",                   # name
    "Multiply two integers.",       # description
    calculator,                     # function to call
    [("a", "int"), ("b", "int")],   # inputs (names and types)
    "int",                          # output
)
```

But we can also use Python's `inspect` module to retrieve all the information for us!
This is what the @tool decorator does! 

Here's the definition of that ==@tool== decorator, which uses Python's introspection abilities through the `inspect` library to generate a Tool object that WRAPS the function!
```python
import inspect

def tool(func):
    """
    A decorator that creates a Tool instance from the given function.
    """
    # Get the function signature
    signature = inspect.signature(func)

    # Extract (param_name, param_annotation) pairs for inputs
    arguments = []
    for param in signature.parameters.values():
        annotation_name = (
            param.annotation.__name__
            if hasattr(param.annotation, '__name__')
            else str(param.annotation)
        )
        arguments.append((param.name, annotation_name))

    # Determine the return annotation
    return_annotation = signature.return_annotation
    if return_annotation is inspect._empty:
        outputs = "No return annotation"
    else:
        outputs = (
            return_annotation.__name__
            if hasattr(return_annotation, '__name__')
            else str(return_annotation)
        )

    # Use the function's docstring as the description (default if None)
    description = func.__doc__ or "No description provided."

    # The function name becomes the Tool name
    name = func.__name__

    # Return a new Tool instance
    return Tool(
        name=name,
        description=description,
        func=func,
        arguments=arguments,
        outputs=outputs
    )
```

Just to reiterate, we can then use this decorator like this:

```python
@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
```

And then we can use this `.to_string()` method to automatically retrieve a text suitable to be used as a tool description for an LLM!

```
Tool Name: calculator, Description: Multiply two integers., Arguments: a: int, b: int, Outputs: int
```

This is then INJECTED into the system prompt:
![[Pasted image 20250417155815.png]]


Note: 
[[Model Context Protocol]] (MCP): A unified tool interface:
- This is an open protocol that standardizes how applications provide tools to LLMs.
- MCP provides:
	- A growing list of pre-built integrations to your LLM that you can directly plug into
	- The flexibility to switch between LLM providers and vendors
	- Best practices for securing your data within your infrastructure.

This means that ==any framework implementing MCP can leverage tools defined in the protocol, eliminating the need to implement the same tool interfaces for each framework==.


Let's move onto Agent Workflows!

## The Thought-Action-Observation Cycle

Agents work in a continuous cycle of:
- Think
- Act
- Observe

The thought-action-observation cycle:
- These basically can work in a while-loop
- ==In many agent frameworks, the rules and guidelines are embedded directly into the system prompt, ensuring that every cycle adheres to a defined logic.==

In a simplified version, our system prompt might look like:
![[Pasted image 20250417160910.png]]

We see here that the System Message defines:
- The Agent's Behavior/Purpose
- The Tools that our Agent has access to
- A description of the Thought/Action/Observation cycle; how the agent should operate.

For a weather agent, whose job is to answer queries using a weather tool, this might look like:
- Thought:
	- "The user needs current weather information for New York. I have access to a tool that fetches weather data. First, I need to call the weather API to get up-to-date details."
- Act:
	- Alfred prepares (e.g.) a JSON-formatted command that calls the weather API tool:
```python
{
	"action": "get_weather":,
	"action_input": [
		{"location": "New York"}
	]
}
```
	- See aboev that we're specifying which tool to call (get_weather) and what parameter to pass (location: New York)
- Observe:
	- The agent gets feedback from the environment, receiving an observation.
		- "Current weather in New York: partly cloudy, 15C, 60% Humidity"
- ==Thought== (again)
	- Agent reflects on the returned observation, updating its internal reasoning
	- "Now that I have a weather data for New York, I can compile an answer for the user."
- Action:
	- Alfred then generates a final response, formatted as we told it to:
	- "I have the weather data now. The current weather in New York is partly cloudy with a temperature of 15C and 60% humidity."


So above, we can see:
- ==Agents iterate through a loop until hte objective is fulfilled.==
	- Thought, Action based on thought, Observation based on action, new Thought.
- ==Tool Integration==
	- The ability to call a tool (like a weather API) enables Alfred to go beyond static knowledge and retrieve real-time data, an essential aspect of many AI Agents.
- ==Dynamic Adaptation==
	- Each cycle allows the agent to incorporate fresh information (observations) into its reasoning (thoughut), ensuring that the final answer is well-informed and accurate.

Above is the core concept behind the [[ReAct]] cycle:
- The interplay of ==Thought==, ==Action==, and ==Observation== that empowers AI agents to solve complex tasks iteratively.



# Thought: Internal Reasoning and the ReAct approach

Thoughts enable agents to break down complex problems into smaller, more manageable steps, reflect on past experiences, and continuously adjust its plans based on new information.

![[Pasted image 20250417161831.png]]

[[ReAct]]: The ==concatenation of "Reasoning" (Think) with "Acting" (Act)==
- A simple prompting technique that appends "Let's think Step by Step" before letting the LLM decode the net tokens.
- Indeed, prompting the model to think "step by step" encourages the decoding process towards next tokens that generate a plan, rather than a final solution, since the model is encouraged to decompose the problem into subtasks.

![[Pasted image 20250417162044.png]]
Above:
- See that the "Let's think step by step" part guides the way in which the model generates, rather than providing examples of thinking step by step occurring. In this example, it looks like both Zero-Shot-CoT and Few-Shot-CoT both get the right answer.

> REACT PAPER: 
> In this work, we present ReAct, a ==general paradigm to combine reasoning and acting with language models== for solving diverse language reasoning and decision making tasks (Figure 1). ReAct prompts LLMs to ==generate both verbal reasoning traces and actions pertaining to a task in an interleaved manner==, which allows the model to perform dynamic reasoning to create, maintain, and adjust high-level plans for acting (reason to act), while also interact with the external environments (e.g. Wikipedia) to incorporate additional information into reasoning (act to reason).
> The idea of ReAct is simple: we augment the agent’s action space to Aˆ = A ∪ L, where L is the space of language. An action aˆt ∈ L in the language space, which we will refer to as a thought or a reasoning trace, does not affect the external environment, thus leading to no observation feedback. Instead, a thought aˆt aims to compose useful information by reasoning over the current context ct, and update the context ct+1 = (ct, aˆt) to support future reasoning or acting.
> Since decision making and reasoning capabilities are integrated into a large language model, ReAct enjoys several unique features: A) Intuitive and easy to design: Designing ReAct prompts is straightforward as human annotators just type down their thoughts in language on top of their actions taken. No ad-hoc format choice, thought design, or example selection is used in this paper. We detail prompt design for each task in Sections 3 and 4. B) General and flexible: Due to the flexible thought space and thought-action occurrence format, ReAct works for diverse tasks with distinct action spaces and reasoning needs, including but not limited to QA, fact verification, text game, and web navigation. C) Performant and robust: ReAct shows strong generalization to new task instances while learning solely from one to six in-context examples, consistently outperforming baselines with only reasoning or acting across different domains. We also show in Section 3 additional benefits when finetuning is enabled, and in Section 4 how ReAct performance is robust to prompt selections. D) Human aligned and controllable: ReAct promises an interpretable sequential decision making and reasoning process where humans can easily inspect reasoning and factual correctness. Moreover, humans can also control or correct the agent behavior on the go by thought editing, as shown in Figure 5 in Section 4.

![[Pasted image 20250417162756.png]]


>NOTE: Compare these types of strategies (prompting techniques like ReAct)  to O1-like things, where a model is trained to generate long CoT enclosed in \<thinking> tags using reinforcement earning. This is something different!



# Actions: Enabling hte Agent to Engage with its Environment
- Actions are the concrete steps that an AI agent takes to interact with their environment.
- Whether it's browsing the web for information or controlling a physical device, each action is a a deliberate operation executed by the agent.

Types of Agent Actions:
![[Pasted image 20250417162907.png]]

Actions themselves can serve many purposes:
![[Pasted image 20250417162928.png]]
Above: Not sure that this is the taxonomy that I would use...

==One crucial part of an agent is the **ability to STOP generating new tokens when an action is complete**==, and that is true for all formats of Agent: JSON, code, or function-calling. This prevents unintended output and ensures that the agent’s response is clear and precise.

The LLM only handles text and uses it to describe the action it wants to take and the parameters to supply to the tool.


#### ==The Stop and Parse Approach==
- This method ensures that the agent's output is structured and predictable

==Generation in a structured format==
- The agent outputs its intended action in a clear, predetermined format (JSON or Code)
==Halting further generation==
- ==Once the action is complete, the agent stops generating additional tokens== (meaning the agent stops invoking the model for next-tokens, and instead executes the function/code). This prevents extra or erroneous output.
==Parsing the output==
- An external parser reads the formatted action, determines which tool to call, and extracts the required parameters.

For example, an agent needing to check the weather might output:
```
Thought: I need to check the current weather for New York.
Action :
{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}
```
The framework can then easily parse the name of the function to call the arguments to apply!
This clear, machine-readable format minimizes errors and enables external tools to accurately process the agent's command.


### ==Code Agents==
- An alternative approach is to use code Agents; Instead of outputting a simple JSON object, a Code Agent generates an ==executable code block==, typically in a high-level language like Python.
- This has some advantages:
	- Expressiveness: Code can naturally represent complex logic, including loops, conditionals, and nested functions, providing greater flexibility than JSON.
	- Modularity and reusability: Generated code can include functions/modules that are reusable across different actions or tasks.
	- Enhanced debuggability
	- Direct integration: Can integrate directly with external libraries and APIs, enabling more complex operations like data processing or real-time decision making.

For example, a Code Agent tasked with fetching the weather might generate the following Python snippet:
```python
# Code Agent Example: Retrieve Weather Information
def get_weather(city):
    import requests
    api_url = f"https://api.weather.com/v1/location/{city}?apiKey=YOUR_API_KEY"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        return data.get("weather", "No weather information available")
    else:
        return "Error: Unable to fetch weather data."

# Execute the function and prepare the final answer
result = get_weather("New York")
final_answer = f"The current weather in New York is: {result}"
print(final_answer)
```

In this example, the Code Agent:
- Retrieves weather data via an API calll
- Processes the response
- Uses the print() function to output a final answer.

This method also follows the stop-and-parse-approach by clearly delimiting the code block and dsignaling when execution is complete (here, by printing the final_answer).



# Observe: Integrating Feedback to Reflect and Adapt
- Observations are how an agent perceives the consequences of its actions.
- Data from an API, error messages, system logs, etc. 
- These ==guide the next cycle of thought!==

In an observation phase, the agent:
- ==Collects feedback==, receiving data or confirmation of its action's outcome.
- ==Appends results==, Integrates the new information into its existing context, effectively updating its memory.
- ==Adapts its strategy==: Uses this updated context to refine subsequent thoughts and actions.

If a weather API returns the data "Partly cloudy, 15C, 60% humidity," then ==this observation is appended to the agent memory==. The ==agent then uses it to decide whether additional information is needed or if it's ready to provide a final answer==.

==The iterative incorporation of feedback ensures the agent remains dynamically aligned with its goals==, constantly learning and adjusting based on real-world outcomes.

These observations can take many forms, from reading webpage text to monitoring a robot's arm position.
==This can be seen like Tool "log" that provide textual feedback of the Action execution==.

![[Pasted image 20250417164203.png]]

==How are the results appended?==
1. Parse the action to identify the functions to call and arguments to use
2. Execute the action
3. Append the result as an Observation







