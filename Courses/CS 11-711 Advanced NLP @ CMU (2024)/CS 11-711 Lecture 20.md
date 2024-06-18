# Topic: Tool Use and Language Agents
https://www.youtube.com/watch?v=d0QSnLjlgzc&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=18

----

LMs are powerful for text generation tasks, but they struggle with complex reasoning, and they're fundamentally unable to access real-world information.

![[Pasted image 20240617184018.png]]

![[Pasted image 20240617184259.png]]
- ART (Software tools)
- ToolLLM (APIs as tools)
- Gorilla/HuggingGPT (Neural models as tools from HF hub)
- TRoVE, Voyager (Expert Crafted Functions)

So there's a lot of diversity in terms of tool use:

We'll cover:
- Tool Basics: Definition and functionality
- Scenarios: Tools, tasks, methods
- Evaluation/empirical benefit/future directions

---

## Tool Basics: Definition
- We think that tools are actually programs that language models can leverage, and call the program to do some function. 
- For a program to be a tool, it needs to satisfy two properties:
	- It needs to be external to our core language model
	- It needs to be functional; it needs to be something that can be applied to other objects in the environment to change to environment. If the environment is a blank canvas, a tool might be a paint brush

Tool Use ==Definition==:
> "An LM-used tool is a function interface to a computer program that runs external to the LM, where the LM generates the function call and input arguments in order to use the tool."

## Tool Basics: Functionality
- What are the main functionalities of tools?
	- ==Perception Tools==: Used to collect data from the environment, without changing the environment state 
		- A search engine
	- ==Action Tools==: Used to exert actions in the environment, changing the environment's state.
	- ==Computation Tools==: General acts of computing, including translation, calculators, etc.
- These categories are not exclusive/disjoint; a tool might have one or more functionalities.
	- Many tools are likely perception and computation tools, for instance, if the search process involves some meaningful computation.

## Basic Tool-use Paradigm
- How do Language Models use tools?
- In a nutshell, it's basically a shift between text generation and tool execution mode:
![[Pasted image 20240617185506.png]]
- After the check_weather() expression is completed, it triggers a request to some weather server. This call will be executed and the result will be returned back to the language model; the LM replaces hte API call by this returned execution output and continues in text generation mode.
	- ((I don't like the tool getting to dictate what the next token is! This is bad separation of concerns! The LM should be the one that decides the next token based on the returned results of the tool!))
- Our models can either/or:
	- Learn to use the tools by training on examples of tools being used.
	- Receive instructions on how to use the tools via the prompt/in-context learning.

![[Pasted image 20240617185545.png]]

What if our tools are unavailable?
- TRoVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks.
	- Given a natural language program, ask the LM for a program, execute the result... in the simple way 



