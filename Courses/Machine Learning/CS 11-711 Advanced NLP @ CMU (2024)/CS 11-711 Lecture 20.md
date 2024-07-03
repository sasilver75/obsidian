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
- TRoVE: Inducing Verifiable and Efficient Toolboxes for Solving Programmatic Tasks. (This is the speaker's research paper)
	- Given a natural language program, ask the LM for a program, execute the result... in the simple way 

(The audio is so bad that I can't hear what she's saying)


## Evaluation, Empirical Benefits, Future Directions
- How do we currently evaluate tool use? Current benchmarks are in two categories:
	- Reusing existing datasets (reasoning), but using tool-augmented approaches:
		- Text: math, BigBench
		- Structured data: table, KG (WikiTable)
		- Other modalities, Image
	- Aggregated API benchmarks
		- Tasks that necessitate tools
		- Issues :(
			- Naturalness: If you look deeper into how they create examples, they usually just heuristically select... one or more APIs, and then ask (eg) GPT examples to synthesize examples of using these APIs. The selected tools may not actually be used together in practice, and might not reflect a natural use case.
			- Executability: You probably think that tools can be executed because they're programs, right? But for half the datasets, the tools aren't even executable, they used synthesized outputs.
		- ![[Pasted image 20240618133511.png]]
		- 

![[Pasted image 20240618134139.png]]
- In addition to task performance, latency/cost/reliability are important to consider as well!
![[Pasted image 20240618134224.png]]


---

# Part 2: Language Models as Agents (Frank Xu)

![[Pasted image 20240618134303.png]]
Agents are anything that can be viewed as perceiving its environment through sensor and acting upon that environment through actuators.
- The agent might have abilities, knowledge, goals, preferences, etc.

We can use LLMs as the agents themselves, and tools can both be used as perceptors or actuators!

To get started on LM agents, let's get started on four stages:
1. Tasks and applciations
2. TRaining-free methods for building agents (eg using API-based models)
3. Evaluations Environment and Benchmark
4. Training methods for improving agents

---

Why do we want agents?
- Imagine if things can be done just by talking, like a human?
![[Pasted image 20240618134607.png]]
Many of us are using Github Copilot plugins to help us write code: "Sort my list in ascending order."
- Natural language instructions ->
	- Robot actions
	- Play minecraft
	- Shoot asteroid in a game
	- Software development

Agents need to be able to take in observations of the current environment:
- Text input (You are in the middle of a room, around you are...)
- Visual input (screenshot of your game frames)
- Audio input (audio in the game)
- Structured input (eg JSON; your inventory, or the HTML tree of the site you're navigating)
This highlight a need for multimodal models

![[Pasted image 20240618135346.png]]
It's not enough to just have planning/reasoning, though, because you can't actually effect the world around you!

This is why it's useful for our language models to have a tool-use ability. ([[Toolformer]], [[Gorilla]], [[ReAct]])
![[Pasted image 20240618135523.png]]
By interacting with the environment (via generating and executing appropriate API calls with arguments), we change it; we can then re-observe the environment and again reason about what the next appropriate step should be.

![[Pasted image 20240618140015.png]]
![[Pasted image 20240618140029.png]]

![[Pasted image 20240618140148.png]]
Sometimes LMs just don't know which section of the site to click
![[Pasted image 20240618140244.png]]

![[Pasted image 20240618140259.png]]
Failures due to repeated typing 🤔

![[Pasted image 20240618140338.png]]
I mean "Sam Silver," not "myself!"

So how do we improve our agents?
- In-context Learning - Learning from few-shot exemplars
- Supervised Finetuning - Learning from Expert trajectories
	- Data hungry, can't learn much from failed trajectories (we only learn from positive examples in SFT)
- Reinforcement Learning - Learning from the environment




