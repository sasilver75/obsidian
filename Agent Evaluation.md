References:
- Cameron R Wolfe: [Agent Evaluation, a Detailed Guide](https://cameronrwolfe.substack.com/p/agent-evals)


Previous LLM evaluation used [[Benchmark]]s composed of static questions or short, often single-turn conversations, but now we have agent systems that operate over long time horizons and interact with the environment.

Due to their complexity and autonomous, to accurately measure the capabilities of an agent system, we must build harnesses that are realistic and capable of testing agent systems similarly to how they're used in practice.

Agents are typically composed of:
- The underlying [[Large Language Model|LLM]]
- Tools for the agent to use
	- APIs, CLIs, or [[Model Context Protocol|MCP Server]]s
	- If we want our agent to reserve a table for us at a local restaurant, we can simply teach the model how to craft a call to the OpenTable API. Tool calls are handled by creating a special set of tokens related to tool calling, and teaching the LLM how to use these tokens.
- Instructions for the agent
	- Should be as clear as possible, clarify edge cases via explicit rules or concrete examples, and specify the exact actions specified of the agent. Should explain the problem being solved, as well as the best solution for solving the problem.
	- Should strike a balance between simplicity and specificity. We want to reliably guide agent behavior, but not give it overly-complex, brittle, and difficult-to-maintain instructions.
	- Can use [[Automatic Prompt Optimizer]] tools (e.g. [[DSPy]]) to iterate on instructions and improve their clarity.

![[Pasted image 20260603143945.png]]
Above: Special Tool-calling tokens.

For [[Qwen 3]] models, there are several [[Extensible Markup Language|XML]]-style tags that are stored as special tokens in the tokenizer:
- `<tool>` and `</tool>`: Used to encapsulate tool definitions
- `<tool_call>` and `</tool_call>`: Used to encapsulate a specific tool call, including the tool to be invoked and associated parameters
- `<tool_response>` and `</tool_response>`: Used to encapsulate the result or response from the tool that is called

In a model's instruction template, we define available tools and the expected format for calling them in the system message, which provides necessary context for the model to:
1. Understand what tools are available
2. Construct valid calls to these tools

Qwen 3:
(Note: "im_start" refers to "input message start")
```
<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:

<tools>
{
  "type": "function",
  "function": {
    "name": "search_web",
    "description": "Search the web for information.",
    "parameters": {
      "type": "object",
      "properties": {
        "query": {
          "type": "string",
          "description": "The search query."
        }
      },
      "required": ["query"]
    }
  }
}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:

<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>

<|im_start|>user
Who was the first president of the United States?
<|im_end|>

<|im_start|>assistant
<tool_call>
{
  "name": "search_web",
  "arguments": {
    "query": "first president of the United States"
  }
}
</tool_call>
<|im_end|>

<|im_start|>user
<tool_response>
{
  "result": "George Washington was the first president of the United States. He served from 1789 to 1797."
}
</tool_response>
<|im_end|>

<|im_start|>assistant
<think>
... Thinking trace goes here ...
</think>

The first president of the United States was George Washington. He served as president from 1789 to 1797.
<|im_end|>
```
Above: See in Qwen3 that the tool response is part of a user turn, rather than a separate persona. Interesting. See also the use of \<think> tags in the assistant's last turn, though it didn't use them before making a tool call in their previous turn.

Creating useful tools that can be easily understood and correctly invoked by agents is an art!
- Tools should be well-documented
- Tools should have a clear purpose
- Tools should overlap minimally with other tools
- Tools should recover gracefully from errors.

There are entire benchmarks (e.g. [[Berkeley Function-Calling Leaderboard]] (BFCL)) about calling tools, where performance is measured using a variety of different metrics:
1. ==Invocation accuracy==: Measures ==whether the LLM correctly decided to call a tool== when it should, or avoided calling one when it should not.
2. ==Selection accuracy==: ==Whether the LLM called the correct tools==, usually by keeping track of a ground-truth trajectory that includes a list of necessary tools for solving a particular problem.
3. ==Structural accuracy== and ==Schema Validity==: ==Whether the structure of the cool call was correct==.
4. ==Trajectory accuracy==: Looks at the sequence of tool calls made by the model when solving a problem and compares them to ground truth in some way (e.g. correct call order, correct selection, using unnecessary tools, and more).

We can also used ==outcome-oriented evaluation== by focusing on whether the LLM's final answer is correct, instead of looking at the tools it uses to produce an answer.

Agents must also be able to:
- Decompose difficult problems into smaller, simpler parts and solve each of those parts.
- Be able to self-reflect and recover from its won mistakes while solving problems.


Sometimes we have [[Multi-Agent System]]s (though they bring complexity that means you should always start with single-agent design and optimize that basic setup as much as possible) to evaluate.
- Can be useful when you have a task whose subparts are easily logically separable and parallelizable.
- Can be useful when the instructions for your single-agent system are bloated, and the agent is struggling to follow instructions even with clear logic or templating, or when tools are being selected incorrectly the agent, because there are too many available tools, and when the existing tools have similar or overlapping purposes/specifications.

Types of multi-agent systems:
- ==Manager setups==: A central "manager" agent orchestrates specialized agents via tool calls, where each agent handles a specific task or domain.
	- ![[Pasted image 20260603151651.png]]
	- Uses the concept of a [[Subagent]]; rather than keep all state for a task in a single agent, we can trigger a subagent (typically given limited context) to handle a limited task component (e.g. exploring code, retrieving information) and return relevant context to the main agent.
- ==Decentralized setups==: Multiple agents operate as peers by handing off tasks to one another based on their respective specializations, without the use of a centralized agent to orchestrate task execution. Instead, control is passed among multiple agents that operate as peers. Agents pass control to one another via tool or function calls.
	- ![[Pasted image 20260603151831.png]]
	- We see an agentic support system that uses an agent to triage the issue, then passes control to a specialized agent to solve that particular type of issue. In this case, that specialized agent doesn't return results back to the triaging agent, it just solves the problem, whatever that means.
	- We should design our systems in a modular fashion, ensuring that the addition of new agents to the system can be accomplished without extensive changes to the remaining agents.
		- Keep components flexible, composable, and driven by clear, well-structured prompts.


[[Context Engineering]] is a term that's arisen more and more as we have these agents that handle long-horizoned tasks, burdened with dozens of tools, tool responses, etc.
![[Pasted image 20260603152239.png]]
- [[Context Rot]] is here a universal problem; as the number of tokens in an agent's context grows, models are less capable of accurately recalling information.
	- ==For this reason, we must actively refine the contents of an agent's context in order for the agent to continue performing well.==

> “We view context engineering as the natural progression of prompt engineering. Prompt engineering refers to methods for writing and organizing LLM instructions for optimal outcomes. Context engineering refers to the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference, including all the other information that may land there outside of the prompts.”

For conventional LLM use, we often load static context in the model using techniques like [[Retrieval-Augmented Generation]] (RAG), which retrieves chunks of information concatenated into the LLM's context window. Typically a static approach; the model receives all necessary information for answering a question before even attempting to answer the question!
- Instead, we can do a more dynamic [[Agentic RAG]]/Agentic Search approach, which lets the agent explore and dynamically decide when new context is necessary.

[[Compaction]]: Unlike conventional LLMs, agents run for multiple turns to solve long-horizoned tasks, across which it will reason, call tools, self-reflect, and more, expanding its context. Compaction is the most common category of techniques to mange this growing context:
- ==Summarization==: Generating a summary of long conversation with an LLM and reinitializing a fresh context window with this shorter summary.
- ==Tool Result Clearing==: Removing the output payload of older tool calls within the agents' conversation history.
- ==Note-Taking==: Writing notes to an external data store (e.g. markdown file) that can later be accessed by the agent when needed.
	- (Not just a compaction technique; agents can maintain notes while solving a problem even without performing compaction, e.g. creating a to-do list to plan a solution and track progress).


[[Agent Harness]]: The system surrounding the agent (tools, instructions, environment, problem-solving strategy, and more) that enables the model to act as an agent:
- Processes inputs
- Orchestrates tool calls
- Return results

The Harness/Scaffold may change the following:
- Agent's interface with the environment
- Prompting strategy for the agent; scaffold may tell the agent to inspect files before editing, break problems down into steps, check its own work before completing, and more.
- The tools available to the agent and their respective interfaces
- The structure of the agent systems (e.g. the agent may leverage sub-agents or a multi-agent setup for different parts of a solution)
- The context management strategy used by the agent.

The Harness has a huge impact on agent performance; ==When we evaluate an agent,we evaluate the ability of the model and scaffold to work together to solve a task.== 
- ((We also know that since late 2025, models are being trained *in their proprietaray harnesses*, e.g. [[Claude Code]] or [[OpenAI Codex|Codex]]. Models and harnesses are merging into single systems, effectively, for some models))


# Common Patterns in Agent Evaluation
- An evaluation is a set of tests that provide inputs to our agent system, collect outputs, and apply grading logic to determine whether the agent was successful.
- We evaluate an agent system by creating an evaluation suite of diverse tests that reflect the ways that the agent can be used in the real world.

Agent evaluations are different from single-turn benchmarks used for LLM evaluations traditionally.
![[Pasted image 20260603153924.png]]
Above:
- The evaluation consists of severals ==tasks==, or individual test cases
- Each attempt at solving a task is called a ==trial==, and we often run several trials for each task to ensure consistent results.
- As the agent complete a trail, it produces a ==transcript/trace/trajectory== that includes outputs, tool calls, reasoning steps, intermediate outputs, and any other interacts from the agent during the trial.
- The final state of the external environment after a trial completes is referred to as the ==outcome== for that trial.
- A ==grader== is used to evaluate the results of a trial by applying specific checks that verify aspects of a grader's transcript *and/or* outcome.

We usually compute performance metrics by aggregating across both tasks and trials.

## Types of Graders
- Graders take a trajectory and outcome of a trial as input and performa quality checks over some portion of these assets.
- ==Human evaluation is the difinitive source of truth for measuring quality==, so many projects start with simple human evaluations (manual inspection, vibe checks) as a first step.
	- Slow and expensive.
	- We often create guidelines or a rubric to define key quality dimensions and how those should be evaluated (e.g. pass-fail, or [[Likert Score]]s).
	- These can enable fast iteration before investing in creating a more comprehensive evaluation suite.
	- We want the results of human evaluation to be accurate and consistent, as judged by the level of agreement between humans ([[Inter-Annotator Agreement]]). Unfortunately, ==human evaluators rarely agree with eachother without investing effort into refining the annotation guidelines and process==.
- ==Automatic evaluation techniques== are later adopted to make experimentation more efficient. Some tasks can be deterministically verified (e.g. using a Python function), while other tasks require an open-ended [[LLM-as-a-Judge]] approach.
	- Can be categorized into two groups:
		- ==Code-based==: Heuristic checks that can be captured in Python functions. String matching, assertions on the outcome or transcript, use of traditional evaluation metrics (e.g. [[ROUGE]], [[BLEU]]). Oftentimes efficient, reproducible, and easier to debug, but often reference-based (requiring a ground-truth) and inflexible/lacking nuance, making them somewhat limited when trying to capture subjective aspects of agent behavior.
		- ==Model-based==: Open-ended checks that are based on an LLM judge prompted to evaluate some aspect of quality for us. Universally used in the evaluation process for both conventional LLMs and agent systems. 
			- Several scoring setups have emerged:
				- ==Pairwise (preference) scoring==: The judge is presented with a prompt and two model responses and asked to identify the better response.
				- ==Direct assessment (pointwise) scoring==: The judge is given a single response to a prompt and asked to assign a score, e.g. using a 1-5 [[Likert Score]].
				- ==Reference-guided scoring==: Judge is given a golden reference response in addition to the prompt and candidate responses to help with scoring.
			- Despite their effectiveness, LLM judges aren't perfect, subject to several well-known biases, and more expensive than code-based graders. It is better to use multiple grading techniques together.
				- We can calibrate model-based graders based on their agreement with human evaluation, and use these models to run more efficient evaluations at scale.

![[Pasted image 20260603155151.png]]

![[Pasted image 20260603155942.png]]

![[Pasted image 20260603160114.png]]

![[Pasted image 20260603160231.png]]

![[Pasted image 20260603160451.png]]
























