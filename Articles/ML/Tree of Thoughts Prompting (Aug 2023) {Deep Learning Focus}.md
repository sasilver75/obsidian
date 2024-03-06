#article 
Link: https://cameronrwolfe.substack.com/p/tree-of-thoughts-prompting
Read: Mar 5, 2024

Related Post:
[[Chain of Thought Prompting for LLMs (Apr 2023) {Deep Learning Focus Newsletter}]]

--------

![[Pasted image 20240305230526.png]]

As LLMs first started to gain in popularity, they were criticized for their shortcomings in solving complex, reasoning-based problems. Although scaling up these models provided a near-uniform boost in performance, we saw virtually no boost in performance on reasoning-based tasks with modern LLMs.

This changed with the proposal of advanced prompting techniques, such as [[Chain of Thought]] prompting and [[Self-Consistency]] -- these methods showed us that LLMs are capable of "reasoning" and solving complex, multi-step problems!

> *"It's perhaps surprising that underlying all this progress is still the original autoregressive mechanism for generating text, which makes token-level decisions one by one and in a left-to-right fashion."*

In the [[Tree of Thought]] prompting approach, we solve problems by explicitly decomposing them into a series of *thoughts*, or intermediate steps.
- Similar to [[Chain of Thought]] prompting, ==tree of thoughts prompting generates a solution that is simply a sequence of individual thoughts.==
- However, this approach goes further by ==allowing multiple reasoning paths to be considered at once== -- *forming a tree of potential thoughts or reasoning paths* -- and explores this entire solution space via LLM-powered self-evaluation.
- With Tree of Thoughts, the LLM can deliberately plan its solution, test various intermediate reasoning paths, and even perform backtracking -- allowing the model to explore the solution space and eventually generate the correct output.


## Connections to Research in Other Fields and Generations

> *A genuine problem-solving process involves the repeated use of available information to initiate exploration, which discloses, in turn, more information until a way to attain the solution is finally discovered."*

Humans are sometimes said to have two separate modes of making decisions:
1. A fast, automatic, unconscious mode ("System One thinking")
2. A slow, deliberate, conscious mode ("System Two thinking")

Authors argue that techniques like *chain-of-thought prompting* seem to mimic the *first* mode outlined above, as the LLM just generates text in a left-to-right manner without deliberate planning or deconstruction of the problem.

The goal of *Tree of Thought* prompting is to inject deliberate *planning and exploration* into the problem space, to mimic the *second* type of thinking.


# The Basics of Prompting

The generic text-to-text format of LLMs is incredibly powerful. To solve any problem we can simply:
1. Write a textual prompt that describes the problem
2. Generate a relevant output/solution with the language model.

==However the effectiveness of [[In-Context Learning]] is highly related to the prompt that is used to solve a particular 
problem!==

Let's look at the basics of prompt engineering, to provide some useful context:

#### What is prompt engineering?
- Prompt engineering refers to the process of iteratively tweaking a prompt for a language model with the goal of discovering a prompt that accurately solves a desired task.
- Typically the process is empirical and full of heuristics.
- Context windows
	- A major consideration ((at the time of writing)) when writing a prompt is the size of the underlying LLM's context window -- we need to be selective about the data that's included in a prompt.

A variety of prompting techniques exist, but each of these techniques utilize a (relatively) common structure.
- ==Input Data==: Data being processed by LLMs
- ==Exemplars==: Input/Output examples demonstrating a correctly solved problem
- ==Instruction==: A detailed, textual description of the LLM's expected behavior
- ==Indicators==: Textual tags that are used to organize and structure the different components of the prompt
- ==Context==:Any extra context that might be useful to provide the LLM ((eg retrieved in a RAG context))


## Hierarchy of Prompting Techniques


#### 1) Zero and Few-Shot Prompting
- Zero-shot prompting is one of the simplest techniques -- to form a zero shot prompt, we have:
	1. A task description
	2. Our input data
- ((Note: Zero-Shot Prompting is *not* Zero-Shot CoT))
> *"Translate the following from English to French: "I'd like some water, please!"*

#### 2) Few-Shot Prompting
- Goes beyond zero-stop prompting by adding ==exemplars== of the model's desired output to the prompt.
- In addition to a task description, we provide several examples of correct input/output pairs.
- This technique was popularized with GPT-3.
> *"
> Translate English to French:
> sea otter => loutre de mer
> peppermint => menthe poivrée
> plush giraffe => girafe peluche
> cheese =>
> "*

#### 3) Instruction Prompting
- Instead of demonstrating correct behavior via a task description and several examples of correct output, instruction prompting includes a detailed *instruction*, or *explanation* *of the correct behavior*, within the prompt that is given to the language model!
>*"
>## Instruction ##
>You're a brilliant mathematician that can solve any problem in the world. Attempt to solve the problem below.
>## Question ##
>What is 12 * 10 / 2 + 5?
>## Response ##
>
>## Instruction ##
  Answer all questions by responding with another question in the style of Shakespeare
  >## Question##
  >What is the meaning of life?
  >## Response##
  >
>"*

Instruction prompting and few-shot prompting are not mutually exclusive -- ==we can easily combine an instruction prompt with several few-shot exemplars to yield improved performance== (in fact, the task descriptions used by zero and few-shot prompting techniques are actually quite similar to an instruction anyways.)
### Advanced Prompting Techniques

#### 4) Chain of Thought (CoT) prompting
- CoT prompting encourages a language model to more efficiently solve complex problems by outputting, along with its solution a *chain of thought* -- step-by-step explanation of how the problem was solved!
- The CoT technique is most effective when the map from input to output is highly non-trivial, like in math or multi-step reasoning problems.

> *"
> # Exemplar showing CoT response
> Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he now have?
> A: Roger started with 5 balls. 2 cans of 3 tennis balls is 6 tennis balls. 5 + 6 = 11. The answer is 11.
>
> # Actual 
>Q: The Cafeteria has 23 apples. If they used 20 to make lunch and bought 6 more, how manny apples do they have? 
> "*

Variants of CoT
- Zero-Shot CoT
	- Instead of providing an exemplar showing CoT examples, it instead just asks the model to "think step by step."
- Least-to-most prompting
	- Decompose the problem into subproblems, then sequentially solve the subproblems.
- [[Self-Consistency]]
	- Generate the output several different times, and the final answer is generated by taking a majority vote of the model's outputs.

==Techniques like CoT prompting follow a left-to-right, continuous generation attempt that uses next-token prediction to output a solution in a single attempt!== ((It's high-performance "system 1" thinking))

These solutions still fail to solve tasks that require strategic planning and exploration, though! (("System 2" thinking))
This is where Tree of Thought comes in!

# Understanding Tree of Thoughts prompting 🌳
- Tree of Thought allows us to break problems down into smaller parts (i.e. like a chain of thought), but goes *further* by allowing us to explore multiple solution paths in parallel, forming a tree 🌳 instead of a chain  ⛓️!
![[Pasted image 20240306112834.png]]

[[Tree of Thought]] (ToT) prompting breaks down a problem into a sequence of smaller steps that are solved individually -- but it doesn't constrain the model to output these steps all at once!
1. Explore multiple choices for each problem-solving thought
2. Evaluate whether certain thoughts bring the model closer to a final solution
3. Perform backtracking when certain thoughts are found to be a dead end
4. Search over a combinatorial space of possible problem-solving steps to find the best final solution.


During exploration an LLM can evaluate progress made by each thought towards a final solution using a *language-based process.*
Then, by leveraging widely-used search algorithms (eg breadth-first search or depth-first search), ToT prompting can be augmented with lookahead and backtracking techniques, allowing the solutions space of any problem to be thoroughly explored!

![[Pasted image 20240306113255.png]]

What does the tree represent?
- When using ToT, we explore several paths, each composed of individual *thoughts that represent potential solutions to a problem.*
- A node in this tree is simply a partial solution (or thought) to our problem, while each connection is an operator that modifies this partial solution to yield the next thought in a problem-solving path.

![[Pasted image 20240306113408.png]]

#### Tree of Thoughts Problem Solving Framework
- How do we actually use this in a practical application?
- The implementation looks a bit different depending on the problem we're trying to solve, but any instantiation of ToT prompting has to concretely define *four* problem solving components:
![[Pasted image 20240306113459.png]]
Thought decomposition
- Unlike CoT, ToT explicitly decomposes a problem into intermediate steps or thoughts, which are then combined into a solution to the underlying problem.
- Depending on the problem, this can take different forms -- like outputting a few words, or a line of equation.
- The definition of thought (Above) is shown to be different for different types of problem.

There are two basic techniques proposed for thought generation:
1. ==Sampling==: Generating several thoughts independently with the same prompt
	- Works well when the thought space is rich, as several independently-generated thoughts are unlikely to be duplicates.
2. ==Proposing==: Generating several thoughts sequentially with a "propose" prompt
	- Works well when the thought space is more constrained.

State evaluation
- To be able to choose which chains of thought are interesting, we have to be able to evaluate the quality of a (chain of) thought.
- Two strategies are proposed for thought evaluation:
	1. ==Value==: Assign some scalar value (eg 1-10) or classification (sure, likely, impossible) to each state.
	2. ==Vote==: Compare different solutions and select the one that's more promising
		- Best when the solution to a problem is hard to directly value (eg creative writing tasks)

Search Algorithm
- The final component of ToT prompting is the search algorithm used to explore the solution space.
- We use BFS and DFS with formulations below::

![[Pasted image 20240306120006.png]]

































