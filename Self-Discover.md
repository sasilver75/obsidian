February 6, 2024
USC, [[DeepMind]], authors include [[Quoc Le]]
Paper: [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620)
#zotero 
Takeaway: ...

Significance: I'd heard this referred to as the most powerful prompting technique out there, by the guy who added it to the [[DSPy]] library. The paper sure makes it seem like that's the case.

---

Notes:
- 1) Introduction
	- Authors note [[Chain of Thought]], decomposition-based prompting, [[Step-Back Prompting]] ... but note that a fundamental limitation is that each technique itself serves as an atomic reasoning module that makes an *implicit prior assumption* of the correct process to tackle a given task.
	- Self-Discover composes a coherent reasoning structure appropriate to a task by selecting from a set of atomic 39 reasoning modules, using three actions to guide the LLM to generate a reasoning structure for the task, and then, during final decoding, following the self-discovered structure to arrive at the final answer.
	- We test Self-Discover on 25 challenging reasoning tasks including [[BIG-Bench Hard]], Thinking for Doing (T4D), and MATH, and see that Self-Discover outperforms CoT on 21/25 tasks, with performance gains of up to 42% -- it's even superior to inference-heavy methods like [[Chain of Thought|CoT]]+[[Self-Consistency]], while requiring 10-40x less inference compute.
- 2) Self-Discovering Reasoning Structures for Problem-Solving
	- When humans face a new problem, we first search internally what knowledge and skills from our prior experience might be helpful to solve it, then we apply these (multiple) relevant knowledge and skills to the task.
		- Given a task and a set of high-level problem-solving heuristics like *"Use critical thinking",* *"Let's think step by step,"* ==Stage 1== of Self-Discover aims to uncover the intrinsic reasoning structure for solving this task, via meta-reasoning. We use three meta-prompts to guide LLMs to select, adapt, and implement an actionable reasoning structure with no labels/training required. We format this structure in key-value pairs, similar to JSON due to interpretability and findings that following JSON boosts reasoning and generation quality. Stage 1 operates on a *task level*, meaning that we only need to run Self-Discover once for each task.
		- In ==Stage 2==, we can simply use the discovered reasoning structure to solve every instance of the given task by instructing models to follow the provided structure by filling each key, and arriving at the final answer.
	- ==Stage 1==
		- The first stage consists of three actions (See Fig 3.):
			1. ==SELECT==: Relevant reasoning modules for task-solving are chosen from the set of reasoning module descriptions. For example, "*reflective thinking*" might help search for first-principle theories on science problems, while "*creative thinking*" helps on generating a novel continuation to a story. Given a set of reasoning module descriptions and a few task examples, we select a subset of reasoning modules useful for solving tasks by using a model and a meta-prompt.
			2. ==ADAPT==:  Descriptions of selected reasoning modules are *rephrased* to be more specific to the task at hand. For example, we might translate *"break the problem into subproblems"* into *"calculate each arithmetic operation in order*, for an arithmetic problem. Given selected reasoning modules from previous step, rephrase each to be more specific to our task using a meta-prompt and a model.
			3. ==IMPLEMENT==: Adapted reasoning descriptions are implemented into a structured, actionable plan so that the task can be solved by following the structure. Given the adapted reasoning module descriptions, we operationalize the reasoning modules into an implemented *reasoning structure,* with specified instruction on what to generate for each step. Uses a meta-prompt and a demonstration of a human-written reasoning structure on *another* task to better-convert the natural language descriptions into a reasoning structure. 
	- ==Stage 2==
		- After three stages, we have an implemented reasoning structure uniquely adapted for the task we need to solve; then we can simply append the reasoning structure to all instances of the task, and prompt models to follow the reasoning structure to generate an answer A.
- 3) Experiment Setup
	- Compares against zero-shot prompting methods (Direct Prompting, CoT, Plan-and-Solve), as well as (CoT-Self-Consistency, Majority voting of reach reasoning modules, and best of each reasoning module).
- 4) Results
	- Does Self-Discover Improve LLM Reasoning?
		- On aggregated 23 tasks of [[BIG-Bench Hard]], Self-Discover achieves 7% and 6% absolute improvement on PaLM 2-L over CoT and Plan-and-Solve, respectively. Similar gains for GPT-4.
	- Which Types of Problems do Self-Discover Help the Most?
		- Performs best on tasks that require *diverse world knowledge,* such as sports understanding, movie recommendation, and ruin names.
	- How Efficient is Self-Discover?
		- Achieves better performance while requiring 10-40x fewer inference compute compared to self-consistency or majority voting.
- ...
- 6) Related Work
	- Prompting Methods
		- CoT, Least-to-most prompting, decomposed prompting, reframing, help me think prompting, stepback prompting, and search-based approaches like [[Tree of Thoughts]], Graph-of-thought, Branch-solve-merge, RAP
	- Reasoning and Planning
		- The development of various reasoning/planning benchmarks like GSM8K and BigBench prompted methods to improve model performance. Chain of Thought, Scratchpad, question summarization, question decomposition, program generation, etc. 
		- Our work in Self-Discover allows models to combine multiple reasoning approaches by self-composing into a structure without the need to access task labels.
		- Related work includes SkiC, devising a strategy, and planning with iterative querying, but these require human-annotating skills and reasoning plans while self-discover leverages a scalable solution with the help of LLM's meta-task reasoning capabilities.
		- ((This might be an interesting place to find a bunch of papers to read on prompting techniques, if that's something that I'd like to learn more about.))

Abstract
> We introduce ==SELF-DISCOVER==, a ==general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems== that are challenging for typical prompting methods. Core to the framework is a self-discovery process where LLMs ==select== multiple ==atomic reasoning modules== such as critical thinking and step-by-step thinking, and ==compose them into an explicit reasoning structure for LLMs to follow== during decoding. SELF-DISCOVER substantially improves GPT-4 and PaLM 2's performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, by as much as ==32%== compared to Chain of Thought (CoT). Furthermore, SELF-DISCOVER ==outperforms inference-intensive methods such as CoT-Self-Consistency== by more than ==20%==, while requiring ==10-40x== fewer inference compute. Finally, we show that the self-discovered reasoning structures are universally applicable across model families: from PaLM 2-L to GPT-4, and from GPT-4 to Llama2, and share commonalities with human reasoning patterns.

# Paper Figures
![[Pasted image 20240520114618.png]]

![[Pasted image 20240520115150.png]]

![[Pasted image 20240520121930.png]]
Above: The three actions of the first stage of Self-Discover include:
- Select: Select relevant modules to our task
- Adapt: Adapt these modules' descriptions to our specific task.
- Implement: Operationalize the adapted modules into a reasoning structure specific to our task.

![[Pasted image 20240520122732.png]]
Above: Showing that Self-Discover is an efficient method. 
- Best of each RM (reasoning module) is a method that assumes that we have access to oracle labels, and uses the highest accuracy from applying each RM. We compare with this to examine whether self-discover competes with methods that depend on perfect prior knowledge of which RM to use on a new task.

![[Pasted image 20240520123105.png]]

![[Pasted image 20240520123213.png]]

![[Pasted image 20240520124752.png]]
Above: Stage 1 of Self-Discover. Includes the meta-prompts used for Select, Adapt, and Implement. Interesting that these seem to be pretty bare-bones, in my opinion.

![[Pasted image 20240520124806.png]]
Above: All 39 reasoning modules used in Self-Discover, adopted from (Fernando et al. 2023)