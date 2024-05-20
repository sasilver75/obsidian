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
	- Self-Discover composes a coherent reasoning structure appropriate to a task by selecting from a set of atomic reasoning modules, using three actions to guide the LLM to generate a reasoning structure for the task, and then, during final decoding, following the self-discovered structure to arrive at the final answer.
	- We test Self-Discover on 25 challenging reasoning tasks including [[BIG-Bench Hard]], Thinking for Doing (T4D), and MATH, and see that Self-Discover outperforms CoT on 21/25 tasks, with performance gains of up to 42% -- it's even superior to inference-heavy methods like [[Chain of Thought|CoT]]+[[Self-Consistency]], while requiring 10-40x less inference compute.
- 2) Self-Discovering Reasoning Structures for Problem-Solving
	- When humans face a new problem, we first search internally what knowledge and skills from our prior experience might be helpful to solve it, then we apply these (multiple) relevant knowledge and skills to the task.
		- Given a task and a set of high-level problem-solving heuristics like *"Use critical thinking",* *"Let's think step by step,"* ==Stage 1== of Self-Discover aims to uncover the intrinsic reasoning structure for solving this task, via meta-reasoning. We use three meta-prompts to guide LLMs to select, adapt, and implement an actionable reasoning structure with no labels/training required. We format this structure in key-value pairs, similar to JSON due to interpretability and findings that following JSON boosts reasoning and generation quality. Stage 1 operates on a *task level*, meaning that we only need to run Self-Discover once for each task.
		- In ==Stage 2==, we can simply use the discovered reasoning structure to solve every instance of the given task by instructing models to follow the provided structure by filling each key, and arriving at the final answer.
	- ==Stage 1==
		- The first stage consists of three actions:
			1. ==SELECT==: Relevant reasoning modules for task-solving are chosen from the set of reasoning module descriptions. For example, "*reflective thinking*" might help search for first-principle theories on science problems, while "*creative thinking*" helps on generating a novel continuation to a story. Given a set of reasoning module descriptions and a few task examples, we select a subset of reasoning modules useful for solving tasks by using a model and a meta-prompt.
			2. ==ADAPT==:  Descriptions of selected reasoning modules are *rephrased* to be more specific to the task at hand. For example, we might translate *"break the problem into subproblems"* into *"calculate each arithmetic operation in order*, for an arithmetic problem. Given selected reasoning modules from previous step, rephrase each to be more specific to our task using a meta-prompt and a model.
			3. ==IMPLEMENT==: Adapted reasoning descriptions are implemented into a structured, actionable plan so that the task can be solved by following the structure.
	- ==Stage 2==

Abstract
> We introduce ==SELF-DISCOVER==, a ==general framework for LLMs to self-discover the task-intrinsic reasoning structures to tackle complex reasoning problems== that are challenging for typical prompting methods. Core to the framework is a self-discovery process where LLMs ==select== multiple ==atomic reasoning modules== such as critical thinking and step-by-step thinking, and ==compose them into an explicit reasoning structure for LLMs to follow== during decoding. SELF-DISCOVER substantially improves GPT-4 and PaLM 2's performance on challenging reasoning benchmarks such as BigBench-Hard, grounded agent reasoning, and MATH, by as much as ==32%== compared to Chain of Thought (CoT). Furthermore, SELF-DISCOVER ==outperforms inference-intensive methods such as CoT-Self-Consistency== by more than ==20%==, while requiring ==10-40x== fewer inference compute. Finally, we show that the self-discovered reasoning structures are universally applicable across model families: from PaLM 2-L to GPT-4, and from GPT-4 to Llama2, and share commonalities with human reasoning patterns.

# Paper Figures
![[Pasted image 20240520114618.png]]

![[Pasted image 20240520115150.png]]

