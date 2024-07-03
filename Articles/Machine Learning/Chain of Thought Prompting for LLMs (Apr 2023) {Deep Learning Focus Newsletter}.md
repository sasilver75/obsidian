#article 
Link: https://cameronrwolfe.substack.com/p/chain-of-thought-prompting-for-llms

--------

![[Pasted image 20240305223037.png|550]]

LLMs are incredibly capable few-shot learners -- meaning that we can solve a variety of different problems just by formulating a textual prompt and having the LLM generate the correct answer.

Despite the power of LLMs, there are some problems that these models consistently struggle to solve. In particular, reasoning problems are notoriously difficult.
- Recent work has found that few-shot learning can be leveraged for a solution!

> "The goal of this paper is to endow language models with the ability to generate a 'chain of thought' -- a coherent series of intermediate reasoning steps that lead to the final answer for a problem!"

In particular, [[Chain of Thought]] (CoT) prompting is a recently-proposed technique that improves LLM performance on reasoning tasks.

Like other prompting strategies, CoT prompting inserts several example solutions to reasoning problems into the LLM's prompt. ==Each example includes a "chain of thought," or a series of intermediate reasoning steps for solving the problem.==
- As a result, the LLM learns to generate similar chains of thought when solving reasoning problems.

Such an approach uses minimal data (i.e. just a few examples for prompting), requires no task-specific fine-tuning, and significantly improves LLM performance on reasoning-based benchmarks.


# Chain of Thought Prompting

![[Pasted image 20240305223752.png|500]]
Above:
- CoT simply refers to a specific prompting technique that inserts a chain of thought (a series of intermediate reasoning steps) into an LLM's prompt.

> *"A prompting-only approach is important because it doesn't require a large training dataset, and a single model can perform many tasks without loss of generality."*

*CoT* prompting combines the strengths of a few-shot prompting with the benefit of generating natural language rationales.
- ((In other words, the results are hopefully both more accurate and more "explainable," though I'm not sure if we can always rely on the generated explanations...))


# How does CoT prompting work?

A simple example: Calculate a tip at a restaurant
- Take the total amount of the bill: $56.00
- Compute 10% of the total: $5.60
- Multiply this value by 2 (yielding a 20% tip): $11.20

Although this example is simple, the idea extends to a variety of mental reasoning tasks that we would want to solve as humans!

![[Pasted image 20240305224141.png]]

## CoT Prompting is Massively Beneficial
- Evaluation is performed using several different pre-trained LLMs including (GPT-3, LaMBDA, PaLM, Codex, and UL2). 
- We discover a few notable properties of CoT prompting
	1. ==CoT prompting seems to work much better for larger LLMs==
		- Smaller models are found to produce illogical chains of thought!
	2. ==More complicated problems (eg GSM8K) see a greater benefit from CoT prompting.==
	3. Compared to prior SoTA methods, CoT prompting with GPT-3 and PaLM-540B achieves comparable or improved performance in all cases.

Put simply, CoT prompting is found to provide a massive benefit on commonsense reasoning problems as well

## Variants of CoT prompting
- ==Zero-Shot==
	- Simply appends the words "Let's think step by step" to the end of the question being asked.
	- By making this simple addition to the LLM's prompt, we see that LLMs are able to generate CoTs without even observing any explicit examples of such behavior, allowing them to arrive at a more accurate answer!
![[Pasted image 20240305225457.png]]
- ==[[Self-Consistency]]==
	- A variant of CoT prompting that uses the LLM to generate multiple chains of thought, then takes the majority vote from those generations as the final answer.
	- Put simply, it replaces the greedy decoding procedure with a pipeline that generates multiple answers with the LLM and takes the most common of these answers.
![[Pasted image 20240305225451.png]]
- ==Least-to-most prompting==
	- Goes beyond CoT prompting by first breaking down a problem into smaller sub-problems, then solving each of the subproblems individually. As each sub-problem is solved, its answers are included in the prompt for solving the next problem.
	- Yields accuracy improvements on several tasks, and improves generalization to out-of-domain problems that require more reasoning steps.

![[Pasted image 20240305225756.png]]
# Takeaways
- The utility of CoT prompting
	- Requires no fine-tuning
	- Minimal extra-data (just a few exemplars and explanations)
- Reasoning emerges with scale
	- Not all models benefit from CoT prompting
- Do LLMs actually know how to reason?
	- It doesn't necessarily mean that LLMs inherently posess complex reasoning capabilities -- CoT prompting doesn't answer whether LLMs are actually reasoning or not.


































