March 30, 2023
CMU, [[Allen Institute|AI2]], UW, [[NVIDIA]], UCSD, [[Google Research]] (Madaan et al.)
[Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
#zotero 
Takeaway: A simple iterative algorithm in which the same model is used to initially generate, and then iteratively alternates between critiquing and refining its outputs based on the critique (either N times, or until some stopping criteria is reached).


⚠️ In [[Large Language Models Cannot Self-Correct Reasoning Yet]], GDM authors note that the improvements in this study result from using *oracles* to guide the self-correction process, and the improvements vanish when oracle labels are not available.


---

## Introduction
- Although LLMs can generate coherent outputs, they often fall short in addressing intricate requirements; in these scenarios, they might benefit from further iterative refinement.
- Iterative refinement usually involves training a refinement model that relies on domain specific data (many times relying on external supervision or reward models trained on large datasets of expensive human annotation).
	- ==Iterative self-refinement== is a process involving creating an initial draft and subsequently refining it based on *self-provided* feedback.
		- In coding, a programmer might refactor an initial "quick and dirty" implementation, and then, upon reflection, refactor their code to something more efficient and readable.
- [[Self-Refine]] is an iterative self-refinement algorithm that alternates between two generative steps -- ==feedback== and ==refine==. These steps work in tandem to generate high-quality outputs.
	- Given an initial output generated by model M, we pass it back to the same model M to get *feedback*. Then, the feedback is passed back to the same model to *refine* the previously-generated draft.
	- This continues for either a specified number of iterations or until our model M determines that no further refinement is necessary.


## Iterative Refinement with Self-Refine
- Given an input sequence, Self-Refine generates an initial output, provides feedback on the output, and refines the output according to the feedback.
- Self-Refine iterates between feedback and refinement until a desired condition is set.
- Self-reflection relies on a suitable language model and three prompts:
	1. A prompt for initial generation
	2. A prompt for feedback
	3. A prompt for refinement
- The model alternates between feedback and refine steps until a stopping condition is met... note that ==to inform the model about the previous iterations, we retain the history of previous feedback and outputs by appending them to the prompt -- this allows the model to learn from past mistakes and avoid repeating them==. See figures.

## Evaluation and Analysis
- Authors evaluate self-refine on 7 diverse tasks:
	- Dialogue response generation
	- Code optimization
	- Code readability improvement
	- Math reasoning
	- Sentiment Reversal
	- Acronym generation
	- Constrained generation
- See table 1 for significant improvements. 
- Authors consider (task specific metrics, human preference metrics, and gpt-4-pref metrics (using GPT-4 as a proxy for human preference))
- Self-refine consistently improves over base models across all model sizes, and additionally outperforms the previous SoTA across all tasks.
- The modest performance gains in Math Reasoning can be traced back to the inability of the model to accurately identify whether there is any error in a draft response.
- Overall, having multiple FEEDBACK-REFINE iterations significantly enhances the quality of the output, although the marginal improvement naturally decreases with more iterations.


## Related Work
- Leveraging human and machine-generated natural language (NL) feedback for refining outputs has been effective for a variety of tasks, including summarization, script generation, program synthesis, and other tasks. Refinement methods differ in the *source* and *format* of the feedback.
	- ==Source of feedback==: Human feedback is costly, so several approaches use a scalar reward function as a surrogate to human feedback. Alternative sources like compilers or Wikipedia edits can provide some domain-specific feedback. Recently, LLMs have been used to generate feedback for general domains, but ours is the only method that generates feedback using LLM on its *own* output.
	- ==Representation of feedback==: Generally can be divided into natural language and non-natural-language feedback, where the latter can come in the form of human-provided example pairs or scalar rewards. We use NL feedback here, since it allows the model to easily provide self-feedback.
	- ==Types of refiners==: Pairs of feedback and refinement have been used to learn supervised refiners. We avoid training a separate refiner, and show that the same model can be used both as the refiner and the source of feedback across multiple domains.
	- ==Non-refinement reinforcement learning (RL) approaches==: Rather than have an explicit refinement, an alternative way to incorporate feedback is to optimize a scalar reward function (eg with RL); these methods differ from self-refine in that the model doesn't access feedback on an intermediate generation... and these methods require updating the model's parameters, unlike Self-Refine.

## Limitations and Discussion
- The main limitation of our approach is that the base models need to have sufficient few-shot modeling or instruction-following abilities in order to learn to provide feedback and refine their answers in an in-context fashion.
	- In appendix, authors experimented with [[Vicuna]]-13B, which struggled to follow the prompts intended for feedback refinement! This often led to outputs resembling assistant-like responses.
- We exclusively experiment with datasets in English; in other languages, the current models may not provide the same benefits.


Abstract
> Like humans, large language models (LLMs) do not always generate the best output on their first try. Motivated by how humans refine their written text, we introduce ==Self-Refine==, an ==approach for improving initial outputs from LLMs through iterative feedback and refinement==. The main idea is to ==generate an initial output using an LLMs; then, the same LLMs provides feedback for its output and uses it to refine itself, iteratively==. Self-Refine does not require any supervised training data, additional training, or reinforcement learning, and instead ==uses a single LLM as the generator, refiner, and feedback provider==. We evaluate Self-Refine across 7 diverse tasks, ranging from dialog response generation to mathematical reasoning, using state-of-the-art (GPT-3.5, ChatGPT, and GPT-4) LLMs. Across all evaluated tasks, outputs generated with Self-Refine are preferred by humans and automatic metrics over those generated with the same LLM using conventional one-step generation, ==improving by ~20% absolute on average in task performance==. Our work demonstrates that ==even state-of-the-art LLMs like GPT-4 can be further improved at test time== using our simple, standalone approach.


# Paper Figures

![[Pasted image 20240718000143.png|600]]
Process of iterative self-improvement. Given an output, we ask ourselves for a critique. Given a critique, we refine our output. We repeat.

![[Pasted image 20240718001432.png|600]]
Each row shows a different use case. In the top row, we see an example of initial generation, feedback, and refinement of a language model response, and in the bottom row, the same for code generation. Inspect the format of the feedback, specifically -- I'm sure that's a critical aspect of good refinement.

![[Pasted image 20240718001829.png|600]]
Incredibly simple. ==Note that when generating the refined response, we pass in all of the history of previous feedback and outputs by appending them to the prompt.== This lets the model learn from past mistakes and avoid repeating them.

![[Pasted image 20240718002912.png|600]]
See that Self Refine improves base prompting meaningfully across various tasks.

![[Pasted image 20240718005340.png|600]]
==Feedback quality plays a crucial role==
- ==Actionable Feedback==: "Avoid repeated calculations in the for loop"
- ==Generic Feedback==: "Improve the efficiency of the code."

![[Pasted image 20240718005703.png|600]]
Showing the impact of multiple iterations on score improvement. It's important and helps, but the effect decreases over time. I think this paper has good figures.

![[Pasted image 20240718005833.png|600]]
Differences of initial code generation (left) and the output after applying self-refine.

![[Pasted image 20240718010208.png|500]]

![[Pasted image 20240718121447.png|600]]

