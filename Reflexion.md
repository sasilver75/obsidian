March 20, 2023
Northeastern University, MIT, Princeton (Shinn et al.)
[Reflexion: Language Agents with Verbal Reasoning](https://arxiv.org/abs/2303.11366)
#zotero 
Takeaway: ...

References:
- Video: [Lead Author Noah Shinn @ London AI4Code: Reflexion](https://www.youtube.com/watch?v=kKNx64AmzwU)

---

## Introduction
- Recent works like [[ReAct]], Saycan, [[Toolformer]]. HuggingGPT, generative agents, and [[WebGPT]] have demonstrated the feasibility of autonomous decision-making agents that are built on top of a LLM core.
	- These methods use LLMs to generate text and "actions" that can be used in API calls, and executed in an environment.
	- Such approaches so far have been limited to using in-context examples as a way of teaching the agents, since optimization schemes that modify parameters require substantial amounts of compute and time.
- In this paper, they propose an alternative approach called [[Reflexion]] that uses verbal reinforcement to help agents learn from prior failings. 
	- It ==converts binary or scalar feedback from the environment into verbal feedback in the form of a textual summary, which is then added as additional context for the LLM agent in the next episode==.
	- Authors say this self-reflective feedback acts as a "semantic gradient signal" by providing the agent with a concrete direction to improve upon.
	- A Reflexion agent learns to optimize its own behavior to solve decision making, programming, and reasoning tasks through trial, error, and self-reflection.
	- Generating useful reflective feedback is challenging
		- Requires a good understanding of where the model made mistakes (i.e. the credit assignment problem)
		- Requires the ability to generate a summary containing *actionable insights* for improvement
	- They explore three ways for doing this:
		1. Simple binary environment feedback
		2. Predefined heuristics for common failure cases
		3. Self-evaluation such as binary classification using LLMs/self-written unit tests
	- ==In all implementations, the evaluation signal is amplified to natural language experience summaries which are then stored in long-term memory.==
- ==Reflexion has several advantages:==
	1. Lightweight and doesn't require finetuning the LLM.
	2. Allows for more nuanced forms of feedback (eg targeted changes in actions).
	3. Allows for more explicit and interpretable form of episodic memory over prior experiences.
	4. Provides more explicit hints for actions in future episodes.
- Authors perform experiments on decision-making tasks to test sequential action choices over long trajectories, reasoning tasks to test knowledge-intensive, single-step generation improvement, and programming tasks to teach the agent to effectively use external tools such as compilers and interpreters.
	- Reflexion agents are better decision makers/reasoners/programmers along all these tasks.



## Related work
- [[Self-Refine]] uses an iterative framework for self-refinement to autonomously improve generation through self-evaluation. Effective, but limited to single-generation reasoning tasks.
- Other methods (*Refiner: Reasoning feedback on intermediate representations*) fine-tune critic models to provide intermediate feedback within trajectories to improve reasoning responses.
- Other methods (*Decomposition enhances reasoning via self-evaluation guided decoding*) use stochastic beam search over actions to perform more efficient decision-making search strategy, which allows the agent to use foresight advantage due to its self-evaluation component.
- Yoran et al and Nair et al use *decider models* to reason over several generations.
- Kim et al use a retry pattern over a fixed number of steps without na evaluation step.
- Goodman (MetaPrompt) perform a qualitative evaluation step that proposes optimizations to the previous generation.
- ==In this paper, we show that several of these concepts can be enhanced with *self-reflection* to build a persisting memory of self-reflective experiences which allow an agent to identify its own errors and self-suggest lessons to learn from its mistakes over time.==


## Reflexion: Reinforcement via Verbal Reflection
- Our modular formulation for Reflexion utilizes three distinct models:
	- ==Actor== ($M_a$) which generates text and actions ("Here's a guess")
	- ==Evaluator== ($M_e$) which scores the outputs produced by $M_a$ ("Here's a score/feedback for your guess")
	- ==Self-Reflection== model ($M_{sr}$) which generates verbal reinforcement cues to assist the Actor in self-improvement. ("Here's an actionable lesson from your feedback")
- ==Actor==: Built upon an LLM specifically prompted to generate necessary text and actions, conditioned on state observations.
	- Authors explore various actor models, like [[Chain of Thought|CoT]] and [[ReAct]]. 
	- Authors also add a memory component *mem* that provides additional context to this agent. 
- ==Evaluator==: Plays a crucial role in assessing the generated outputs produced by the Actor. Takes as input a generated trajectory and computes a reward score that reflects its performance within the given task context. 
	- Defining effective value and reward functions that apply to semantic spaces is difficult, so they try several variants, including exact-match grading, pre-defined heuristic functions, and using LLMs themselves as evaluators.
- ==Self-Reflection==: Plays a crucial role by generating verbal self-reflections to to provide feedback for future trials.
	- Given a sparse reward signal (eg binary success status), the current trajectory, and its persistent memory *mem*, the self-reflection model generates nuanced and specific feedback. 
	- This is then stored in the agent's memory *mem*. 
	- ((This seems to conflict with the Figure 2 diagram, where the Self-Reflection model only seems to take the trajectory and reward signal from the evaluator?))
- Memory
	- One of the core components of the Reflexion process are the notion of short-term and long-term memory.
	- At inference time, the Actor conditions its decisions on short and long-term memory.
		- ==The trajectory history serves as the short-term memory, while the outputs from the Self-Reflection model are stored in long-term memory.==
- Reflexion Process
	- Thus formalized as an iterative optimization process:
		1. Actor produces a trajectory $\tau_0$  by interacting with the environment.
		2. Evaluator produces a score $r_0$ which is computed as $r_t = M_e(\tau_0)$ ; it's only a scalar reward.
		3. To amplify $r_0$ to a feedback form that can be used for improvement by an LLM, the Self-Reflection model analyzes the set of $\{\tau_0, r_0\}$ to produce a summary $sr_0$ , which is stored in the memory *mem*. $sr_0$ is a verbal experience feedback for trial $t$.
	- The Actor, Evaluator, and Self-Reflection models work together through trials in a loop until the Evaluator deems $\tau_T$ to be correct.

## Experiments
- We evaluate various natural language RL setups on decision-making, reasoning, and code generation tasks.
	- QA using [[HotpotQA]] (20% improvement over strong baseline)
	- Multi-step tasks in common household environments in AlfWorld (22% improvement over strong baseline)
	- Code-writing tasks using [[HumanEval]], [[MBPP]], and LeetcodeHard (11% improvement on HumanEval over strong baseline)
- ... See figures for results on each

## Limitations, Impact, and Conclusion
- In this study, they limit long-term memory to a sliding window with some maximum capacity because of LM context length limits, but we encourage future work to extend the memory component of Reflexion with more advanced structures like vector embedding databases or traditional SQL databases.
- 


Abstract
> Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose ==Reflexion==, a ==novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback==. Concretely, Reflexion ==agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials==. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion ==achieves a 91% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80%.== We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance.


# Paper Figures

![[Pasted image 20240718130353.png|600]]
Agent makes an action against the environment, and receives an observation. The Evaluator scores the output, and the Self-Reflection module generates verbal reinforcement cues to assist the actor in self-improvement. It seems that the actor receives both the trajectory log as well as the reflection/lesson log.

![[Pasted image 20240718132640.png|600]]

![[Pasted image 20240718132653.png|600]]

![[Pasted image 20240718132703.png|600]]

![[Pasted image 20240718132713.png|600]]

![[Pasted image 20240718132741.png|600]]

![[Pasted image 20240718133212.png]]

![[Pasted image 20240718133309.png]]

