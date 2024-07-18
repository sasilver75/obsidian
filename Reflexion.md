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


## Related work


## Reflexion: Reinforcement via Verbal Reflection


## Experiments


## Limitations, Impact, and Conclusion



Abstract
> Large language models (LLMs) have been increasingly used to interact with external environments (e.g., games, compilers, APIs) as goal-driven agents. However, it remains challenging for these language agents to quickly and efficiently learn from trial-and-error as traditional reinforcement learning methods require extensive training samples and expensive model fine-tuning. We propose ==Reflexion==, a ==novel framework to reinforce language agents not by updating weights, but instead through linguistic feedback==. Concretely, Reflexion ==agents verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials==. Reflexion is flexible enough to incorporate various types (scalar values or free-form language) and sources (external or internally simulated) of feedback signals, and obtains significant improvements over a baseline agent across diverse tasks (sequential decision-making, coding, language reasoning). For example, Reflexion ==achieves a 91% pass@1 accuracy on the HumanEval coding benchmark, surpassing the previous state-of-the-art GPT-4 that achieves 80%.== We also conduct ablation and analysis studies using different feedback signals, feedback incorporation methods, and agent types, and provide insights into how they affect performance.


# Paper Figures