Link: https://www.interconnects.ai/p/why-reward-models-matter
#article 

-----

We're entering an era of model-adaptation research where some people that bash on [[Reinforcement Learning from Human Feedback|RLHF]] doen't even know what a reward model is.
- In 2024, pmost folks are going to embrace that we might not need RL for RLHF, because of the simplicity and success of [[Direct Preference Optimization|DPO]]
	- Link: [[Do we need RL for RLHF? (Dec 2023) {Nathan Lambert, Interconnect Newsletter}]]
- DPO is actually making reward models more prevalent, but less likely to be used. (?)

Reward models give us an entirely new angle to audit the representations, pitfalls, and strengths of our LLMs without relying on the messiness of prompting and per-token computation limits.

Reward models tend to not need prompting, and only generate one token given any input text... which is a compelling simplicity.

In RLHF, reward models are LLMs that are tasked with approximating part of the traditional environment in an RL problem. 
- ==The reward model takes in any piece of text and returns a score.==

![[Pasted image 20240214134242.png]]
Above:
- Left: Traditional trial and error reinforcement learning problem
- Right: RLHF, where an agent (our LLM) is trained on data, produces an output, and that output is graded by another language model (our reward model), and the reward is used to update the original LLM's parameters.

In the RL worldview, the *environment* is the most important part of solving a problem -- it's the reason that it's assumed to be static!

- ==We can construct static comparisons for a reward model (e.g. for bias or specific viewpoints, and then calculate the specific preference between any two sides (which returns are higher scores)==

If we can effectively query our environment from the RL point of view at any made-up state, we In practice, reward models are seen as intermediate artifacts of the training process, rather than as a core piece of the framework -- almost ignoring the potential signal!
- The reward model is engineered in order to improve the downstream model post-RL optimization.

The core thing to remember about DPO in all of this is that its loss function is derived directly from the loss function for a pairwise reward model...
- Secondly, the reward is a ratio of log-probs, rather than an explicit score. 

![[Pasted image 20240214134649.png]]
- It's not clear why certain scores are higher
	- If this is the type of filter that the rest of us (not OpenAI, Anthropic) are pushing model updates through when doing RLHF, a lot can go wrong.

Reward models are a wonderful way to learn about the representation of language models, and we need more of them -- we have so few, and none from the top training labs. ==Reward models are a wonderful way to learn about the representation of language models, letting us peak behind the curtain of the content represented by stochastic parrots.==

Today, generative reward labels are very popular -- the largest application of this is in LLM-as-a-judge evaluation tools like [[AlpacaEval]] and [[MT-Bench]], which require prompting a model to choose which response is better.


