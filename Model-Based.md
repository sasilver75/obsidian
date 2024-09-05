A model is anything the agent uses to predict the environment's response to its actions.
- This might be an estimate of $p(s',r|s,a)$, the MDP distribution function.

Model-based methods are those where the agents uses a model to *plan* actions before they're taken.
That plan consists of:
- Transition dynamics of the environment
- Reward dynamics of the environment

If you have a good model of your environment (an MDP, some differential equation) to start with, then you can work in this world. Some people don't even consider Model-Based RL as reinforcement learning.
- If there's a specified, known, deterministic probability function P(s' | s, a) ... then  really powerful techniques to optimize $\pi$ exist:
	- [[Policy Iteration]]
	- [[Value Iteration]]
	- [[Dynamic Programming]]

