---
aliases:
  - RL
---
Resources:
- [WIKIPEDIA: Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

An interdisciplinary area of ML and optimal control concerned with how an intelligent ==agent== with some ==state== in a dynamic environment should take ==actions== to maximize cumulative ==reward==.

One of the three basic ML paradigms, along [[Supervised Learning]] and [[Unsupervised Learning]]

The agent must find a balance between ==exploration== (of uncharted territory) and exploitation (of current knowledge), with the goal of maximizing long term reward, whose feedback may be incomplete or delayed.
- Related is the [[Credit Assignment Problem]], in which rewards are only sparsely received throughout exploration of the environment, and we need to determine which actions taken by the agent resulted in the reward.

The purpose of an agent is to learn an optimal (or nearly-optimal) policy that maximizes the "reward function" or other user-provided reinforcement signal that accumulates from the immediate rewards.

The environment is typically stated in the form of a [[Markov Decision Process]]:
- A set of environment and agent states, $S$
- A set of actions, $A$ , of the agent
- $P_a(s, s') = Pr(S_{t+1}=s'|S_t=s,A_t=a)$ , the probability of transition (at time t) from state $s$ to another state $s'$, under action $a$.x/
- $R_a(s, s')$, the immediate reward after transition from $s$ to $s'$ with action $a$.

Formulating as a MDP assumes that the agent directly observes the current environmental state; in this case, the problem is said to have ==full observability.==
- If the agent only has access to a subset of states, or if observed states are corrupted by noise, the agent is said to have *==partial observability==*, and the problem must be formulated as a ==Partially-observable Markov Decision Process.==

RL is great for situations where:
- A model of the environment is known, but an analytic solution is not available.
- Only a simulation model of the environment is given.
- The only way to collect information about an environment is to interact with it.