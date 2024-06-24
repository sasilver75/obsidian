---
aliases:
  - MDP
---
MDPs are an extension of [[Markov Chain]]s.
Related: Partially-Observable Markov Decision Process (POMDP) vs Fully-Observable MDPs

A process in which an agent takes actions against an environment, which produces a reward and a new state, which are returned to the agent. The process repeats until (in the episodic case) some terminal state is reached, marking the end of an episode.
In a finite MDP, the states, actions, and rewards are restricted to finite sets. 
The dynamics of the environment are given with a probability function providing p(s', r | s, a) -- the probability of the next sta
Our goal is to determine some optimal policy, which is a state-dependent distribution over actions that maximizes expected rewards over many trajectories.
To make progress against this, we consider state-value functions, and action-value functions, which give us the expected return G_t, assuming the agent follows a policy pi and is at a given state or state/action pair.
If we can find the optimum of these functions, we can find the optimal policy.

![[Pasted image 20240623195940.png|400]]