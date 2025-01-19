---
aliases:
  - MDP
---
MDPs are an extension/generalization of [[Markov Chain]] in which we also have *actions.*
Related: Partially-Observable Markov Decision Process (POMDP) vs Fully-Observable MDPs


![[Pasted image 20250118193349.png]]
[[Markov Decision Process]]
- (They're being really inconsistent lecture-to-lecture regarding their use of variables. For instance here we have $p$ representing the mode, which gives probabilities of next-states and rewards given actions and states.)

- See that from the given model $p$, we can  marginalize out/extract the state transition probability or expected reward individually.
	- First looks at the transition to the next state
	- Second looks at all of the rewards that we could expect from a transition starting from state s and taking an action a.
		- Takes into account all of of the states that we could end up in, and then the reward of being in each of those states.

Alternative definition of the MDP, which is quite common in the literature:
![[Pasted image 20250118193711.png]]


-------

A process in which an agent takes actions against an environment, which produces a reward and a new state, which are returned to the agent. The process repeats until (in the episodic case) some terminal state is reached, marking the end of an episode.
In a finite MDP, the states, actions, and rewards are restricted to finite sets. 
The dynamics of the environment are given with a probability function providing p(s', r | s, a) -- the probability of the next sta
Our goal is to determine some optimal policy, which is a state-dependent distribution over actions that maximizes expected rewards over many trajectories.
To make progress against this, we consider state-value functions, and action-value functions, which give us the expected return G_t, assuming the agent follows a policy pi and is at a given state or state/action pair.
If we can find the optimum of these functions, we can find the optimal policy.

![[Pasted image 20240623195940.png|400]]