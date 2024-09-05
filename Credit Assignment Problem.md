---
aliases:
  - Temporal Credit Assignment Problem
---
In Reinforcement Learning, when rewards are only encountered sparsely, and rewards might occur some time after a causal action was taken, associating rewards with actions is not straightforward.

![[Pasted image 20240625165109.png|400]]
- With [[Eligibility Trace]]s, we basically look over time at the states we visit.
- Graph shows Eligibility Trace for a single state
- Every time we visit a state, we increase the Eligibility Trace, and when we don't visit it, we decrease it exponentially. This combines our frequency and recency heuristics together.
- When we see an error, we can now update the value function in *proportion to the eligibility trace*.