In [[Reinforcement Learning]], on-policy algorithms learn and update their policy directly from the experience gathered by following the current policy.
- ==The policy used for generating actions and the policy being optimized are the same.==
- The agent learns from the actions it takes in the environment based on its current policy.

Compared to [[Off-Policy]] learners, On-policy algorithms are generally more stable and converge more easily, as they rely on the current policy to gather experience. 