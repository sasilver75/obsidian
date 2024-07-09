---
aliases:
  - Bandit
---


A Bandit refers to a type of problem known as the [[Multi-Armed Bandit]] problem, which imagines a gambler at a row of slot machines (one-armed bandits), trying to determine which machine offers the best payout.

Variants: 
- Stochastic Bandits: Rewards are random variables (Usually the default)
- Adversarial Bandits: Rewards can be chosen by an adversary.
- Contextual Bandits: Rewards depend on a context or state.

Use cases:
- Online advertising (which advertisement to show?)
- Clinical trials (which treatment to test?)
- Recommendation systems (which item to recommend?)

At the center of Bandit problems is the tension between exploration and exploitation.
- Without an appropriate amount of exploring, it's possible that we won't learn the true distribution of an individual bandit's returns before we decide to "lock in" and exploit what we've learned.
- At the same time, an "fully-baked" optimal policy can't be one that has a(n, eg, stochastic) component of exploration to it.

Algorithms
- [[Epsilon-Greedy]]: Choose the best arm most of the time, explore randomly sometimes.
- [[Upper Confidence Bound]] (UCB): Balance exploration and exploitation based on uncertainty.
- [[Thompson Sampling]]: Use a Bayesian approach to balance exploration and exploitation.

Performance in bandit problems is often measured by "regret," the difference between optimal and actual performance.