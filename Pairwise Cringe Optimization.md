---
aliases:
  - PCO
---
December 27, 2023
[[Meta AI Research]]
Paper: [Some things are more CRINGE than others: Iterative Preference Optimization with the Pairwise Cringe Loss](https://arxiv.org/abs/2312.16682)
...
Takeaway: ...

See also: [[CRINGE Loss]]

Comparison: Similar to [[Self-Reward]], except this paper uses a fixed reward model, whereas the "reward model" co-evolves with the instruction-following component in the SR paper.

-----

Notes:
- ((I'm not sure how interesting it is that they were able to convert a binary feedback method to a pairwise preference method))

Abstract
> Practitioners commonly align large language models using ==pairwise preferences==, i.e., given labels of the type response A is preferred to response B for a given input. Perhaps less commonly, ==methods have also been developed for binary feedback==, i.e. training models given labels of type response A is good or bad. We show how ==an existing performant binary feedback method, the Cringe Loss== (Adolphs et al., 2022), ==can be generalized to the pairwise preference setting== using a simple soft margin extension. Pairwise Cringe Loss is straightforward to implement and efficient to train, and ==we find it outperforms state-of-the-art preference optimization algorithms such as PPO and DPO on the AlpacaFarm benchmark==. We show that iterations of training of our model are important for improved results, and that we can generalize DPO to Iterative DPO in the same way.