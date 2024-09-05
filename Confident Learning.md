---
aliases:
  - CL
---


October 21, 2019
MIT, [[Google Research]]
[Confident Learning: Estimating Uncertainty in Dataset Labels](https://arxiv.org/abs/1911.00068)

References
- Lecture Notes: [[Data-Centric AI (2) - Label Errors]]

---

==[[Confident Learning]] (CL)== is a framework of theory and algorithms focused on identifying and handling noisy or mislabeled data in datasets. The core concept is estimating the joint distribution between noisy (observed) labels and true (unknown) labels in a dataset.

It's useful for:
- Finding label errors in the dataset
- Reranking data by likelihood of there being a label issue
- Learning with noisy labels
- Complete characterization of label noise in a dataset
	- We'll focus on this one, in the context of classification with single-labels.
- Data curation (via ordering examples by their probability of being mislabeled and pruning appropriately)

The `Cleanlab` package implemented many CL methods.

