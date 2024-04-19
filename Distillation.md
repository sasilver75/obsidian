---
aliases:
  - Knowledge Distillation
  - Distilled
  - Distill
---
Traditional knowledge distillation involves transferring knowledge from a larger, more complex model (teacher) to a smaller, simpler model (student). Generally this involving training on the logits of the teacher, rather than on the data labels.

Variant: [[Self-Distillation]]


Note:
- I've heard "Distillation" used in two contexts:
	- "Model Distillation" where a weaker model is trained on the probability distributions (a strong training signal) of the larger model
	- "Dataset Distillation" in the context of synthetic data, where we use (eg) GPT-4 to *distill* a set of synthetic data (eg using a technique like that from [[Self-Instruct]])

