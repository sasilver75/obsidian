---
aliases:
  - ORPO
---
Link: https://arxiv.org/abs/2403.07691

A supervised (?) alignment/fine-tuning method introduced in March 2024

The authors emphasize that a minor penalty for the disfavored generation style is sufficient for preference-aligned SFT.

The authors attempt to paint this as a straightforward and innovative reference [[Model-Free]] monolithic odds ratio preference optimization algorithm (ORPO), and that ==it (perhaps) entirely eliminates the need for an additional preference alignment phase.==

The authors claim that the finetuned Phi-2, Llama-2, Mistral 7B with ORPO on the [[UltraFeedback]] dataset alone, and were able to surpass the performance of SoTA LMs with more than 7B and 13B parameters.
