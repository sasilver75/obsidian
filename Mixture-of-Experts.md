---
aliases:
  - MoE
---
Related:
- [[Expert Capacity]]
- [[Capacity Factor]]

Examples (roughly chronological):
- Hinton/Jordan's "Adaptive Mixtures of Local Experts"
- [[Noam Shazeer]]'s 2017 "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
- [[GShard]]
- [[Switch Transformer]]
- [[GLaM]]
- [[MegaBlocks]]

*Conventional* explanation for MoE:
- While Attention layers implement algorithms for reasoning, MLP layers store knowledge.
- Thus, by MoE-ifying MLPs, we're supposed to get a boost in knowledge.