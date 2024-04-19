---
aliases:
  - SLM
---
April 11, 2024 -- [[Microsoft Research]] and others
Introduced in [[Rho-1]] paper: [Rho-1: Not All Tokens Are What You Need](https://arxiv.org/abs/2404.07965)

From abstract:
> Rho-1 employs ==Selective Language Modeling (SLM)==, which ==selectively trains on useful tokens that aligned with the desired distribution==. This approach involves ==scoring pretraining tokens using a reference model==, and then training the language model with a ==focused loss on tokens with higher excess loss==