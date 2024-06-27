---
tags:
  - technique
aliases:
  - CoT
---
January 28, 2022 (8 months before ChatGPT)
[[Google Research]] - Authors include Jason Wei, [[Quoc Le]]
Paper: [Chain of Thought Prompting Eilcits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903)
Takeaway: ...

Inspired: Chain of Explanation (Huang et al., 2023), Chain of Knowledge (Wang et al, 2023a), Chain of Verification (Dhuliawala et al., 2023), IR Chain of Thought (Trivedi et al., 2023), [[Chain of Note]] (Yu et al., 2023)

-----

Notes:
- ...

Abstract
> We explore how ==generating a chain of thought== -- a ==series of intermediate reasoning steps== -- ==significantly improves the ability of large language models to perform complex reasoning==. In particular, we show how such reasoning abilities emerge naturally in sufficiently large language models via a simple method called chain of thought prompting, where a few chain of thought demonstrations are provided as exemplars in prompting. Experiments on three large language models show that chain of thought prompting ==improves performance on a range of arithmetic, commonsense, and symbolic reasoning tasks==. The empirical gains can be striking. For instance, prompting a 540B-parameter language model with just eight chain of thought exemplars achieves state of the art accuracy on the GSM8K benchmark of math word problems, surpassing even finetuned GPT-3 with a verifier.



# Figures


# Non-Paper Figures
![[Pasted image 20240626214831.png]]
Above: This is from the "let's think step by step" zero-shot CoT paper. It's just a funny little graphic because the examples are good. "*Abrakadabra!"*