November 25, 2022 (6 months before [[Let's Verify Step by Step]])
[[DeepMind]] (Uesato et al)
[Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)
#zotero 
Takeaway: 

Note: A year after this DeepMind paper, OpenAI did their [[Let's Verify Step by Step]] paper, which similarly did an analysis of PRMs applied to math, but the used a more capable base model, significantly more human feedback, and used the more challenging [[MATH]] dataset, instead of [[GSM8K]].


---

## Introduction




## Problem and Methods



## Results



## Discussion



## Related Work




## Conclusion






Abstract
> Recent work has shown that asking language models to generate reasoning steps improves performance on many reasoning tasks. When moving beyond prompting, this raises the question of how we should supervise such models: outcome-based approaches which supervise the final result, or process-based approaches which supervise the reasoning process itself? Differences between these approaches might naturally be expected not just in final-answer errors but also in reasoning errors, which can be difficult to detect and are problematic in many real-world domains such as education. We run the first comprehensive comparison between process- and outcome-based approaches trained on a natural language task, GSM8K. We find that pure outcome-based supervision produces similar final-answer error rates with less label supervision. However, for correct reasoning steps we find it necessary to use process-based supervision or supervision from learned reward models that emulate process-based feedback. In total, we improve the previous best results from 16.8% → 12.7% final-answer error and 14.0% → 3.4% reasoning error among final-answer-correct solutions.


# Paper Figures