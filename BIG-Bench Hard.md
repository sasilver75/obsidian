---
aliases:
  - BBH
---
October 17, 2022 -- [[Google Research]], Stanford (but recall this is really just a subset of BIG-Bench problems)
Paper: [Challenging BIG-Bench Tasks and whether Chain-of-Thought can Solve Them](https://arxiv.org/abs/2210.09261)

A subset of 23 challenging tasks from the [[BIG-Bench]] dataset (204 tasks), which use objective metrics, are hard, and contain enough samples to be statistically significant. Contain multistep arithmetic and algorithmic reasoning (understanding boolean expressions, SVG for geometric shapes, etc), language understanding (sarcasm detection, name disambiguation, etc) and some world knowledge.

Abstract
> BIG-Bench (Srivastava et al., 2022) is a diverse evaluation suite that focuses on tasks believed to be beyond the capabilities of current language models. Language models have already made good progress on this benchmark, with the best model in the BIG-Bench paper outperforming average reported human-rater results on 65% of the BIG-Bench tasks via few-shot prompting. But on what tasks do language models fall short of average human-rater performance, and are those tasks actually unsolvable by current language models?  
> In this work, we focus on ==a suite of 23 challenging BIG-Bench tasks which we call BIG-Bench Hard (BBH).== These are the task for which prior language model evaluations did not outperform the average human-rater. We find that ==applying chain-of-thought (CoT) prompting to BBH tasks enables PaLM to surpass the average human-rater performance on 10 of the 23 tasks, and Codex (code-davinci-002) to surpass the average human-rater performance on 17 of the 23 tasks==. Since many tasks in BBH require multi-step reasoning, few-shot prompting without CoT, as done in the BIG-Bench evaluations (Srivastava et al., 2022), substantially underestimates the best performance and capabilities of language models, which is better captured via CoT prompting. As further analysis, we explore the interaction between CoT and model scale on BBH, finding that ==CoT enables emergent task performance on several BBH tasks with otherwise flat scaling curves.==

![[Pasted image 20240420135330.png]]
![[Pasted image 20240420135348.png]]

![[Pasted image 20240420135407.png]]

Examples of Questions from each subset (are available on the  Sample Explorer on this [blog post](https://huggingface.co/spaces/open-llm-leaderboard/blog?utm_source=ainews&utm_medium=email&utm_campaign=ainews-et-tu-mmlu-pro))