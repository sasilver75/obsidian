---
tags:
  - benchmark
  - paper
---
November 21, 2023 -- [[Meta AI Research]] x [[HuggingFace]]
Paper: [GAIA: A Benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983)

GAIA: a benchmark for General AI Assistants that proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency.

There are ==466 questions + answers==; 300 are retained to power a leader-board available at [link](https://huggingface.co/gaia-benchmark)

> Regarding contamination: "For the GAIA benchmark, we've even had people from a specific countries ask us: 'Can we have the answers for the test set? We want to keep it for internal benchmarks and we think it could really help safety for our use case.' - Clémentine Fourrier from HuggingFace on Latent Space

----


Abstract
> We introduce GAIA, a benchmark for General AI Assistants that, if solved, would represent a milestone in AI research. GAIA proposes real-world questions that require a set of fundamental abilities such as reasoning, multi-modality handling, web browsing, and generally tool-use proficiency. ==GAIA questions are conceptually simple for humans yet challenging for most advanced AIs: we show that human respondents obtain 92\% vs. 15\% for GPT-4 equipped with plugins==. This notable performance disparity contrasts with the recent trend of LLMs outperforming humans on tasks requiring professional skills in e.g. law or chemistry. GAIA's philosophy departs from the current trend in AI benchmarks suggesting to target tasks that are ever more difficult for humans. We posit that the advent of Artificial General Intelligence (AGI) hinges on a system's capability to exhibit similar robustness as the average human does on such questions. Using GAIA's methodology, we devise 466 questions and their answer. We release our questions while retaining answers to 300 of them to power a leader-board available at [this https URL](https://huggingface.co/gaia-benchmark).

![[Pasted image 20240420131203.png]]

![[Pasted image 20240420131004.png]]
