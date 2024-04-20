---
aliases:
  - Holistic Evaluation of Language Models
---
November 16, 2022 -- Stanford
Paper: [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110)
Authors include [[Percy Liang]]
#benchmark 

A framework developed at Stanford's [[Center for Research on Foundation Models]] (CRFM) in Nov 2022

A multi-metric approach to evaluation of LLMs, assessing ==7 metrics== (accuracy, calibration, robustness, fairness, bias, toxicity, efficiency) ==for each of the 16 core scenarios/use cases== that were identified, to show hte trade-offs of models.

Also 7 targeted evaluations based on 26 targeted scenarios

Also large-scale evaluation of 30 prominent language models on ==all 42 scenarios==

It's an extremely long paper that might be interesting if you choose to get more into evaluations.

Abstract
> Language models (LMs) are becoming the foundation for almost all major language technologies, but their capabilities, limitations, and risks are not well understood. We present ==Holistic Evaluation of Language Models== (HELM) to improve the transparency of language models. First, we ==taxonomize the vast space of potential scenarios (i.e. use cases) and metrics (i.e. desiderata) that are of interest for LMs==. Then we ==select a broad subset based on coverage and feasibility==, noting what's missing or underrepresented (e.g. question answering for neglected English dialects, metrics for trustworthiness). Second, we ==adopt a multi-metric approach==: We measure ==7 metrics== (==accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency==) for each of 16 core scenarios when possible (87.5% of the time). This ensures metrics beyond accuracy don't fall to the wayside, and that ==trade-offs are clearly exposed==. We also perform ==7 targeted evaluations==, ==based on 26 targeted scenarios==, to analyze specific aspects (e.g. reasoning, disinformation). Third, we conduct a ==large-scale evaluation of 30 prominent language models== (spanning open, limited-access, and closed models) ==on all 42== ==scenarios==, 21 of which were not previously used in mainstream LM evaluation. Prior to HELM, models on average were evaluated on just 17.9% of the core HELM scenarios, with some prominent models not sharing a single scenario in common. We improve this to 96.0%: now all 30 models have been densely benchmarked on the same core scenarios and metrics under standardized conditions. Our evaluation surfaces 25 top-level findings. For full transparency, we release all raw model prompts and completions publicly for further analysis, as well as a general modular toolkit. ==We intend for HELM to be a living benchmark for the community==, continuously updated with new scenarios, metrics, and models.


![[Pasted image 20240420140731.png]]
![[Pasted image 20240420140739.png]]

![[Pasted image 20240420140756.png]]
