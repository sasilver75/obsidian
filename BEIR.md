---
aliases:
  - Benchmarking-IR
---
April 17, 2021 -- TU-Darmstadt
Paper: [BEIR: A Heteorogenous Benchmark for Zero-Shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
Authors include [[Nils Reimers]]

The BEIR paper is a great practical resource for anyone that wants to understand the different components of search systems and their impact on top-level performance.

Abstract
> Existing neural information retrieval (IR) models have often been studied in homogeneous and narrow settings, which has considerably limited insights into their out-of-distribution (OOD) generalization capabilities. To address this, and to facilitate researchers to broadly evaluate the effectiveness of their models, we introduce Benchmarking-IR (BEIR), a ==robust and heterogeneous evaluation benchmark for information retrieval==. We leverage a ==careful selection of 18 publicly available datasets from diverse text retrieval tasks== and domains and evaluate 10 state-of-the-art retrieval systems including lexical, sparse, dense, late-interaction and re-ranking architectures on the BEIR benchmark. Our results show ==BM25 is a robust baseline== and ==re-ranking and late-interaction-based models on average achieve the best zero-shot performances==, however, at high computational costs. In contrast, dense and sparse-retrieval models are computationally more efficient but often underperform other approaches, highlighting the considerable room for improvement in their generalization capabilities. ==We hope this framework allows us to better evaluate and understand existing retrieval systems==, and contributes to accelerating progress towards better robust and generalizable systems in the future. BEIR is publicly available at [this https URL](https://github.com/UKPLab/beir).

![[Pasted image 20240223145524.png]]
