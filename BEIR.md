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

> "BEIR is practically useless nowadays; it was useful as a zero-shot test set. Among models trained openly on MS MARCO _without_ BEIR contamination or feedback, ColBERTv2 is ahead of everything else. Nowadays, all the top models are build FOR BEIR/MTEB evals, and use (at best) the training splits of BEIR tasks (and more often than not, direct/indirect validation on BEIR test sets). I have a big text file with 50 papers that shows the ColBERT paradigm being 50-100x more data efficient and up to 15-30 points better in quality than single vector. Note: "Everything else" here refers to dense retrievers of the same size and budget/tricks; you CAN build expensive cross-encoders that do better than vanilla BERT-base ColBERTv2. SPLADE is a great competitor at mid-scale, but dense bi-encoders are generally not."
> - Omar Khattab, Jan 28 (ColBERT): https://x.com/lateinteraction/status/1751661624539357550
> - Followup: If BEIR isn't valid due to contamination, are there any benchmarks that are still valid? Omar: We're working on it ;) -- pace is slightly slower since DSPy takes a bit more of my time.

![[Pasted image 20240223145524.png]]
