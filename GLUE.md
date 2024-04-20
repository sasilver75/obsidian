---
aliases:
  - General Language Understanding Evaluation
---
April 20, 2018 -- Courant Institute, [[AI2]], [[DeepMind]], others
Paper: [GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding](https://arxiv.org/abs/1804.07461)
#benchmark 

Model-agnostic, and incentivizes sharing knowledge across tasks because certain tasks have very limited training data.
Models surpassed humans on this benchmark within a year.
Succeeded by [[SuperGLUE]] in May 2019.

Abstract
> For ==natural language understanding (NLU)== technology to be maximally useful, both practically and as a scientific object of study, it must be general: it must be able to process language in a way that is not exclusively tailored to any one specific task or dataset. In pursuit of this objective, we introduce the General Language Understanding Evaluation benchmark (GLUE), a tool for evaluating and analyzing the performance of models across a diverse range of existing NLU tasks. GLUE is model-agnostic, but it incentivizes sharing knowledge across tasks because certain tasks have very limited training data. ==We further provide a hand-crafted diagnostic test suite that enables detailed linguistic analysis of NLU models==. We evaluate baselines based on current methods for multi-task and transfer learning and find that they do not immediately give substantial improvements over the aggregate performance of training a separate model per task, indicating room for improvement in developing general and robust NLU systems.
