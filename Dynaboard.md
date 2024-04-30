---
aliases:
  - Dynascore
---

May 21, 2021 (~1 month after Dynabench) -- [[Meta AI Research]] (Authors include [[Douwe Kiela]], [[Christopher Potts|Chris Potts]])
Paper: [Dynaboard: An Evaluation-As-A-Service Platform for Holistic Next-Generation Benchmarking](https://arxiv.org/abs/2106.06052)

It seems like it's an online benchmarking tool where you submit your model to be benchmarked, using features from the [[Dynabench]] tool.
A ==Dynascore== is computed, which is a composite metric of a bunch of elements, depending on the viewing user's preferences.

A method for integrating a bunch of diverse metrics into a single metric.


Abstract
> We introduce ==Dynaboard==, an ==evaluation-as-a-service framework for hosting benchmarks and conducting holistic model comparison==, ==integrated with the Dynabench platform==. Our platform evaluates NLP models directly instead of relying on self-reported metrics or predictions on a single dataset. Under this paradigm, ==models are submitted to be evaluated in the cloud==, circumventing the issues of reproducibility, accessibility, and backwards compatibility that often hinder benchmarking in NLP. This allows users to interact with uploaded models in real time to assess their quality, and permits the ==collection of additional metrics such as memory use, throughput, and robustness==, which -- despite their importance to practitioners -- have traditionally been absent from leaderboards. On each task, models are ranked according to the ==Dynascore==, a ==novel utility-based aggregation of these statistics==, which users can customize to better reflect their preferences, placing more/less weight on a particular axis of evaluation or dataset. As state-of-the-art NLP models push the limits of traditional benchmarks, Dynaboard offers a standardized solution for a more diverse and comprehensive evaluation of model quality.

![[Pasted image 20240422134135.png]]