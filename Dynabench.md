April 7, 2021 -- [[Meta AI Research]] and others, [[Douwe Kiela]] lead author, [[Christopher Potts|Chris Potts]]
Paper: [Dynabench: Rethinking Benchmarking in NLP](https://arxiv.org/abs/2104.14337)

A tool that runs in the browser for creating dataset. Specifically, they hope to create examples that a *model* would mis-classify, but a human would not.
- The idea is that dataset creation, model development, and model assessment can directly inform eachother.

Dynabench consists of four dynamic tasks with multiple rounds of datasets that will grow over time. ([[Natural Language Inference|NLI]], [[Question Answering]], [[Sentiment Analysis]], Hate Speech)

Abstract
> We introduce ==Dynabench==, an open-source ==platform for dynamic dataset creation and model benchmarking==. Dynabench ==runs in a web browser and supports human-and-model-in-the-loop dataset creation==: annotators ==seek to create examples that a target model will misclassify, but that another person will not==. In this paper, we argue that Dynabench addresses a critical need in our community: contemporary models quickly achieve outstanding performance on benchmark tasks but nonetheless fail on simple challenge examples and falter in real-world scenarios. ==With Dynabench, dataset creation, model development, and model assessment can directly inform each other, leading to more robust and informative benchmarks==. We report on four initial NLP tasks, illustrating these concepts and highlighting the promise of the platform, and address potential objections to dynamic benchmarking as a new standard for the field.

![[Pasted image 20240422134015.png]]