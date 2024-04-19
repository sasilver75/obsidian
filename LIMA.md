---
aliases:
  - Less is More for Alignment
---
May 18, 2023
Paper: [Less is More for Alignment](https://arxiv.org/abs/2305.11206v1)


Used ==1,000 high-quality examples and fine-tuned using SFT==. The result shows us that ==data quality is the most important factor in performing alignment via SFT==. This smaller model fell only a little short of GPT-4 and Claude in human preferences, despite being only a 65B model!

In it, the authors propose the [[Superficial Alignment Hypothesis]], which is summarized in the quote below:
> A models' knowledge and capabilities are learnt almost entirely during pre-training, while alignment teaches it which sub-distributions of formats should be used when interacting with users.

Abstract
> Large language models are trained in two stages: (1) unsupervised pretraining from raw text, to learn general-purpose representations, and (2) large scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences. We measure the relative importance of these two stages by training ==LIMA==, a ==65B parameter LLaMa language model== ==fine-tuned== with the standard ==supervised== loss ==on only 1,000 carefully curated prompts and responses==, ==without any reinforcement learning or human preference modeling==. LIMA demonstrates remarkably strong performance, learning to follow specific response formats from only a handful of examples in the training data, including complex queries that range from planning trip itineraries to speculating about alternate history. Moreover, the model tends to generalize well to unseen tasks that did not appear in the training data. In a controlled human study, responses from LIMA are either equivalent or strictly preferred to GPT-4 in 43% of cases; this statistic is as high as 58% when compared to Bard and 65% versus DaVinci003, which was trained with human feedback. Taken together, these results strongly suggest that almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output.

