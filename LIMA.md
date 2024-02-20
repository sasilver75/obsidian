---
aliases:
  - Less is More for Alignment
---
Paper: https://arxiv.org/abs/2305.11206v1
Paper/Model from May 2023

Used 1,000 high-quality examples and fine-tuned using SFT. The result shows us that data quality is the most important factor in performing alignment via SFT. This smaller model fell only a little short of GPT-4 and Claude in human preferences, despite being only a 65B model!

In it, the authors propose the [[Superficial Alignment Hypothesis]], which is summarized in the quote below:

> A models' knowledge and capabilities are learnt almost entirely during pre-training, while alignment teaches it which sub-distributions of formats should be used when interacting with users.

