---
aliases:
  - MOSAIKS
---
UC Berkeley, ~2021

A deliberate anti-complexity statement against deep learning pipelines.

Insight:
- Random convolutional filters applied to natural images produce features that are surprisingly informative; you don't even need to learn the features, you just need enough of them.
- This comes from the "==Random Kitchen Sinks==" theory (2007), where random projections of high-dimensional inputs into rich-enough feature space can be linearly separated for a wide variety of tasks.
	- ((Lol, that's pretty funny))


Pipeline:
- Given satellite image patch (256x256 RGB)
- Apply ~8,000 random convolutional filters (random weights, never updated, never trained)
- [[Rectified Linear Unit|ReLU]] activation
	- (Different context than how it's used in Transformer, where we need a nonlinearity injected. Here, random features + ReLU approximate an arc-cosine kernel, which is important for reasons)
- Average pool each filter response -> scalar
- One vector of ~8,000 numbers per location
- Train a regressor on top, for any task

That's it. No backpop, no GPU required for feature extraction. No labeled data needed until the last step.

Key properties:
- One-time computation: Compute features once per location, store them. Any new prediction task = just train a new linear model on the same stored features. No re-running the imagery pipeline.
- Multi-task by design: Same feature vector → predict forest cover, income, population density, flood risk — whatever you have labels for. Each task just trains a new linear layer.