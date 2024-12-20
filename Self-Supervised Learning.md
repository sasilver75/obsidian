---
aliases:
  - SSL
---
A paradigm of ML in which the model is trained on data that *provides its own labels* -- we leverage the structure of the data itself to create labels that are already inherent to the data, removing the need for manual labeling.

There are a variety of ==[[Pretext Task]]==s that we can ask models to solve, which help them ==learn useful representations==.

In NLP, we use SSL for LM pretraining by performing next-token-prediction on sequences of text.

In Vision, there are pretext tasks where we predict the rotation of an image, fill in missing parts of an image, etc.

A popular SSL task is to learn to distinguish between similar and dissimilar instances; in image processing, we might be trained to recognize that different views of the same object are similar, whereas different objects are dissimilar.

Like [[Unsupervised Learning]], SSL isn't especially hindered by not having supervision from provided labels that we know beforehand (as in [[Supervised Learning]]).