---
aliases:
  - Bootstrap Your Own Latent
---
June 13, 2020
Paper: [Bootstrap Your Own Latent: A new approach to Self-Supervised Learning](https://arxiv.org/abs/2006.07733)

A [[Self-Supervised Learning]] approach to Computer Vision that simplifies the process by ==eliminating the need for negative pairs==. Using two networks, a target network and an online network. We create two augmentations of the same image and feed one as input to each of the networks; the core idea is for the online network to learn representations that can predict the target network's representations of the same data. The target network's weights are a slowly-moving-average of the predictor network's weights.

Abstract
> We introduce Bootstrap Your Own Latent (BYOL), a new ==approach to self-supervised image representation learning==. BYOL relies on ==two neural networks==, referred to as ==an online network and a target network==, that ==interact and learn from each other==. From an augmented view of an image, we ==train the online network to predict the target network representation of the same image under a different augmented view==. At the same time, we ==update the target network with a slow-moving average of the online network==. While state-of-the art methods rely on negative pairs, BYOL achieves a new state of the art without them. BYOL reaches 74.3% top-1 classification accuracy on ImageNet using a linear evaluation with a ResNet-50 architecture and 79.6% with a larger ResNet. We show that BYOL performs on par or better than the current state of the art on both transfer and semi-supervised benchmarks. Our implementation and pretrained models are given on GitHub.


