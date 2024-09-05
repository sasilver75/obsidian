---
aliases:
  - Kaiming Initialization
---
2015
[[Kaiming He]], Xiangyu Zhang, Shaoqing Ren, Jian Sun
[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)

Widely adopted as the default initialization for networks using ReLU activations (or its variants, like Leaky ReLU).

See previous: [[Xavier Initialization]]

----

Like [[Xavier Initialization]] (which this elaborates on), the goal is to prevent vanishing/exploding gradients, but designed specifically for [[Rectified Linear Unit|ReLU]] activation functions.
- For a layer with $n$ inputs, the weights are initialized randomly from a normal distribution with mean 0 and variance $\sqrt{2/n}$ .
- It takes into account the fact that reLU functions only pass positive values, effectively reducing the variance of the activations by half.


Abstract
> ==Rectified activation units== (rectifiers) are essential for state-of-the-art neural networks. In this work, we study rectifier neural networks for image classification from two aspects. First, we propose a Parametric Rectified Linear Unit (PReLU) that generalizes the traditional rectified unit. PReLU improves model fitting with nearly zero extra computational cost and little overfitting risk. Second, ==we derive a robust initialization method that particularly considers the rectifier nonlinearities==. This method enables us to train extremely deep rectified models directly from scratch and to investigate deeper or wider network architectures. Based on our PReLU networks (PReLU-nets), we achieve 4.94% top-5 test error on the ImageNet 2012 classification dataset. This is a 26% relative improvement over the ILSVRC 2014 winner (GoogLeNet, 6.66%). To our knowledge, our result is the first to surpass human-level performance (5.1%, Russakovsky et al.) on this visual recognition challenge.

