---
aliases:
  - ResNet
---
December 10, 2015 (A few months after [[Batch Normalization|BatchNorm]])
[[Microsoft Research]], including [[Kaiming He]]
Paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Combats [[Vanishing Gradients]], allowing for deeper networks to be trained

Abstract
> ==Deeper neural networks are more difficult to train==. We present a ==residual learning framework== to ease the training of networks that are substantially deeper than those used previously. ==We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions==. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset ==we evaluate residual nets with a depth of up to 152 layers==---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result ==won the 1st place on the ILSVRC 2015 classification task==. We also present analysis on CIFAR-10 with 100 and 1000 layers.


![[Pasted image 20240701105108.png]]
In residual learning, the input passes through 1 or more Convolutional layers as usual, but at the end, the original input is added back to the final output.
Called Residual Blocks because they don't actually need to learn the final output feature maps in the traditional sense, but instead learn just the *residual features* that need to be *added* to the input to get the final feature map.
![[Pasted image 20240701105254.png]]
During backpropagation, the gradients can flow back through these residual residual connections to reach the earlier layers of the networks fast, without much vanishing (letting us create deeper networks). This allowed ResNet to train a 152-layer model, shattering existing records.

![[Pasted image 20240701105352.png]]
