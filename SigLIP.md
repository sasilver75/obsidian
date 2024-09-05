---
aliases:
  - Sigmoid Loss for Language-Image Pretraining
---
March 27, 2023
[[DeepMind|Google DeepMind]]
[Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343)

Abstract
> We propose a simple pairwise Sigmoid loss for Language-Image Pre-training (SigLIP). Unlike standard contrastive learning with softmax normalization, the sigmoid loss operates solely on image-text pairs and does not require a global view of the pairwise similarities for normalization. The sigmoid loss simultaneously allows further scaling up the batch size, while also performing better at smaller batch sizes. Combined with Locked-image Tuning, with only four TPUv4 chips, we train a SigLiT model that achieves 84.5% ImageNet zero-shot accuracy in two days. The disentanglement of the batch size from the loss further allows us to study the impact of examples vs pairs and negative to positive ratio. Finally, we push the batch size to the extreme, up to one million, and find that the benefits of growing batch size quickly diminish, with a more reasonable batch size of 32k being sufficient. We release our models at [this https URL](https://github.com/google-research/big_vision) and hope our research motivates further explorations in improving the quality and efficiency of language-image pre-training.


# Non-Paper Figures

![[Pasted image 20240620141544.png|300]]