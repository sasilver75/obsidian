---
aliases:
  - Distillation with No Labels
---


April 29, 2021
Paper: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) ("Self-DIstillation with NO Labels")

A [[Vision Transformer|ViT]] that uses self-supervised learning. Uses [[Self-Distillation]], where a teacher model and a student model are trained simultaneously; the student model learns to predict the output of the teacher model.

Abstract
> In this paper, we question if self-supervised learning provides new properties to Vision Transformer (ViT) that stand out compared to convolutional networks (convnets). Beyond the fact that adapting self-supervised methods to this architecture works particularly well, we make the following observations: first, ==self-supervised ViT features contain explicit information about the semantic segmentation of an image==, which does not emerge as clearly with supervised ViTs, nor with convnets. Second, ==these features are also excellent k-NN classifiers==, reaching 78.3% top-1 on ImageNet with a small ViT. Our study also underlines the importance of ==momentum encoder==, ==multi-crop training==, and the use of ==small patches== with ViTs. We implement our findings into a ==simple self-supervised method, called DINO==, which we interpret as a form of self-distillation with no labels. We show the synergy between DINO and ViTs by achieving 80.1% top-1 on ImageNet in linear evaluation with ViT-Base.
