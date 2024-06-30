---
aliases:
  - Similarity Search for Active Learning and Search
---
June 30, 2020
Stanford, [[Meta AI Research]], UW Madison (incl Peter Bailis)
[Similarity Search for Efficient Active Learning and Search of Rare Concepts](https://arxiv.org/abs/2007.00077)

Enables the scaling of [[Active Learning]] learning to large datasets by restricting the candidate pool of the unlabeled dataset to only those that are nearest neighbors of the currently-labeled dataset. 

---


Abstract
> Many active learning and search approaches are intractable for large-scale industrial settings with billions of unlabeled examples. Existing approaches search globally for the optimal examples to label, scaling linearly or even quadratically with the unlabeled data. In this paper, we improve the computational efficiency of ==active learning== and search methods by ==restricting the candidate pool for labeling to the nearest neighbors of the currently labeled set instead of scanning over all of the unlabeled data==. We evaluate several selection strategies in this setting on three large-scale computer vision datasets: ImageNet, OpenImages, and a de-identified and aggregated dataset of 10 billion images provided by a large internet company. ==Our approach achieved similar mean average precision and recall as the traditional global approach while reducing the computational cost of selection by up to three orders of magnitude, thus enabling web-scale active learning.==