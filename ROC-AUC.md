---
aliases:
  - Area under Curve
  - AUROC
---
"Area under the Curve" of a [[ROC Curve]]

The area under the ROC curve provides an aggregate measure of performance across all possible classification thresholds; one way of interpreting AUC is as the probability that the model ranks a random positive example *more highly* than a random negative example.

A model that's no better than a coin flip would have a ROC-AUC of 0.5 while a model that's always correct would have a ROC-AUC = 1.0.

Advantages:
- Robust to class imbalance because it specifically measures true and false positive rate.
- Doesn't require picking a threshold, since it evaluates performance across all thresholds.
- Scale invariant, thus it doesn't matter if your model's predictions are skewed.

![[Pasted image 20240422144538.png]]