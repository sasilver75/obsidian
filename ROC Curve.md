---
aliases:
  - Receiver Operating Characteristic
---
A ROC curve is a graph showing the performance of a classification model at all classification thresholds, plotting two parameters:
1. True Positive Rate (a synonym for [[Recall]])
2. False Positive Rate

True Positive Rate (TPR): Of the times where something is *actually* true, what % of the time do we say it's true?
$TPR = TP / (TP+FN)$ 

False Positive Rate (FPR): Of the times where something is *actually* false, what % of the time do we wrongly say it's true?
$FPR = FP / (FP + TN)$ 

A ROC curse plots TPR vs FPR at different classification thresholds; lowering the threshold intuitively means that we're more likely to classify things as positive, thus increasing ***both*** false positives and true positives.

![[Pasted image 20240422143714.png]]


Related: [[AUC]]: Area Under the ROC Curve
![[Pasted image 20240422144317.png]]
The area under the ROC curve provides an aggregate measure of performance across all possible classification thresholds; one way of interpreting AUC is as the probability that the model ranks a random positive example *more highly* than a random negative example.