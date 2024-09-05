Used in both binary and multi-class classification, especially useful when dealing with class-imbalanced data.

==In a single class scenario:==
![[Pasted image 20240707164250.png]]
See: [[Recall|Sensitivity]] (Recall) and [[Precision|Specificity]] (Precision)
- Recall = TP / (TP + FN)
	- Of the positive examples in the data, what proportion were predicted positive by the model?
- Precision = TP / (TP + FP)
	- Of the positive predictions by the model, what proportion of them were actually positive examples?

==In a multi-class scenario:==
	Balanced Accuracy = (Sum of recalls of each class / number of classes)
- There isn't a single "negative" class; instead, each class is treated as the "positive" class in turn, with the other classes combined acting as the negative class.
