---
tags:
  - F-Score
---
A measure of predictive performance, calculated as the ==***harmonic*** mean of the [[Precision]] and [[Recall]]==, thus symmetrically representing both precision and recall in one metric.

The highest possible F1 score is 1.0, indicating perfect precision and recall, with the lowest possible value being 0, if either precision or recall are zero.

![[Pasted image 20240422143530.png]]

Why is the F1 score important?
- Datasets can often be imbalanced, with a disproportionate number of instances in one class, compared to another. This imbalance can skew traditional evaluation metrics. 
	- If 95% of instances are in class A, you can have a 95% accuracy classifier that could naively predict class A in all cases. This ==accuracy paradox== highlights how traditional metrics can be misleading in the presence of class imbalance.
	- Similarly, under class imbalance, models with high precision might simply be conservative in predicting positive instances, whereas models with high recall might over-predict minority classes.
- In such scenarios, F1 is useful in providing a balance between precision and recall by taking their harmonic mean. The averaging ensures that an increase in one metric doesn't disproportionately affect the overall score without a corresponding improvement in the other.

Note that F1 assumes equal importance of precision and recall, which may not align with your use case/business objectives.


----
Aside: What are Precision and Recall, again?
- Remember: [[Precision]] is the number of true positive results divided by all samples predicted to be positive, including those not predicted correctly. "Of all the times you say 'yes', what percentage are you correct?"
- Remember: [[Recall]] is the number of true positive results divided by the number of all samples that *should* have been identified as positive. "Of all the times that something was *actually ground-truth true*, what percentage of the time did you correctly identify this?"
---


---
Aside: What's the Harmonic Mean, compared to the Geometric Mean, or the Arithmetic mean?

==Arithmetic mean==: The sum of the values divided by the total number of values. Appropriate when all values in the data sample have the same units of measure (eg all numbers are heights, dollars, miles).

==Geometric mean==: Calculated as the N-th root of the *product* of all values, where N is the number of values. For example, if we have three values, it's the cube-root of the product of these three values. Does not accept negative or zero values; all values must be positive. Appropriate when the data contains values with different units of measure (some are height, some dollars, some miles).

==Harmonic mean==: Calculated as the number of *N* values divided by the divided by the sum of the reciprocal of the values (1 over each value). Appropriate when the dataset is comprised of rates, where rates are the ratio between two quantities with different measures (eg speed, acceleration, frequency -- or, in ML, true positive rates, false positive rates, etc.). Does not take rates with negative or zero values; all rates must be positive.

---