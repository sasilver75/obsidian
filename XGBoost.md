---
aliases:
  - Extreme Gradient Boosting
---
An efficient and scalable implementation of [[Boosting|Gradient Boosting]] Decision Trees that became popular in machine learning competitions on Kaggle thanks to its versatility in handling various types of predictive tasks. Great for tabular learning.

- XGBoost has a built-in routine to handle missing values; when it encounters a missing value at a split point, it automatically learns which direction to take for missing values, based on what minimizes the loss.
- Support various objective functions (classification, regression, ranking), making it applicable to a wide array of problems.
- Built-in Cross-validation at each iteration of the boosting process, allowing for the assessment of the performance at each step.