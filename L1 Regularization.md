---
aliases:
  - Lasso Regularization
---
("Lasso" = Least Absolute Shrinkage and Selection Operator)

Enhances the model by adding the sum of the "absolute values" of the coefficients as a penalty term to the loss function, promoting sparsity and the selection of relevant features.

Because L1 regularization can reduce some coefficients to zero, it effectively removes some features entirely from the model -- it's a built-in form of feature selection. A popular choice for models where simplicity and interpretability are essential, or when you're dealing with data that includes irrelevant features.