---
aliases:
  - Lasso Regularization
---
("Lasso" = Least Absolute Shrinkage and Selection Operator)

Enhances the model by adding the sum of the "absolute values" of the coefficients as a penalty term to the loss function, promoting sparsity and the selection of relevant features.

Because L1 regularization can reduce some coefficients to zero, it effectively removes some features entirely from the model -- it's a built-in form of feature selection. A popular choice for models where simplicity and interpretability are essential, or when you're dealing with data that includes irrelevant features.

With respect to handling [[Multicollinearity]], L1 is less effective as it might randomly select one feature over another when they are highly correlated. In contrast, [[L2 Regularization]] is very effective in handling multicollinearity by distributing the coefficient values among the correlated features.

![[Pasted image 20240602222526.png]]With respect to optimization, L1 involves the absolute value of coefficients and isn't differentiable at zero, which often requires specific optimization algorithms like coordinate descent or subgradient methods.