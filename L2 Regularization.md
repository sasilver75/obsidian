---
aliases:
  - Weight Decay
  - Ridge Regularization
---
Adds the sum of the "squared values" of the coefficients as a penalty term to the loss function, which encourages smaller, but non-zero coefficients.

Unlike [[L1 Regularization]], L2 doesn't typically result in sparsity (zeroes) in the coefficients. All features have some contribution (even it's small), meaning no coefficients are set to zero.

L2 regularization is particularly effective in handling [[Multicollinearity]], which is when independent variables are highly correlated.

![[Pasted image 20240602222453.png]]