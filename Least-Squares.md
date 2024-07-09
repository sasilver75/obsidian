---
aliases:
  - Ordinary Least Squares
  - OLS
---


A method primarily used for model fitting in regression.
It's a method for estimating the parameters of a model by minimizing the sum of squared differences between observed values and predicted values.

For example:
Minimize $\sum{(y_i - (\beta_0 + \beta_1x_i)})^2$

This objective is sensitive to outliers and may overfit with too many parameters.
Equivalent to maximum likelihood estimation under Gaussian error assumptions.

==Ordinary Least Squares== is the most common form, used for linear regression.
- One of its key advantages is that it can be solved directly, without need for iterative optimization of the parameters.

