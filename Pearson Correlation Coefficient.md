---
aliases:
  - Correlation
  - Pearson's r
  - PCC
---
Measures the strength and direction of the linear relationship between two continuous variables. 
- It is a number in $[-1,1]$ where -1 indicates perfect negative linear correlation, 0 indicates no linear correlation, and 1 indicates perfect positive linear correlation. A positive number means "as one variable increases, so does the other," and a negative number means "as one variable increases, the other decreases."
- It's a ==symmetric measure==, meaning r(X,Y) = r(Y,X)
- Assumes that the variables are continuous and normally distributed. Widely used due to its interpretability and connections to linear regression.


For a population:

$\rho_{X,Y} = cov(X, Y)/ (\sigma_X\sigma_Y)$
- where $cov$ is the [[Covariance]]
- where $\sigma_X$ is the standard deviation of X

The covariance term can be expanded so that $\rho$ can be written as:
$\rho_{X,Y} = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]/ (\sigma_X\sigma_Y)$
- Where $\mu_X$ is the mean of X

If we want to expand the standard deviation terms in the denominator as well:
$\rho_{X,Y} = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]/ (\sqrt{\mathbb{E}[X^2] - (\mathbb{E}[X])^2}*\sqrt{\mathbb{E}[Y^2] - (\mathbb{E}[Y])^2})$




Pearson Correlation Coefficient is what's usually meant when we say "correlation" without any qualifier.
- Pearson Correlation: Most commonly used; measures linear relationship between continuous variables. Assumes normally-distributed variables and is sensitive to outliers.
- [[Spearman Rank Correlation Coefficient]]: Measures monotonic relationships, where the variables change together, but not necessarily at a constant rate. Based on ranks of the data, not raw values. Doesn't assume normality and less sensitive to outliers.
- [[Kendall Rank Correlation Coefficient]]: Measures ordinal association between two variables. More robust than Spearman and often used for small samples sizes and when there are many tied ranks.

