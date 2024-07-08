A situation in regression analysis where two or more predictor/independent variables in a model are highly correlated with eachother, rather than being independent. Often a result of an *over-defined* model (with too many variables), or including predictors that are derived from other predictors.

Results in reduced statistical power, inflated [[Standard Error]]s, and difficulty in determining the true individual importance of predictors.

Can be detected via correlation matrices, variance inflation factor (VIF), etc.
You should remove one of the correlated predictors, combine predictors (eg via [[Principal Component Analysis|PCA]]), or use a [[Regularization]] technique (eg [[L1 Regularization]] (Lasso, which can effectively perform feature selection) or [[L2 Regularization]] (Ridge, which cannot)).