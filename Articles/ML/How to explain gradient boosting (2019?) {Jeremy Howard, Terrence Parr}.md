#article 
Link: https://explained.ai/gradient-boosting/index.html
Link 2: https://www.gormanalysis.com/blog/gradient-boosting-explained/

------

Roadmap
- [[Gradient Boosting Machine]] (GBMs) are currently very popular and so it's a good idea for ML practitioners to know how they work.

Understanding the mathematical machinery is tricky, and unfortunately, these details are needed to tune the hyper-parameters.

Our goal in this article is to explain the intuition behind gradient boosting, provide visualizations for model construction, explain the math as simply as possible, and answer thorny questions such as why GBMs perform "gradient descent in function space."


Three morsels:

1. Gradient Boosting: Distance to target
	- The most common form of GBM optimizes the Mean Squared Error (MSE), also called the $L_2$ loss or *cost*. 
	- ==A GBM is a composite model that combines the efforts of multiple weak models to create a strong model, and each additional weak model reduces the mean-squared-error (MSE) of the overall model.== 
2. Gradient Boosting: Heading in the right direction
	- ==Optimizing a model according to MSE makes it chase outliers, because squaring the difference between targets and predicted values emphasizes extreme values.==
	- When we can't remove outliers, it's better to optimize the Mean Absolute Error (MAE), also called the $L_1$ loss, or cost.
3. Gradient Boosting: Performing gradient descent
	- The previous two articles give the intuition behind GBM and the simple formulas; no attempt was made to show how to abstract out a generalized GBM that works for any loss function, though! ==This article demonstrates that gradient boosting is really doing a form of gradient descent, and, therefore, is in fact optimizing MSE/MAE depending on the direction vectosr we use to train the weak model.==
4. FAQ




























