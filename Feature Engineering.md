Reference: 
- [WandB blog: Feature Selection in Machine Learning](https://wandb.ai/mostafaibrahim17/ml-articles/reports/Feature-selection-in-machine-learning--Vmlldzo3NjIzNzQ1)

Transforming your data, rather than using a different algorithm.
If you transform some feature x into a log(x) feature, you can get away with keeping (eg) a linear model (which is linear in $\phi(x)$, but not in $x$, but still has many of the nice benefits of a linear model.)

==Feature selection== is the process of selecting a subset of relevant features for use in model construction. This involves identifying and using only those features in your data that contribute most to your predictive ability.
- In supervised learning, feature selection is guided by the performance of the model regarding a specific output variable.
- In Unsupervised learning, it's more challenging, since there's no easy way to evaluate the importance of a feature based on prediction accuracy.

==Feature extraction== involves creating new features by combining or transforming the original features in a way that preserves important information while reducing the dimensionality of the data.


![[Pasted image 20240531155006.png]]

![[Pasted image 20240531155208.png]]
1. Filter Methods
	- Methods that apply a statistical measure to assign a score to each feature. Features are then ranked by this measure and then selected to be kept or removed.
	- Generally independent of the ML algorithm being used, meaning they're often used as a preprocessing step to reduce the dimensionality of data before applying more complex feature selection techniques or training ML models.
	- Example: Using [[Chi-Squared]] to examine the independence of categorical features from a target variable and keeping the K features with the highest score.
2. Wrapper Methods
	- Works by evaluating and selecting a subset of features that contribute most to the prediction power of a model. 
	- Example: ==Recursive Feature Elimination (RFE)== systematically creates models and determines the best or worst performing feature, setting it aside, and repeating the process with the rest of the features.
3. Embedded Methods
	- Unlike filter and wrapper methods that select features before or after the learning process, embedded methods perform it *during* model training.
	- Example: [[L1 Regularization|Lasso Regularization]] (L1 Regularization) has the objective of obtaining a subset of predictors that minimize prediction errors while also shrinking the coefficients of less-important variables to exactly zero.
	- Example: [[L2 Regularization]] (Ridge Regularization), unlike Lasso, includes an L2 penalty that doesn't set coefficients to zero, but reduces their size. It's often used for comparison, but isn't strictly a feature selection method in the way that L1 is.
4. Hybrid Methods
	- Combine the best elements of filter, wrapper, and embedded methods to capitalize on their strengths while mitigating their weaknesses. 

