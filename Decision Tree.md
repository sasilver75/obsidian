See: [[Gini Index]] (Gini Impurity)
Notable improvement: [[Random Forest]], [[Bootstrap Aggregation|Bagging]]

A popular machine learning algorithm used for *both* clustering and regression tasks. We learn a series of simple decision rules from the data, resulting in an interpretable model that can handle both numerical and categorical data.

Process:
- Starting with the entire dataset in the root node
- For each feature, find the best split that maximizes ==information gain== or minimizes [[Gini Index|Gini Impurity]] (Really, the "Gini Gain") (for classification) or reduces variance (for regression).
- Create child nodes based on the best split, among all features.
- Repeat the process for each child node until some stopping criterion is met (eg maximum depth reacher, minimum samples per leaf, etc.)

==Information Gain:== A measure of the reduction in entropy (or increase in information) that would result from splitting the data on a particular feature. [[Gini Index|Gini Impurity]] is more commonly used as an alternative.

Advantages:
- Easily interpretable and visualizable
- Handles both numerical and categorical data
- Requires little data preparation
- Can capture non-linear relationships (approximating arbitrary functions as a series of steps, allowing it to model non-linear patterns)
Disadvantages:
- Can create overly-complex trees that don't generalize well
- Can be biased if some classes dominate
- May struggle with highly-correlated features

