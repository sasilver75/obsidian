---
aliases:
  - Gini Impurity
  - Gini Coefficient
---

A measure of inequality or impurity used in various fields, from economics to machine learning.

In the context of decision trees, the Gini index measures the impurity of a node, indicating how mixed the classes are in the node. ==Lower Gini Index signifies a *purer node*, where one class predominantly occupies the node.==

We use the Gini Index to determine the best split at each node: We evaluate potential splits by calculating the Gini Index for each possible split, and select the one resulting in the lowest Gini Index (the purest child nodes), with the goal of building a tree that effectively separates classes.

![[Pasted image 20240603193129.png]]

### Example

If we have a node with 10 instances, 6As and 4Bs, the Gini Index is:
$Gini = 1 - (0.6^2 + 0.4^2) = 1 - 0.52 = .48$
If a split then results in two child nodes with Gini Indices of .2 and .3, and the *weighted* averages of these indices is lower than .48, the split is considered good/useful.