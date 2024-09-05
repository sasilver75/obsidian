Plots the trade-off between precision and recall across all thresholds. As we update the threshold for positive predictions, precision and recall change in opposite directions.

A higher threshold leads to higher precision (fewer false positives), but lower recall (more false negatives), and vice versa.

The area under this curve, [[PR-AUC]], summarizes performance across all thresholds.

![[Pasted image 20240525215645.png]]
Above:
- The standard PR curve (left) plots precision and recall on the same line, starting from the top-right corner (high precision, low recall) and moving towards the bottom-left corner (low precision, high recall).
- Eugene Yan prefers the variant on the (right), where precision and recall are plotted as separate lines, making it easier to understand the trade-off between precision and recall, since they're both on the y-axis.



