---
aliases:
  - Cook's D
---
Reference:
- Wikipedia: [Cook's Distance](https://en.wikipedia.org/wiki/Cook%27s_distance)

A method for detecting the influence of a datapoint when performing a least-squares regression analysis.
It can be used to identify influential datapoints (eg outliers), or to indicate regions of space where it would be good to obtain more data points.

Datapoints with large residuals (errors from the model, like outliers) or high leverage may distort the outcome and accuracy of a regression; Cook's D measures the effect of deleting a given observation.
- Points with large Cook's D are considered to merit closer examination.