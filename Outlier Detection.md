References:
- [[Data-Centric AI (5) - Class Imbalance, Outliers, and Distribution Shift]]

Example:
- [[Cook's Distance]] for least-squares regression
- Tukey Quartile Method
- Using Z-Scores with a threshold (assumes Normal)
- [[Isolation Forest]]
- Checking the kNN distance for points, and removing those with mean distance to cluster neighbors above some threshold.
- Using an Autoencoder trained on in-distribution data, and throwing out elements that have reconstruction losses (eg L2 distance between input and output) above some threshold, as they're like out-of-distribution.

