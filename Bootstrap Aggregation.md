---
aliases:
  - Bagging
---
Developed in 1996 by Leo Breiman.

Bootstrap Aggregation (Bagging) is an ML [[Ensemble Learning]] meta-algorithm designed to improve the stability and accuracy of ML algorithms. Used, for example, in the lovely [[Random Forest]] algorithm.

Basic Concept
1. Bootstrap Samples
	- Create multiple [[Bootstrap]] samples from the original dataset (randomly drawn supersets of data with replacement, meaning each draw is independent and duplicate observations are possible).
2. Model Training
	- Train a separate model (often of the same type) on each bootstrap sample. Because the data in each sample is slightly different, each model will learn to make predictions slightly differently, capturing different aspects of the data.
3. Aggregation
	- The model's predictions are then aggregated to form a final prediction -- this is often by averaging the predictions of all models.

Advantages
- Reduces overfitting
- Improves accuracy

Disadvantages
- Increased computational cost
- Model interpretability (~)