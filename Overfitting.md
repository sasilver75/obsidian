Overfitting refers to a phenomenon in which our model learns the training data *too well*, including it noise and outliers. As a result, the model generalizes poorly to new, unseen data (either a test set or inference-time data).

Characteristics
- High training accuracy, low test accuracy
- Complex model (models with too many parameters relative to the number of observations are typically thought of as more likely to overfit)
- Low generalization ability

Symptoms
- Significant gap between training performance (very high) and test performance (much lower)
- High variance in errors indicates overfitting; means the models' predictions change dramatically with very small changes in the training data.

Mitigations
- [[Cross-Validation]] (eg K-Fold Cross-Validation) to ensure the model can perform well on different subsets of the data.
- Simplify the model, reducing its complexity by selecting fewer parameters or using a less flexible model architecture.
- Apply regularization techniques like [[L1 Regularization]], [[L2 Regularization]], [[Dropout]], [[Early Stopping]], [[Model Pruning]], [[Data Augmentation]], or ensemble methods.