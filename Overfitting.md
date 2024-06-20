Overfitting refers to a phenomenon in which our model learns the training data *too well*, including it noise and outliers. As a result, the model generalizes poorly to new, unseen data (either a test set or inference-time data).

Characteristics
- High training accuracy, low test accuracy
- Complex model (models with too many parameters relative to the number of observations are typically thought of as more likely to overfit)
- Low generalization ability

Symptoms
- Significant gap between training performance (very high) and test performance (much lower)
- High variance in errors indicates overfitting; means the models' predictions change dramatically with very small changes in the training data.

![[Pasted image 20240620160813.png|300]]

Mitigations
- [[Cross-Validation]] (eg K-Fold Cross-Validation) to ensure the model can perform well on different subsets of the data.
- Simplify the model, reducing its complexity by selecting fewer parameters or using a less flexible model architecture.
- Apply regularization techniques like [[L1 Regularization]], [[L2 Regularization]], [[Dropout]], [[Early Stopping]], [[Model Pruning]], [[Data Augmentation]], or ensemble methods.
- Use the same model, but collect more data!


![[Pasted image 20240519165821.png]]
Regularization (making the model less flexible, basically) often times means that we add an additional term to our objective function.

The most common regularization (outside NNs) is [[L2 Regularization]]
![[Pasted image 20240519165854.png]]
- Here, we penalize based on the L2 norm of our parameters. This penalizes large-magnitude numbers in the parameters. The intuition is that we want to avoid relying on any specific feature.

![[Pasted image 20240519165956.png]]

![[Pasted image 20240519170217.png]]
Above: Showing an example of with and without regularization. Specifically optimizing a regularized objective vs optimizing a non-regularized objective.
