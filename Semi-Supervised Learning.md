See also: [[Supervised Learning]]

A technique falling under the category of [[Weak Supervision]] in which you initially have a large amount of unlabeled data, and a small amount of labeled data. The small amount of labeled data is used to train an initial model, which then is used to make predictions on the unlabeled data (these are called ==pseudo-labels==). The model is retrained using both the original labeled data and the pseudo-labeled data; This process can be iterative, where the model's predictions on the unlabeled data are continually refined and used to improve the model further.
- Sometimes regularization techniques like [[Consistency Regularization]] are used to encourage consistent outputs for slightly-perturbed versions of the same input, which are used to prevent overfitting to noisy pseudo-labels.

Benefits:
- Reduced labeling cost/Scalability (can scale to datasets where labeling every datapoint is impractical in terms of time or money)
- Improved performance (as compared with the small amount of initial training data)