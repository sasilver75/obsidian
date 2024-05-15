Related (IMO): [[Semi-Supervised Learning]]

A technique in ML where a model is trained using its own predictions as ==soft labels==, typically to improve performance and robustness. A variant of the broader concept of [[Distillation|Knowledge Distillation]], where a smaller student model learns to mimic a larger teacher model. In self-distillation, the same model serves *both* roles at different stages of training.

(Note: ==soft labels== are probability distributions over classes, rather than ==hard labels== (eg [[One-Hot Encoding]] vectors). These carry more information about a model's *confidence* in its predictions)

---
Example
*Consider a classification problem where a trained NN is used to predict the classes of some unlabeled training examples, producing soft labels. These labels, which include information about the model's confidence in predictions, are then use to retrain the model. Repeat.

---