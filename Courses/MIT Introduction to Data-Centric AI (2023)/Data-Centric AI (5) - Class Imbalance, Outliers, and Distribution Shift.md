https://www.youtube.com/watch?v=xnSFpswWm_k&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=6

----


==Class Imbalance==
- When certain classes in your dataset are more present than others.
- Examples
	- A canonical example of Class Imbalance is in Fraud Detection tasks -- The vast majority of transactions are not fraudulent.
	- A similar example might be for certain rare medical diagnoses.
	- Car crashes in self-driving car data; certain events on the road are less prevalent than others.

Relation between Class Imbalance and Underperforming Subpopulations
- Underperforming Subpopulations is more about slices of the data that *don*'t explicitly involve the labels. Maybe in your dataset you do a worse job of detecting skin cancer on people of color, in your dataset.

So how do we evaluate a model trained on a class-imbalanced dataset?
- If we're doing fraud detection, and our metric is accuracy, then a classifier that always predicts *no fraud* is going to have something like 99.8% accuracy!
- So you have to work to define an evaluation metric that makes sense for your problem set.
	- The author suggests the F-Beta Score

[[Precision]]: What proportion of things that are flagged as positive are *actually positive?
- TP / (TP+FP)

[[Recall]]: Of all of the ground-truth positives in the dataset, what proportion did we predict correctly as positives?
- TP / (TP+FN)

When we're dealing with binary classification problems with class imbalance, we might want a summary metric that balances the tradeoff between these two things.
- A common metric to use here is the [[F1 Score]], which is the harmonic mean between precision and recall.

When dealing with imbalanced classes, a related metric that's useful is the [[F-Beta Score]], which is parametrized by a parameter $\beta$ that lets us control the tradeoff between precision and recall.

$F_{\beta}$ = $(1 + \beta^2) * (precision*recall)/(\beta^2 * precision + recall$)$


When B=1, this just turns into your usual F1 Score.