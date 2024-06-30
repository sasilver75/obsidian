
[[Precision]]: What proportion of things that are flagged as positive are *actually positive?
- TP / (TP+FP)

[[Recall]]: Of all of the ground-truth positives in the dataset, what proportion did we predict correctly as positives?
- TP / (TP+FN)

When we're dealing with binary classification problems with class imbalance, we might want a summary metric that balances the tradeoff between these two things.
- A common metric to use here is the [[F1 Score]], which is the harmonic mean between precision and recall.

When dealing with imbalanced classes, a related metric that's useful is the [[F-Beta Score]], which is parametrized by a parameter $\beta$ that lets us control the tradeoff between precision and recall.

$F_{\beta}$ = $(1 + \beta^2) * (precision*recall)/(\beta^2 * precision + recall$)

![[Pasted image 20240629221040.png]]
When B=1, this just turns into your usual F1 Score.

Let's talk about the example of taking COVID tests.
- These tests have a false positive rate themselves, remember? If there is one, that'd suck and you'd have to stay home from class.
- But if the test gives a false negative, you could give everyone else in the class covid!
So in this situation, should we weigh precision or recall more heavily? Certainly recall! If there's even a (meaningful) *chance* that we have covid, we want to get a positive result, to avoid spreading the virus! A Beta > 1 gives more weight to recall, while a Beta < 1 favors precision.