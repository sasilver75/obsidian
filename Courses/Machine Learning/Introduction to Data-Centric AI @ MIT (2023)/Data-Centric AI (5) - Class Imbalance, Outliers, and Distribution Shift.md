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

$F_{\beta}$ = $(1 + \beta^2) * (precision*recall)/(\beta^2 * precision + recall$)

![[Pasted image 20240629221040.png]]
When B=1, this just turns into your usual F1 Score.

Let's talk about the example of taking COVID tests.
- These tests have a false positive rate themselves, remember? If there is one, that'd suck and you'd have to stay home from class.
- But if the test gives a false negative, you could give everyone else in the class covid!
So in this situation, should we weigh precision or recall more heavily? Certainly recall! If there's even a (meaningful) *chance* that we have covid, we want to get a positive result, to avoid spreading the virus! A Beta > 1 gives more weight to recall, while a Beta < 1 favors precision.

----

==What  are some techniques for combatting class imbalance?==
- ==Under-sample the majority class==
	- This is throwing some data away, but it actually works pretty well in many cases.
- ==Over-sample the minority class==
	- You don't throw away data here, but sometimes "you might need to be careful about it resulting in unstable training."
- You could ==create synthetic data== (eg using data augmentation) for the under-represented class.
- "==[[Sample Weights]]==":  If you have a loss function that's a the sum of per-datapoint losses, instead of that, we can multiply the loss by sample-dependent weights (so that we underweight the majority class or overweight the minority class).
- ==[[Balanced Mini-Batch Training]]:== When choosing your subset of data to compose a mini-batch, instead of sampling uniformly at random, we weight how we choose our data so that we have an over-representation of the minority class (very similar to over-sampling a minority class).

----

What is an [[Outlier]]?
- A datapoint that doesn't fit the distribution; it doesn't look like the rest of the data (usually on at least one feature dimension).
How might we end up with outliers in our data?
- Bad Sensors (Can we toss these?)
- Gaps in data (eg missing fields in a tabular dataset) (Can we impute these?)
- Rare events (these things we don't want to drop!) (Is it possible to automatically distinguish between these are the former categories? It's not always easy to tell if data is a "rare event" or if it's "bad data")
Often we handle these problems in two steps:
- Finding outliers
- Handle outliers

There are some model-centric techniques for dealing with outliers, but we want to focus more on the data-centric side -- focusing first on *identifying outliers.*

Tasks
- ==[[Outlier Detection]]==
	- We have an unlabeled dataset, and we want to find the subset of data that's "out of distribution" with respect to the rest.
- ==[[Anomaly Detection]]==
	- A similar task to outlier detection; Given the in-distribution set separately, given a new datapoint $x_*$, determine whether that datapoint is in-distribution or not.
	- Interestingly, you *could* cast anomaly detection as outlier detection by concatenating our given $x_*$ with our in-distribution dataset $\{x_i\}$, and then run an outlier-detection algorithm in that combined dataset, considering what happens to $x_*$. But often we use anomaly-detection-specific algorithms.

What makes Anomaly detection different from a standard supervised learning problem?
- Why isn't it just binary classification? We aren't given any "negative", out-of-distribution examples in the training set that we can train on.

==A simple algorithm devised by John Tukey in 1977 to identify outliers==
- Concerned with real-valued, scalar data
- Suppose we have a bunch of data spread out on the real number line.
- The algorithm looks at the lower-quartile, upper quartile.
- The Q1-> Q3 range is the Inter-Quartile Range,
	- Say we look at 1.5x the IQR on either side, and the the things past that are "cutoffs".
	- We say that anything inside that is "in distribution," and anything outside that is "out of distribution"
![[Pasted image 20240629224916.png|300]]
This is a simple, heuristic way to find outliers.

Next let's talk about ==Z-Scores== for outlier detection
- The Z-Score is just the number of standard deviations your datapoint is from the mean.
- We just say that when your abs(Z-Score) is greater than some threshold (eg 3), then it's an outlier.
	- If your data is normally distributed, 99.7% of your data falls within 3 standard deviations of the mean.
- This only makes sense with low-dimensional data -- but you can do this same thing in multiple dimensions -- perhaps a particular feature has outliers in a tabular dataset.

A neat algorithm is something called an [[Isolation Forest]] (cool name)
- If you have a random decision tree, and if take all of our data and see how far down the decision tree we have to go to end up with only that single piece of data, isolated from the rest of the data... the more a piece of data is an outlier, the shallower you'll have to go down the decision tree!

![[Pasted image 20240629225522.png|200]]
Given a dataset, choose a feature at random, choose a cutoff at random, and then create that decision boundary in your tree.
![[Pasted image 20240629225627.png|300]]
Making up some draws here, but see that we're already isolated that point on the right that isn't part of the two clusters? Nice!

Think: Would it make sense to apply this to image data, where we're working with raw pixel values (lots of features, where every feature is a pixel value for some channel)?
- No! You probably want to embed your image into a lower-dimensional space where the place the image ends up in the embedding space means that similar images end up near eachother.

Note that all the methods we've talked about so far involve computing some sort of ==Score== to our datapoints, and then we ==choose== some threshold for a cutoff.
- In a moment, we'll talk about how to choose that threshold.


Another method for identifying outliers is looking at kNN distance:
![[Pasted image 20240629225930.png|300]]
We assign a score to each datapoint by looking at its k nearest neighbors, and assign a score to our datapoint by computing the average distance to its nearest neighbors.

![[Pasted image 20240629230008.png|300]]
Looking at the outlier, the mean distance is much higher than the ones in the cluster.


There's another method that falls into the category of *reconstruction-based methods*
- Related to [[Autoencoder]]s, where we map a high dimensional input to a lower-dimensional latent space, and then learns to remap that latent space to our original feature space, with minimal loss.
![[Pasted image 20240629230354.png]]
Suppose we trained this on handwritten digits from MNIST.
- A 5 that gets passed through comes out as a 5
- But if we feed an "M" though, it might come out as something a little weird -- maybe something that looks sort of like a 3 might come out. If we look at the reconstruction loss (eg the L2-distance between the input and output), it's going to be much higher for this strange, out-of-distribution input!

-----

Let's now talk about [[Covariate Shift|Distribution Shift]]/Covariate Shift!
- This is a real problem that probably occurs in every ML task, to various degrees.

What is distribution shift?
- When we train on one distribution, and our model at inference time makes predictions on a different distribution. In short, when your input distribution for training and test don't match, but the relationship between your inputs and outputs don't change.

Examples
- Problems where the distribution of data is time-variant (eg finance)
- A dataset for CV on X-ray images that uses a particular brand of X-ray, but then you deploy it and people around the world use different X-ray machines with their own quirks.
- Self-driving cars, when your training data doesn't have any snowy driving conditions, but you have to deal with them during driving.

((Im my opinion, some of these are just distribution mismatch, as opposed to distribution shift/drift. I usually think of this as your deployed model initially matching the test-time distribution, but over time your model becomes stale as the distribution changes))

- We should be monitoring our data during deployment, and looking for outliers.
- We should also be monitoring our model and the predictions that it's making.

----

Another type of shift is called [[Concept Shift]]
- This is when your probability of y, given x,  (the relationship between inputs and outputs) is different from train and test time ... but the distribution of inputs themselves do not change.
- $P_{train}(y|x) \neq P_{test}(y|x)$ , but $P_{train}(x) = P_{test}(x)$ 

Suppose we have a two-class classification setting, and we're drawing the two classes across two features.
![[Pasted image 20240629232813.png|300]]
What if at testing time, the *boundary* changes (the points stay in the same place)

![[Pasted image 20240629232838.png|300]]
It's the same distribution of data (with respect to (x1, x2)), but the labels have changed, as we "moved the boundary"!

It's hard to find examples where the input data distribution *does not* change from train to test time, so there's often [[Covariate Shift]] involved as well.

Examples:
- The rating for a song on Spotify
- Predicting the popularity of celebrities based on some features, and a celebrity's popularity goes down or up when they do {some public action} (maybe the input data you're using hasn't changed, but the way people)
- Stock price (If you're predicting a company's stock price based on some fundamentals about the company)

((This in the real world feels like some input variables *have changed*, but they're not input variables that you've captured in your training set.))


