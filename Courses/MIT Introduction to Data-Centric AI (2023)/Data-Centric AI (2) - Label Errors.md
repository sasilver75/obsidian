 https://www.youtube.com/watch?v=AzU-G1Vww3c&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=2

---

This lecture is gonna be pretty fast-paced and jam-packed compared to last time.

![[Pasted image 20240626003833.png]]
Here are some label errors from the [Label Errors](https://labelerrors.com) site.
- Label issues can take many forms!
	- ==Correctable== (Where there's only one clear label in the dataset as far as we can tell, and the given label is just wrong. This is the simplest and clearest case, and we'll focus in this lecture on how to detect these.)
	- ==Multi-Label== (If you have a dataset that you intend to be single layer, but two of your classes are present in an image)
	- ==Neither== (Potentially out-of-distribution; that's an L, but this is a digits dataset! Maybe something is clearly two people, but the two possible labels are rose and apple).
	- ==Non-agreement== (Just "hard" examples)

Agenda
1. Label issues (kinds, why they matter, etc.)
2. Noise processes and types of label noise (How do they become erroneous?)
3. How to find label issues
4. Mathematical intuition for why the methods work
5. How to rank data by likelihood of having a label issue
6. How to estimate the total number  of label issues in a dataset
7. How to train a model on data with noisy labels
8. Label errors in test sets, and the implications for ML benchmarks

OVERAL GOAL: ==Improve ML models trained on data with label issues== (this is most datasets)

----

Why is sorting by loss/error ***not enough*** to find our data labeling errors?
- If we don't regularize our data well... one form of bias is just by overfitting to the data; we might actually have zero loss in incorrectly-labeled data. Recall that we've seen that we can randomly shuffle labels and train a model on it to get zero loss.
- Say you have a dataset of 1,000,000 things, and we've sorted the loss... how do we know where the "cutoff" is? How far down do we go before we stop manually checking?

We need to know how much error is *in* the dataset, and use that information to help with things like sorting to help us find problems in our dataset.

==[[Confident Learning]] (CL)== is a framework of theory and algorithms for:
- Finding label errors in the dataset
- Reranking data by likelihood of there being a label issue
- Learning with noisy labels
- Complete characterization of label noise in a dataset
	- We'll focus on this one, in the context of classification with single-labels.
- Data curation (next lecture!)

Key idea: With CL, we can use ANY model's predicted probabilities to find label errors.

![[Pasted image 20240626010153.png]]


==There are a bunch of different types of noisy labels:==
- User clicked the wrong button (upvote/downvote, rating 1 star instead of 5 stars)
- Mistakes from humans labeling data
- Mis-measurement (you're out reading a geiger counter, and you'er a little off)
- Incompetence (You just don't know the task; if I were asked to rate radiology reports)
- Another ML model's bad predictions
- Corruption (of data)

All of these result in labels being "flipped" to other labels
![[Pasted image 20240626155030.png]]
We can get this matrix of counts, which is the thing that we want to estimate. Once we have this thing, we can do a lot with it!


==Types of label noise (how noisy labels are generated)==
![[Pasted image 20240626155621.png|400]]
- Uniform/symmetric class-conditional noise
	- Uniform random flipping might not be what really happens in the real world. The probability of a "seat" being mislabeled as a "couch" is pretty high, but the probability of a "banana" being mislabeled as a "dog" is low. Uniform is easy, because we're spreading out the (usually small) noise uniformly, but if you have a lot of noise concentrated in a few spots, it can be really hard for the model to learn.
- Systematic/Asymmetric Class-Conditional Label Noise
	- When the errors are concentrated in certain parts of the data; this is the focus on this lecture.
- Instance-dependent label noise
	- Where it doesn't depend on the class, but every example has its own noisy distribution.
	- Requires a lot of assumptions; Out of scope for this lecture



What is Uncertainty?
- *Uncertainty* is the opposite of *confidence*.
	- Confidence is just how high is the predicted probability
	- Uncertainty is how low is the probability for whatever it thinks the label is.
- It's the *lack of confidence" a model has about its class prediction, given a datapoint.*
- Depends on
	- The difficulty of an example (*==aleatoric noise==*)
		- Can have a bad label, a weird data point
	- A model's inability to understand the example (*==epistemic noise==*)
		- Hasn't been trained on data like that, model is too simple

![[Pasted image 20240626155828.png]]

So if we have some predicted probabilities:
- The probability of a noisy label i, given an x and some model $\theta$.

![[Pasted image 20240626160224.png]]

We'll make an assumption; that once we know the true label, we don't need to know the data.
![[Pasted image 20240626160238.png]]
If you were given the true label, there's some constant flipping rate that doesn't matter what x is. If you have images and the true label of that image is fox, then there's some constant probability that the fox is labeled dog, and it doesn't matter what the fox looks like.


![[Pasted image 20240626160540.png]]
If this is class-conditional noise, what would uniform label nosie look like?
![[Pasted image 20240626160656.png]]
This doesn't really happen meaningfully in the real world; where we think that pizzas and tires and trains are pigs.


![[Pasted image 20240626160724.png]]
Many papers make the claim that no matter the noise, deep learning just solves everything!
- (Note the papers are older): Most of these papers focus on uniform random noise, and when you have ==class-conditional noise==, these claims don't turn out to be true. Finding label issues helps meaningfully.

![[Pasted image 20240626160950.png]]
We won't focus on noise in data itself, or in annotator label noise, in this lecture.

![[Pasted image 20240626161018.png]]
The right side will be the focus of this lecture!

Now let's go into how it works!

---

![[Pasted image 20240626161106.png]]
Once you have this matrix describing the joint probability distribution, you can solve everything!


![[Pasted image 20240626161344.png]]
The first thing we're do is to find class thresholds;
- For dogs, we look at every dog image, and see what the predicted probability of the dog image is, from our model. WE do it in an out-of-sample way, meaning we use cross-validation. We ask: "Whats the average, of all of our dogs, of the predicted probability of it being a dog."

This is going to be useful! If a model isn't confidence in a class, but there's some image that has a really high probability of being in that class, according to our model, it's likely that that's a *super probable* member of that class.

![[Pasted image 20240626165356.png]]
For each image, we take a classifier, use cross-validation (or imagine this is a test set) and run each of the images through the model to spit out predicted probabilities. We also have class thresholds.
- We then find which of these pieces of data are likely mislabeled!
If the predicted probability of some image is greater than the class threshold, we label it as class j, though the given label was i. So we have an example labeled i, but it has a really big probability of being labeled j; more than the average confidence of an image being in that class j.
- We count the number of such examples that are higher than the threshold probability
Off diagonals are incorrectly labeled data, and the diagonal is correctly labeled data.


![[Pasted image 20240626165728.png]]
We can normalize these by summing and dividing by the total

![[Pasted image 20240626165746.png]]
This joint distribution is a full characterization of the label noise in the dataset.

![[Pasted image 20240626165818.png]]

If we count up the off-diagonal cells in our count matrix, then we have the count of predicted label errors in our dataset. Say we have 52 predicted errors.

Remember that problem we talked about earlier of: "Oh, let's just rank our images by the loss, and then examine them!" But that raises the question of "Okay, but how many do you examine?"
- ==Now, we can rank images by loss, and take the top 52 or so==! (Since this is the number of predicted noisy images)


![[Pasted image 20240626171355.png]]

![[Pasted image 20240626171552.png]]
If a model had very high loss on example (?), then you would just ... downweight that example during training; this is a model-centric approach. We want to see why the data-centric approach works better than the model-centric probability.
- This is now what we're doing.
- When you do this, this means that any error in your model's predicted probabilities... this noisy estimate...All of the noise is now multiplied directly into your loss function and propagated to your parameters.
	- So when you do your SGD to update your NN, all that error gets propagated into your weights, and you get a noisier model.
	- There's no support for robustness to the noisy model, in the model-centric view.


![[Pasted image 20240626171806.png]]



![[Pasted image 20240626171819.png]]
What would be a perfect predicted probability for a model, for some example?
- As long as we assume this class-conditional error, then the perfect probability would be the flipping rate!
- So the perfect probability for an example labeled i, where the true label is j, is just whatever the probability that j is mislabeled to i.

![[Pasted image 20240626172151.png]]
Let's see what happens when we have imperfect prediction probabilities.
- When we have miscalibrated models that are maybe overconfident in some classes.


![[Pasted image 20240626172359.png]]
Picture 2 is from inside the cockpit, looking out.

![[Pasted image 20240626172516.png]]
((This seems like %accuracy, not %error?))

With these pervasive label errors in test sets in machine learning, what are the implications for Machine Learning?
- Are practitioners unknowingly benchmarking ML using erroneous test sets?
- To answer this, let's consider how ML traditionally creates test sets, and why it can lead to problems for real-world deploy AI models.

Traditional view of a dataset:
- Split into a train set and a test set
- Train a classifier on the train set, and measure its performance on the test set.

Real world view of a dataset:
- We have a dataset with some label errors.
- Those label errors end up in both our training and test set.
- We train our dataset on training sets that have label errors in it, and it learns a perverted view of the world. We predict on the test set that also has label errors, and we again get 100% accuracy. Cool! We must have a good 
- ==Then we have a real-world distribution, which doesn't have a notion of label errors; our model performs poorly.==

They corrected datasets using CL + Mechanical Turk agreement

What if we compared models trained/tested on noisy datasets, and then we took those same models and we benchmarked them on the corrected test datasets?

==But we noticed that large models that would normally strongly outperform stronger models... We notice that on the corrected data, the large models perform WORSE than the smaller models, because it's overfit to the training data.==
- Intuition: The simple model doesn't have the chance/capacity to overfit to the minority noisy data
- So it seems like noisy data negatively impacts large models more

If we increase the noise above a certain threshold, ResNet-18 will start to outperform ResNet-50, if we train on noise and compare on corrected data.