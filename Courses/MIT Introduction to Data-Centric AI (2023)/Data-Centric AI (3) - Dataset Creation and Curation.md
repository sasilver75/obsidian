https://www.youtube.com/watch?v=R9CHc1acGtk&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=5

---

![[Pasted image 20240627190334.png]]
Remember this count matrix that we saw yesterday; it's an un-normalized joint distribution, basically.
- The 32 means that there were 32 examples that were *labeled* as cow that were *actually* dogs.

![[Pasted image 20240627190517.png]]
Say we had this count matrix, where all of the off-diagonals (mislabels).
We take the 50+50=100 things, and we can get them ...
We can remove them from the dataset and get a better model
What about another way, that has a lot to do with dataset curation?

Unlike in school where you have a dataset where you have ot predict 3 or 4...
You can create a new class called 3-4! So now you just have three classes (1,2,(3,4)), or something.
Or have a second model that then tries to discriminate between the two, once you've predicted an image to be a (3,4)

So this lecture we're thinking about how we change our dataset

Other things we could do:
- If the upper-right 50 were instead a 20...
	- Now we have a different rate of error in each direction. 
	- This chart is more informative that
	- this might happen when we have a "is a" relationship; a missile is a projectile. Someone frmo california is an american. It's more likely that someone in california is mislabeled as being from the us, versus someone in the us being labeled as being in california, because there are 50 states

Other one
![[Pasted image 20240627194350.png]]
Say these were the frequencies of our classes
- We have 4 classes with a lot of data, and many classes (7) that have just a little bit of data
We can train a model to do well on all of them, or you could (from the dataset perspective) join all of these classes together, and make it one clutter class.
- You could then train another model that, once something is predicted as clutter, we give it to this second model to predict the resulting class.


![[Pasted image 20240627194556.png]]
Ranking the off-diagonals in terms of what occurs most
- Mislabeling a missile as a projectile is the most mistaken
	- Note that projectle is also mislabaeled as a missile often too
- Notice that this is an "is a" relationship
- This is a broken dataset! there's no single label that's necessarily "more true"
- ==Interstingly, imagnet actually has two classes in it that are teh same class -- that's just a mistake! There are two "maillot" classes that are different classes!==

So
- Discovering issues in the dataset (classes labeled as other classes)
- Maybe we'd like to merge these classes; it's supposed to be a single-label classification dataset, but there are records with multiple true labels.

![[Pasted image 20240627194852.png]]
This is from Imagenet too.
The one in the left has a curved tail! It's up to the datset curator to choose whether your model needs to be able to discern so finely.
We can use our appropach to identify these two related classes (liekly with mislabels) and perhaps merge those classes

![[Pasted image 20240627195115.png]]

![[Pasted image 20240627195223.png]]

![[Pasted image 20240627195325.png]]
There aren't many cows at the beach in our training dataset, is why!
The problem of "Spurious Correlation"! Our models are looking for *any sort of purchase* to help them predict the label of a tensor of numbers (that we call an image).
- A Spurious Correlation is one that's present in the data that you're training your model on that doesn't remain in the data that you're going to deploy your model on, like in the real world.
- ML models are almost cheaters looking for shortcuts; they're trying to find any sort of pattern in the training data that's highly correlated with the labels. It has NO KNOWLEDGE for the real world, and so will latch onto anything that's available -- you need to be careful about what kind of patterns might be present in a training dataset.

Spurious Correlation are an instance of Selection Bias
- Where the training data isn't fully representative of the real world, or of deployment distributions. It's a distribution mismatch between the training dataset and the the real-world deployment distribution.

Added some more information to the new [[Selection Bias]] note.


---

If you goal is 95% accuracy, how much data do we want? Let's assume that we've already got some data of sample size N with some performance.
- Can we estimate how much data we need?

What if we train our model on some subset of the data, and see how it performs?
Then train it on a bigger subset (including the previous subset, perhaps?), and see how it performs?

Then we can predict our accuracy on the valdiation data as a function of the training data size.

![[Pasted image 20240628004904.png]]

But this might be pretty stochastic, and a result of us perhaps sampling good data

So we do this multiple times, training different versions of the model on differently-centered growing subsets of the data.
![[Pasted image 20240628004946.png]]
Now that we have all of these observations of models and validation accuracies, we can fit a ML model to figure out at what N will I achieve the goal accuracy?

What are the challenges with this?
- Our training data is all between 0 and 1x the size of the training set.... but the actual scenario we're interested in might be something like 3x the size of the training set -- it's a huge extrapolation problem!
	- Some models architectures may not extrapolate well.
- It's an empirical fact that, roughly speaking, the error rate of these models tends to behave according to the following formula:

$log(error) = -a*log(n)+b$
- Where a and b are just hyperparameters
- n is the size of data


Now that we've talked about how much data we might need, and the concerns we might have about collecting data, the final missing piece is *where do I get my labels from?*
- Suppose we hd had a classification dataset with images, where some of them are of a person, and some of them are cars. To label such a dataset, the most natural way is often to use crowdsource workers.
	- For common-sense, you can use anyone
	- For medical images, you'd need to use doctors

So let's say we have 3 annotators for our dataset.
- We show each annotator an image, and 
	- A1: This is a person
	- A2: This is a person
	- A3: (Never sees the image for some reason)

![[Pasted image 20240628005545.png]]
What can we see here?
- The first annotator is doing pretty well! They're giving the labels that most often seem to agree with the other annotators.
- We should be most concerned about annotator 3, because they gave an image a *different label* as another annotator.

==How do we know which annotator is giving us good labels, and which are giving us bad labels?==
==How do we assess the correct label of an image (like the second one) that only has a single annotator prediction?==

Let's introduce some labels.

The annotator predictions are $Y_{ij} \in \{\emptyset\, 1,2,...\}$ 
- $i$ is the example
- $j$ is the annotator

What are some other problems we can think about with annotators?
- If we pay-per-label, annotators are incentivized to work quickly, which makes them more likely to give bad labels.
- Some annotators might be colluding with other annotators -- or maybe one annotator made three accounts on your platform.

==The best way to diagnose these is to insert "gold standard" examples into the data, where you know what the label should be, so that you know when annotators are agreeing with you, or, when making errors from your gold standard, are there multiple annotators that tend to make the same error?==
- This is challenging in its own right, because we're spending a chunk of our budget having annotators label data that we already know the label for.

![[Pasted image 20240628011107.png]]
Given a multi-annotator dataset, what might we be trying to estimate from this dataset?
- ==What is the *consensus label* for each example?== 
	- Simple: Majority vote
	- Problems
		- What if there's just one annotation (eg image 2)
		- How do we account for ties?
			- We could train an additional classifier, and treat that classifier as an additional annotator used to break ties. This classifier can also help with these problems where there's just one annotation -- if the classifier is extremely confident about agreeing with the the one annotation, then we can be more confident in the consensus label.
		- We don't account for annotator quality; if our really good annotator says X, but two really bad annotators say Y, then the consensus label is Y.
- ==What is our confidence in our consensus label? How likely is it to be wrong?==
	- If we were to use majority voting to establish the consensus label, we could consider the percentage of annotators (who annotated that image) that voted for the consensus label.
	- $Agreement(i) = (1/|J_i|) \sum_j{y_{ij}==\hat{y}_{ij}}$ 
- ==What's the quality of a given annotator? What's the overall accuracy of their labels?==
	- Of the images that an annotator annotated, what percentage of those labels agreed with the consensus label?
	- Careful: See A1 in the context of the second image? They're the only annotator, so they're really just agreeing with themselves; they really should be getting points for that.


Lets' talk about a better method called ==[CROWDLAB]==
- The idea of this method is that we're going to form a probability distribution of what we think the true class should be (just like we saw in [[Confident Learning]] based on the features of the example, as well as all the annotations.)

$P(Y_i^*=k | X_i, \{Y_{ij}\})$

The way we form this distribution is as a weighted average of all of our annotators... we convert each annotator's label to a probability distribution.

![[Pasted image 20240628012751.png]]
These underlined weights allow us to downweight annotators that are bad, and up or downweight the classifier based on how good we think it is.
- I think this is the specific-annotator-confdience-weighted probabilities, plus the estimate from the classifier.

And the consensus label just becomes the label assigned to the most likely label.

We can start with majority vote labels, train the classifier on majority vote labels, compute this thing, get new consensus labels using CROWDLAB, retrain the classifier, etc...



