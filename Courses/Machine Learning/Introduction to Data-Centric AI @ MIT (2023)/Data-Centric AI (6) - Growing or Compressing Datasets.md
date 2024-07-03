https://www.youtube.com/watch?v=XssFXStigTU&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=8
Speaker is Cofounder/CEO of Coactive AI


We're going to focus on labeling today: Specifically *what to label*, and how we grow or compress datasets:
- [[Active Learning]] for growing datasets
- [[Coreset Selection]] for compressing datasets

----


![[Pasted image 20240630112122.png]]
Labeling data is expensive! Especially if we want to have multiple annotators label every piece of data.

![[Pasted image 20240630112152.png]]
If we do random sampling of an unlabeled dataset and label those, we get the blue line; if we do the process of Active Learning (using a technique called maximum entropy), choosing data to label based on model uncertainty, we get massive improvements very very quickly.


So what is [[Active Learning]]?


![[Pasted image 20240630114259.png]]
Active Learning
- The goal is to select the best examples to improve the model.
- Started with a large amount of unlabeled data, we take some small initial subset of the data (eg via random sampling) and we label those examples from our unlabeled pool.
- With this initial labeled set, we train a model on that labeled data.
- We take that new model and apply that model to the unlabeled dataset (perhaps with some selection strategy?) to quantify the informativeness or the value of those datapoints.
- Once we have this metric for each of the unlabeled datapoints, we can select the most valuable ones to label, and label those examples.
- We repeat the process; we retrain the model on this slightly bigger label subset, apply it to the remaining unlabeled data, select the most valuable records to label, etc... until we reach a stopping criteria (eg running out of $ for labeling).


![[Pasted image 20240630115929.png]]
Active learning learns to select as a datapoint to learn from the datapoint that happens between beteween the two that we've seen so far.

![[Pasted image 20240630120128.png]]
Its not even until in this example they did 100 examples of passive learning that they got a decision boundary like the active learning one got in 6 examples.

==PROBLEM:== Note that active learning will save you in terms of labeling costs, but not necessarily for computational costs (indeed, for very large datasets, full active learning can be intractable, as we train large models over and over, and perform inference on the entire unlabeled dataset every time.)
![[Pasted image 20240630120355.png]]
Instead of retraining the entire model for every iteration, we'll instead select the next ***batch*** of data that we feed through a continually-trained model.
- (This helps us with having to retrain the entire model from scratch, but I don't think it saves us from having to do inference over the remaining unlabeled subset of data every iteration).

==PROBLEM:==One of the problems is that depending on our selection strategy, we can still end up with redundant examples (where, for the current model, two selected examples are equally informative, but they're informative in the same way).

We can take the [[Entropy]] of the predicted probabilities for an example, which can be written as:  $-\sum{p(\hat{y}|x, A_r)}log(p(\hat{y}|x, A_r))$ , where $A_r$ is our current model.
((This wasn't clear to me))


==PROBLEM==: The size of the datasets that we have in practice. You might have millions or billions of images uploaded every single day.
![[Pasted image 20240630121226.png]]
We can't process the data from every single Tesla in the world, we have to be thoughtful there!
When we have a massive unlabeled dataset and the thing that we care about is only a very small subset of that (eg fraud in a transactions database), we run into the second bottleneck: Processing all of the unlabeled data we have!

![[Pasted image 20240630121833.png]]
[[SEALS]]
- We might have 10 billion images at Facebook collected over a single span! Just running a single inference pass with a ResNEt-50 model takes 38 exaFFLOPs of computation, or 40 GPU-months using P100 GPUs! The compute costs along of processing that amount of unlabeled data is going to dominate your costs, and it may be too slow! 
You don't actually have to look at all of your data in order to process it!
- Instead, what we can do is start the same way, with a large pool of unlabeled examples, and a small subset of labeled examples that we train an initial model with.
![[Pasted image 20240630121905.png]]
- Instead of applying our selection strategy to ALL of our unlabeled data, we use similarity search to find the closest examples to our currently-labeled (positive) examples, and only consider them!
	- (So given in our initial small labeled dataset, it's going to include some examples of credit card fraud (eg). We do similarity search to find the closest examples to these examples, and only consider labeling them.)
We do this iteratively.
The surprising thing is that even though we only look at a small fraction of the overall data, we reach a similar accuracy as if we had looked at the overall data.
![[Pasted image 20240630122613.png]]
![[Pasted image 20240630122657.png]]
They were able to get to the same accuracy as maximum entropy while only looking at less than 10% of the data.
![[Pasted image 20240630122841.png]]
Again close to the same mean average precision while only having 1% of data considered
![[Pasted image 20240630122858.png]]
Again, with only >0.1% of the unlabeled data.
- So techniques like [[SEALS]] make it possible to do active learning on web-scale datasets.
You can tune the number of neighbors you look at to increase how far you go in a single leap, etc.

![[Pasted image 20240630123845.png]]

----

What is [[Coreset Selection]]?
- Selecting a subset of the data that aims to accurately approximate the full dataset.
- In some ways, it's the reverse of [[Active Learning]].
- ![[Pasted image 20240630124124.png]]
- We can compute some type of heuristic over the large dataset to say "What are the most informative/representative datapoints?" Imagine using (eg) K-Means clustering, then select the subset of representative points based on our strategy. We can then train our model on that final, representative subset of the dataset.
	- Because datasets are often redundant, we can often do this to "compress" our datasets without losing much accuracy.

As a practical example, we can do this on a dataset like [[CIFAR-10]], which is a small, well-curated datasets of 10 evenly-represented classes.

![[Pasted image 20240630124503.png]]

----

![[Pasted image 20240630124512.png]]
- Active Learning as a generative process for figuring out what datapoints to label from some large pool, and Coreset Selection as a way of distilling it down.
- Even with active learning, there are many variants of active learning.
	- There are a bunch of different selection strategies to figure out for different
	- ==Generative Active Learning==: Can we generate the examples that are most important for us to label?
	- ==Active Search== is a sub-area of Active Learning used in drug discovery, where we don't care about model performance -- we just want to find as many positive examples (real drugs) as possible.
	- In ==Hard Example Mining==, we care about generating the most difficult positive examples (using heuristics or other methods).. commonly used in recommendation systems and search problems.
	- In ==Curriculum Learning==, it's the process of selecting the *ordering* of which examples to show to the model.
	  ...and much more.
