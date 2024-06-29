https://www.youtube.com/watch?v=m1tEl7a1atQ&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=6

Data-centric AI is all about improving the performance of our models (via considering the data) -- but we have to be able to evaluate the performance of models to do this!

----

![[Pasted image 20240628164018.png]]
We're going to focus on step 5 here, as well as steps 6 and 7 after you find problems.

Topics:
- Evaluation of ML models
- Handling poor model performance once we do detect a problem on some subpopulation in the data
- Measuring influence of individual datapoints on the model, in order to attribute model performance to training examples.

----

Recap of Multi-Class Classification:
- Given a training dataset $\mathcal{D}$ with $n$ examples: $(x_i, y_i) ~ P_{XY}$
- Our goal is to train a model $M$ using this data, which, when given an example with NEW feature values $x$, produces a vector of predicted class probabilities whose $k$th entry approximates $P(Y=k|X=x)$.

For a particular loss function, we seem a model that optimizes:
![[Pasted image 20240628164251.png]]
Where the loss function here is evaluated on some held-out test data.

![[Pasted image 20240628164328.png]]
- We know that all of these can be violated in the real world.

The Loss Function may be a function of:
- The predicted class deemed most likely, for x
	- [[Accuracy]], [[Balanced Accuracy]], [[Precision]], [[Recall]]
- The predicted probabilities for each class, for x
	- Log-Loss, [[ROC-AUC]]/AUROC, calibration error

![[Pasted image 20240628164645.png]]
- It's harder to look at two confusion matrices and say "this model is better than that model"; a single metric lets us do it, but... that's a wildly imperfect comparison.

==Thinking about HOW you evaluate models is as important as thinking about what models to apply, and how to improve them. In real applications, model evaluation has a huge impact.==

Let's consider "Fraud" vs "Not-Fraud" binary classification tasks:
- Why would we not want to choose "overall accuracy" as the evaluation metric?
	- We care more about missing fraud, than we do about accidentally classifying real transaction as fraud (which we can confirm by sending a text message). We probably want a high-recall solution.


![[Pasted image 20240628165816.png]]
Pitfalls when evaluating model
can lead to:
- Model that performs well only on light-skinned men and women

Underperforming Subpopulations
- A ==data slice== is a subset of the dataset that shares a common characteristic 
	- Also referred to as ==cohort, subpopulation, or subgroup==
Examples
- Data captured by one sensor vs another, one location vs another
- Factors in human-centric data (age, gender, socioeconomics, age)

Can we just "delete" all of this slice information from our dataset, and pretend that we've solved the problem?
- There might still be aspects of your dataset that correlate heavily with the removed information, so your model still depends on information from the slice, just explicitly.
- It's better to keep that information in, and analyze what's happening in the model, over the slices.

## Ways to improve performance for a particular slice/cohort that's performing poorly:
1. Try a more flexible model
2. Over-sample/up-weight examples from the minority subgroup during training
3. Collect additional (real) data from the subgroup with poor performance
4. Measure/engineer additional features that let the model perform better for problematic cohorts

![[Pasted image 20240628170633.png]]
If you were to fit a linear model, it would have to make a tradeoff over which examples it's going to give high error to (because the data itself isn't linearly separable).

![[Pasted image 20240628171432.png]]


![[Pasted image 20240628171523.png]]
(This is going to be expensive!)

![[Pasted image 20240628171804.png]]


But how do we detect subpopulations that are underperforming?

![[Pasted image 20240628171918.png]]
- Sorting and examining data by loss value is always a good idea when training an ML model; things you're making bad predictions on might be mislabeled or poorly represented in the training data.
- We can also apply clustering techniques to uncover cluster that share common themes (and then perhaps attach loss statistics to these clusters).

![[Pasted image 20240628172152.png]]


## So why did my model get a *particular prediction* wrong?
1. Given label may be incorrect, and our model made the correct prediction
	- Correct the label
2. The example might not belong to *any* of the K classes at all! (or is fundamentally not predictable, like too-blurry of an image)
	- Toss this example from the dataset
	- Add an "other" clutter class to the dataset if many such examples
3. Example is an outlier (no similar examples in the training data)
	- Toss examples if similar examples would *never* be seen in deployment
	- Otherwise, collect additional training data in that subdistribution
	- (🤢) Apply data transformation to make outliers' features more similar to other examples (eg normalization of numeric features, deleting a feature)
4. The type of model we're using is suboptimal for such examples
	- Diagnose
		- Up-weight similar examples, or even duplicate them many times in the dataset
		- Retrain the model
		- See if the model can now classify this example correctly
		- If this didn't meaningfully help, you probably need to consider a different model architecture if you care about this example.
5. Data has a bunch of other examples with nearly identical features, but different labels. (Imagine two ~identical datapoints with different labels!)
	- This can be two images of keyboards, where one is labeled "keyboard", and the other "space bar". Your model is inevitably going to have non-zero loss on these classes.
	- We can try to define the classes more distinctly (perhaps giving annotators more instruction as to how to label similar examples).
	- Merge classes if needed.
	- Measure extra features that enrich the dataset and make these examples distinguishable.



## Influence of individual datapoints on the model
- A simple way to answer this question is to leave out the piece of data, retrain the model, and see how our predictions change as a result. This is the ==Leave One Out influence (LOO)==
	- We can also see how the parameters changed as a result, but that's less interesting in a world where people are using NNs with a large number of mostly-uninterpretable parameters.
![[Pasted image 20240628174750.png]]
Why might this not be the end-all-be-all of evaluating the impact of an element of data?
- If we have (eg) 5 very similar data points in our dataset, and we delete 1/5 of them, then the model might still perform well on that subpopulation.

==Data Shapely== ("Shap-Lee") value
- From economics/game theory is a way of determine how much of the profits from a project should be allocated to each coworker.
	- Satisfies some nice axioms:
		- If you add an additional person to your set who doesn't contribute anything, you want to make sure that existing returns are the same.
		- Want to be able to evaluate how much an employee is contributing.
- We want to be able to determining the leave-one-out influence of datapoint in a hypothetical subset of the dataset. Then we average these values over all possible subsets of the dataset, which gives us a pretty broad understanding of how the datapoint contributes in all possible contexts.

Example:
- Suppose there are two identical datapoints in a dataset, and deleting both elements harms model accuracy, but omitting just one of them doesn't.
- The ==Leave-One-Out== influence for the datapoint would be 0
- The ==Data Shapeley== influence would be meaningful, because there would be subsets of data where neither of the data points are present.


![[Pasted image 20240628180414.png]]
((OMG, it's Cook's Distance lol))
