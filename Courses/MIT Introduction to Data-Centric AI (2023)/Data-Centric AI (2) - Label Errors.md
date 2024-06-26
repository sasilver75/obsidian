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

==Confident Learning (CL)== is a framework of theory and algorithms for:
- Finding label errors in the dataset
- Reranking data by likelihood of there being a label issue
- Learning with noisy labels
- Complete characterization of label noise in a dataset
	- We'll focus on this one, in the context of classification with single-labels.
- Data curation (next lecture!)

Key idea: With CL, we can use ANY model's predicted probabilities to find label errors.

![[Pasted image 20240626010153.png]]











