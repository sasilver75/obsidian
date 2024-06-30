- Given an unlabeled dataset $U$ and a fixed amount of labeling cost $B$, [[Active Learning]] aims to select a subset of $B$ examples from $U$ to be labeled, such that they can result in maximized improvement in model performance.
	- This is an effective way of learning, especially when data labeling is difficult and costly (e.g. medical images).

Interesting Variant:
- ==Generative Active Learning==: Generating the examples that are most important to label?

![[Pasted image 20240527155418.png]]


![[Pasted image 20240630112122.png]]
Labeling data is expensive! Especially if we want to have multiple annotators label every piece of data. This motivates Active Learning, where we intelligently select which data to add to our labeled dataset.



![[Pasted image 20240630112152.png]]
If we do random sampling of an unlabeled dataset and label those, we get the blue line; if we do the process of Active Learning (using a technique called maximum entropy), choosing data to label based on model uncertainty, we get massive improvements very very quickly.

![[Pasted image 20240630114259.png]]
Active Learning
- The goal is to select the best examples to improve the model.
- Started with a large amount of unlabeled data, we take some small initial subset of the data (eg via random sampling) and we label those examples from our unlabeled pool.
- With this initial labeled set, we train a model on that labeled data.
- We take that new model and apply that model to the unlabeled dataset (perhaps with some selection strategy?) to quantify the informativeness or the value of those datapoints.
- Once we have this metric for each of the unlabeled datapoints, we can select the most valuable ones to label, and label those examples.
- We repeat the process; we retrain the model on this slightly bigger label subset, apply it to the remaining unlabeled data, select the most valuable records to label, etc... until we reach a stopping criteria (eg running out of $ for labeling).

