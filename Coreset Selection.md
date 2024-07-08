Selecting a subset of the data that aims to accurately approximate the full dataset.
- In some ways, it's the reverse of [[Active Learning]].
![[Pasted image 20240630124124.png]]
- We can compute some type of heuristic over the large dataset to say "What are the most informative/representative datapoints?" Imagine using (eg) K-Means clustering, then select the subset of representative points based on our strategy. We can then train our model on that final, representative subset of the dataset.
	- Because datasets are often redundant, we can often do this to "compress" our datasets without losing much accuracy.

As a practical example, we can do this on a dataset like [[CIFAR-10]], which is a small, well-curated datasets of 10 evenly-represented classes.

![[Pasted image 20240630124503.png]]
