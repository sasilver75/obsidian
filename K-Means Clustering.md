---
aliases:
  - K-Means
---
1957
Bell Labels
{No paper, but used as a technique for pulse-code modulation in signal processing}

---

One of the most popular clustering techniques, which uses iterative refinement of clusters to partition a dataset into a *predetermined number of k clusters*. Goals are:
- Group similar datapoints together
- To minimize the within-cluster variance (dispersion)

(Only requires specifying one main parameter, K -- the number of clusters to form)

Process:
- At initialization, we randomly select K points from the dataset as our initial cluster centroids.
- Repeat (until termination condition):
	1. Assignment step
		- Assign each datapoint to the nearest centroid, using Euclidean distance.
	2. Centroid update step
		- Recalculate the centroid of each cluster by taking the mean of all points assigned to the cluster.
- The above step repeats until either:
	- The centroids no longer move (significantly)
	- Some maximum number of iterations is reached

Metrics and techniques:
- Within-Cluster Sum of Squares (WCSS): Sum of squared distances between each point and its assigned centroid.
- Elbow method: A technique to help choose the optimal K by plotting WCSS against K and looking for an "elbow" in the curve (resulting in best bang for your buck between reduced WCSS and minimal number of clusters).

Advantages:
- Easy to understand and implement
==Disadvantages==:
- Requires specification of K in advance
- Sensitive to initial centroid placement
- Assumes clusters are spherical and of similar size
- Can struggle with outliers

Extensions:
- K-Means++: Improves initialization to choose better centroids
- Mini-batch K-Means: Uses small random batches to reduce computation time
- Fuzzy C-Means (FCM): Allows datapoints to belong to multiple clusters, with different degrees of membership.


![[Pasted image 20240519183717.png]]