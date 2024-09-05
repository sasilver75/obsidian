---
aliases:
  - Density-Based Spatial Clustering of Applications with Noise
---
1996
Martin Ester, Hans-Peter, Jörg Sander, Xiaowei Xu (University of Munich)
[A Density-Based Algorithm for Dicovering Clusters in Large Spatial Databases with Noise](https://www.dbs.ifi.lmu.de/Publikationen/Papers/KDD-96.final.frame.pdf)

----

A clustering algorithm in machine learning developed to address the limitations of other clustering algorithms like [[K-Means Clustering|K-Means]], with the motivation of:
- Discovering clusters of arbitrary shape (not just circular)
- Handling noise and outliers in data
- Not having to require specifying the number of clusters in advance

DBSCAN uses two main parameters:
- $\epsilon$ : The radius of the neighborhood around a point
- MinPts: The minimum number of points required to form a dense region

Process:
- For each point in the dataset:
	- Find all points within $\epsilon$ distance of that point (its $\epsilon$-neighborhood)
	- If there are at least MinPts points within the $\epsilon$-neighborhood, consider this a "core point"
- Form clusters
	- Any two core points that are are within $\epsilon$ distance between eachother are put in the same cluster.
	- Any non-core point within $\epsilon$ distance of a core point is also put in that cluster (these are called border points).
	- Points that are not within $\epsilon$ distance of any core points are labeled as "noise".

Advantages:
- Can find arbitrarily-shaped clusters, not just circular ones.
- Robust to outliers (considered as noise)
- Doesn't require specifying the number of clusters beforehand (clusters are determined by the number of core point clusters, which is determined by density of neighborhoods)

Disadvantages:
- Sensitive to the choice of both $\epsilon$ and MinPts parameters
- Can struggle with datasets that have *varying densities*


Abstract
> ==Clustering algorithms== are attractive for the task of class identification in spatial databases. However, the application to large spatial databases rises the following requirements for clustering algorithms: minimal requirements of domain knowledge to determine the input parameters, discovery of clusters with arbitrary shape and good efficiency on large databases. The well-known clustering algorithms offer no solution to the combination of these requirements. In this paper, we present the new clustering algorithm ==DBSCAN== ==relying on a density-based notion of clusters which is designed to discover clusters of arbitrary shape==. DBSCAN requires only one input parameter and supports the user in determining an appropriate value for it. We performed an experimental evaluation of the effectiveness and efficiency of DBSCAN using synthetic data and real data of the SEQUOIA 2000 benchmark. The results of our experiments demonstrate that (1) DBSCAN is significantly more effective in discovering clusters of arbitrary shape than the well-known algorithm CLARANS, and that (2) DBSCAN outperforms CLARANS by a factor of more than 100 in terms of efficiency. Keywords: Clustering Algorithms, Arbitrary Shape of Clusters, Efficiency on Large Spatial Databases, Handling Noise.