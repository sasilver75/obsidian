---
aliases:
  - T-SNE
  - T-Distributed Stochastic Neighbor Embedding
---
August 11, 2008
Laurens van der Maaten and [[Geoff Hinton]]
[Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)

 >"That's what you use T-SNE for, to try to separate out otherwise overlapping clusters!"
> - Some random twitter user

----

A dimensionality reduction algorithm (like [[Principal Component Analysis|PCA]], [[UMAP]]), t-SNE is based on the idea of preserving similarities between data points while reducing dimensionality. It preserves local structure, but does not strongly preserve global structure (as the later [[UMAP]] does).
- Note: Because of random initialization, results are non-deterministic and can vary between runs. It also has a tendency to crowd points in the center of the visualization.

Compared with [[Principal Component Analysis|PCA]], t-SNE is better at preserving local structure and non-linear relationships, whereas PCA is faster, simpler, and preserves *global* structure better. t-SNE is primarily used for visualization, while PCA can be used for feature extraction.

Compared with [[UMAP]], UMAP tends to preserve global structure better, and UMAP is generally faster and scales better to larger datasets.

Core steps: 
1. Compute similarities between data points in high-dimensional space
	- Calculate pairwise similarities using [[Gaussian Distribution]]s, and convert distances to conditional probabilities.
2. Create a similar probability distribution in low space
	- Initialize points randomly in low-dimensional space and use [[Student's t Distribution]] to compute similarities in this space.
3. Minimize the difference between these distributions
	- Minimize the [[Kullback-Leibler Divergence|KL-Divergence]] between the high and low-dimensional distributions, and use gradient descent for optimization.


Abstract
> We present a new technique called “t-SNE” that ==visualizes high-dimensional data by giving each datapoint a location in a two or three-dimensional map==. The technique is a variation of Stochastic Neighbor Embedding (Hinton and Roweis, 2002) that is much easier to optimize, and produces significantly better visualizations by reducing the tendency to crowd points together in the center of the map. t-SNE is better than existing techniques at creating a single map that reveals structure at many different scales. This is particularly important for high-dimensional data that lie on several different, but related, low-dimensional manifolds, such as images of objects from multiple classes seen from multiple viewpoints. For visualizing the structure of very large data sets, we show how t-SNE can use random walks on neighborhood graphs to allow the implicit structure of all of the data to influence the way in which a subset of the data is displayed. We illustrate the performance of t-SNE on a wide variety of data sets and compare it with many other non-parametric visualization techniques, including Sammon mapping, Isomap, and Locally Linear Embedding. The visualizations produced by t-SNE are significantly better than those produced by the other techniques on almost all of the data sets.

