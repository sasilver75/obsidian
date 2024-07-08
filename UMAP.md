---
aliases:
  - Uniform Manifold Approximation and Projection
---
February 9, 2018 (10 years after [[t-SNE]])
Leland McInnes, John Healy, James Melville
[UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426)

A dimensionality reduction technique used in ML and data visualization (similar to [[Principal Component Analysis|PCA]], [[t-SNE]]), based on insights from Riemannian geometry and algebraic topology (assumes data has a uniform distribution on a Riemannian manifold that's highly connected).
It's particularly useful for:
1. Reducing high-dimensional data to a lower-dimensional space for visualization.
2. ==Preserving both the local and global structure of the data.==
3. Processing large datasets efficiently (generally faster than [[t-SNE]], especially for larger datasets)

Core steps:
1. Create a weighted k-neighbor graph
	- For each datapoint, find its nearest neighbors, and assign a probability to each neighbor based on distance. Create a weighted graph from these probabilities.
2. Create a low-dimensional representation of this graph
	- Initialize points randomly in low-dimensional space, and use force-directed graph layout algorithms to arrange points, where attractive forces pull similar points together, and repulsive forces push dissimilar points apart.
3. Optimize this representation to be as close as possible to the original high-dimensional data
	- Use [[Stochastic Gradient Descent|SGD]] to minimize the [[Cross-Entropy]] loss between the high and low-dimensional representations.

Abstract
> UMAP (Uniform Manifold Approximation and Projection) is a novel ==manifold learning technique for dimension reduction==. UMAP is constructed from a theoretical framework based in Riemannian geometry and algebraic topology. The result is a practical scalable algorithm that applies to real world data. ==The UMAP algorithm is competitive with [[t-SNE]] for visualization quality, and arguably preserves more of the global structure with superior run time performance==. Furthermore, UMAP has no computational restrictions on embedding dimension, making it viable as a general purpose dimension reduction technique for machine learning.