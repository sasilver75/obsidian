
The number of degrees of freedom actually needed to describe your data, regardless of the space it's embedded in.

Classic example:
> *"A 2D surface (like a sheet of paper) curled up in 3D space still only has **2 intrinsic dimensions;** you only need 2 numbers to describe any point on it, even though it lives in 3D space. The [[Ambient Dimension]] is 3, but the intrinsic dimension is 2.*


Say you have text [[Embedding]]s in 1536-dimensional space (OpenAI's embedding size)...
- The intrinsic dimension asks: ==How many dimensions are actually "doing work?"==
- The embeddings don't fill that space uniformly, they lie on some lower-dimensional [[Manifold]] within it. 
	- The intrinsic dimension estimates what that manifold's dimension is.

In practice, it might be something like 30-100 for a typical embedding space, even if the vectors are 1536-dimensional.


Why care about intrinsic dimensions?
- ==Model Evaluation==: A higher intrinsic dimension means your embeddings are encoding more ==independent factors of variation==, which generally means richer representations.
- ==Compression==: Tells you how much you could reduce dimensionality without losing structure.
- ==Training Dynamics==: Tracking intrinsic dimension during training tells you whether your model is learning richer or more collapsed representations over time.
	- Collapsed embeddings (low intrinsic dimension) are a known failure mode in [[Contrastive Loss|Contrastive Learning]].
- ==Generalization==: Some research links lower intrinsic dimension to better generalization.


The failure mode it detects: [[Representational Collapse]], where your model learns to map everything to nearly the same vector. 
- The intrinsic dimension drops towards 0. You should be able to catch this during training by logging it per-epoch.


# How do you determine the intrinsic dimension of an embedding?
- Several estimators exist, all based on analyzing local distance structure. The most common:
	- TwoNN (Two Nearest Neighbors): The simplest and most popular
		- For each point, look at the distance to its 1st and 2nd nearest neighbor:
			- If the data is locally d-dimensional, the ratio `r = dist_2nd / dist_1st` follows a predictable distribution, depending on d.
			- Fit this distribution across all points to estimate d.
		- Intuition: In higher dimensions, the ratio between nearest neighbor distances behaves differently than in low dimensions; a 2D neighborhood looks different from a 10D one.
	- MLE ([[Maximum Likelihood]] Estimation)
		- For each point, look at distances to its k nearest neighbors and fit a Poisson process model to how those distances are distributed. The MLE of the rate parameter gives you local dimensionality. Average across all points.
	- [[Principal Component Analysis|PCA]]-based methods
		- Global approach: Look at the eigenvalue spectrum of the covariance matrix. Count how many principal components are needed to explain ~95% of variance. Simpler but assumes the manifold is globally linear, which is often wrong.
	- Correlation Dimension
		- Look at how the number of neighbor pairs within radius r scales as r grows. In a d-dimensional space, it scales as r^d, so just fit the exponent.
- These methods all reduce to: ==Look at how distances or neighbor counts scale locally, and infer what dimension would produce that scaling.==
	- ==The [[K-Nearest Neighbors|kNN]] search is the shared bottleneck,== which can be accelerated by tools like [[FAISS]], which is an [[Approximate Nearest Neighbor Search|Approximate Nearest Neighbor]] tool that can speed up this comparison.
		- FAISS uses [[Inverted File Index]] (IVF), [[Product Quantization]] (PQ), and [[Hierarchical Navigable Small Worlds]] (HNSW), along with GPU acceleration to parallelize distance computations, getting orders of magnitude speedup over CPU.





