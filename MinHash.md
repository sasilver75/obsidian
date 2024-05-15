A probabilistic data structure; compact representation ==used to efficiently estimate the similarity between sets==.

The core concept of MinHash lies in its ability to approximate the ***Jaccard similarity*** between sets through a hashing technique that preserves similarity.

I've often seen this used in the deduplication portion of dataset construction (eg for pretraining data). 

---
Aside: What is the ==Jaccard Similarity== 
The Jaccard Similarity is a statistic for gauging the similarity and diversity of sample sets. It is defined as the size of the *intersection* divided by the size of the *union* of the sample sets:

![[Pasted image 20240422140936.png|300]]

---

# How it works
- Computing the Jaccard similarity for two sets can be computationally expensive, especially when comparing many sets to eachother.
- MinHash provides an efficient solution by ==converting sets into a more compact form== -- a ==MinHash signature== -- that can make similarity comparisons much faster.

![[Pasted image 20240422141248.png]]
