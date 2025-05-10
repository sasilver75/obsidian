References:
- Video: [Intuitively understanding the cross-entropy loss](https://youtu.be/Pwgpl9mKars?si=JyHt2cKJhz_hksNm)
- Video: [Luis Serrano's KL Divergence Video](https://www.youtube.com/watch?v=sjgZxuCm_8Q&list=WL&index=24&t=9s)
	- Has some great intuition on entropy, cross-entropy, KL divergence
- [Artem Kirsanov: They Key Equation Behind Probability Video](https://youtu.be/KHVR587oW8I?si=kPgUbkjrIk8cQ9ze)
- 

See also:
- [[Kullback-Leibler Divergence]]
- [[Entropy]]

==TODO==: "Binary Cross Entropy"/BCE

$H(P,Q) = -\sum_{x\exists{X}} P(x)log(Q(x))$ 

- Cross entropy measures the amount of "information" required to identify an event from set $X$ if the wrong distribution $Q$ is used instead of the true distribution $P$.
- The ==TOTAL== expected "information content" (or surprise) you would experience when observing data with distribution $P$, but predictions are based on distribution $Q$. ==This metric encompasses both the inherent unpredictability of $P$ itself (its [[Entropy]]) AND the *additional* unpredictability introduced by any discrepancy between $P$ and $Q$ ([[Kullback-Leibler Divergence]].==

The cross entropy is the average number of bits needed to encode data coming from a source with distribution $P$ when we use model $Q$.

Cross entropy is a ==non-symmetric measure==.

-------
Comparing with KL Divergence (see [[Kullback-Leibler Divergence]] for more)


$H(P, Q) = H(P) + D_{KL}(P||Q)$ 
CrossEntropy(P,Q) = Entropy(P) + KLDivergence(P, Q)

Thus minimizing the Cross-Entropy and KL Divergence are equivalent in terms of an objective function for (eg) classification.

-----

