---
aliases:
  - KL-Divergence
  - Relative Entropy
  - Forward KL Divergence
---
References:
- https://youtu.be/SxGYPqCgJWM?si=CKmBi34_mv0oayTZ
- https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence
- [Video: KL Divergence - CLEARLY Explained! ~ Kapil Sachdeva](https://youtu.be/9_eZHt2qJs4?si=SveGupYXpPJ5VOHe)




$D_{KL}(P||Q) = \sum_{x\exists{X}}{P(x)log(\dfrac{P(x)}{Q(x)})}$ 
This measures how much $Q$ diverges from $P$, or how much "information" is lost when using $Q$ to approximate $P$.


Captures distances between probability distributions. It is a non-symmetric measure.

Minimizing the Cross Entropy Loss is equivalent to minimizing the KL loss. Since the CE Loss has a simpler form, it's become (one of) the standard loss functions.

Variants:
- ==Forward KL Divergence== (from p, the reference distribution, to q, the approximating distribution. This is the one we use in ML; it's mean-seeking)
- ==Reverse KL Divergence== (from q, the approximating distribution, to p, the reference distribution. Mode-seeking behavior.)

 
- ---
Relationship between KL Divergence and [[Cross-Entropy]]
- Both metrics aim to measure how one distribution differs from a second, reference probability distribution. When applied to the problem of classification.
- When applied to the problem of classification, where we're comparing a predicted distribution $Q$ against a true distribution $P$ , the KL divergence becomes part of the cross-entropy formula!

Mathematically, we usually say:

$H(P, Q) = H(P) + D_{KL}(P||Q)$ 
CrossEntropy(P,Q) = Entropy(P) + KLDivergence(P, Q)

Given two distributions, this makes sense that ==minimizing the CE and minimizing the KLD both amount to doing the same thing.==

Q: So why do we choose to use Cross-Entropy instead of KL Divergence in most classification task losses?
- Primarily due to the fact that Cross Entropy is formulated in a way that is computationally convenient and interpretable in the context of classification problems -- it directly ties to maximizing the likelihood of the observed data under the model, which is a fundamental principle in statistical modeling.


![[Pasted image 20240425160031.png]]
![[Pasted image 20240425160043.png]]
----

# Intuitions
#### 1. ==Expected Surprise==

$D_{KL}(P||Q)$  = how much ***==more surprised*==** you would expect to be when observing data with distribution $P$ , if you ***==falsely==*** believed the distribution is $Q$ (vs if you knew the true distribution)

#### 2. Hypothesis Testing

$D_{KL}(P||Q)$  = the amount of **==evidence==*** that we expect to get for $P$ over $Q$ in hypothesis testing, if $P$ were true.

#### 3. MLEs

If $P$ is an empirical distribution of data, $D_{KL}(P||Q)$ is minimized (over $Q$) when $Q$ is the ***maximum likelihood estimator*** for $P$.

#### 4. Suboptimal Coding

$D_{KL}(P||Q)$  = the number of bits we're wasting if we try and compress a data source with distribution $P$ using a code which is actually optimized for $Q$ (i.e. a code which would have minimum expected message if $Q$ were the *true* data source distribution)

((This is very similar to #1, to me -- which is fine!))

#### 5A. Gambling games -- Beating the house

$D_{KL}(P||Q)$  = the ==amount== (in log-space) we ==can win from a casino game, if we know the true distribution== is $P$ ==but the *house* **incorrectly believes*** it== to be $Q$.

#### 5b. Gambling games -- gaming the lottery

$D_{KL}(P||Q)$  = the amount (in log-space) we can win from a lottery if we know the winning ticket probabilities $P$ and the distribution of ticket purchases $Q$

((This of course seems to be the same thing as 5A))

#### 6. Bregman Divergence

$D_KL(P||Q)$ is in some sense ==a natural way of measuring how far $Q$ is from $P$,== if we're using the entropy of a distribution to capture how far away it is from zero (analogous to how $||x - y||_2$ is a natural measure of the distance between vectors $x$ and $y$, if we're using $||x||_2$ to capture how far the vector $||x||_2$ is from zero)


==The common theme for most of these:==

$D_KL(P||Q)$  -> [[Kullback-Leibler Divergence]] is a measure of how much our model $Q$ differs from the true distribution $P$. In other words, we care about how much $P$ and $Q$ differ from eachother ***in the world where P is true***, which explains why KL-divergence is not symmetric!


![[Pasted image 20240411185747.png]]
((I'm not sure if this is *really* a good example, because KL divergence is asymmetric, which isn't what I'd intuitively do.))
Other examples for intuition: https://news.ycombinator.com/item?id=37214898


---

Recapping, these, we find that if $D_{KL}(P||Q)$ is large, this indicates that:
1. Your model $Q$ will be very surprised by reality $P$
2. You expect to get a lot of evidence in favour of hypothesis $P$ over $Q$, if $P$ is true
3. $Q$ is a poor model for observed data $P$ 
4. You would be wasting a lot of message content if you tried to encode $P$ optimally, while falsely thinking that the distribution were $Q$ 
5. You can make a lot of money in betting games where other people have false beliefs $Q$, but you know the true probabilities $P$

These all have in common:
> $D_{KL}(P||Q)$ is a measure of how much our model $Q$ differs from the true distribution $P$. In other words, we care about how much $P$ and $Q$ differ from eachother *in the world where $P$ is true*, which explains why KL-divergence is not symmetric.

To put this last point another way, $D_{KL}(P||Q)$  "doesn't care" when $q_x >> p_x$ (assuming both probabilities are small), because even when our model is wrong, reality doesn't frequently show us situations in which our model fails to match reality. But if $p_x >> q_x$, then the outcome $x$ will occur more frequently than we'd expect, consistently surprising our model and thereby demonstrating the model's inadequacy.
