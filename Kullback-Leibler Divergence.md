---
aliases:
  - KL-Divergence
  - Relative Entropy
  - Forward KL Divergence
---
References:
- [Intuitively Understanding the KL Divergence](https://youtu.be/SxGYPqCgJWM?si=CKmBi34_mv0oayTZ)
- [LessWrong: Six and a Half Intuitions for KL Divergence](https://www.lesswrong.com/posts/no5jDTut5Byjqb4j5/six-and-a-half-intuitions-for-kl-divergence)
- [Video: KL Divergence - CLEARLY Explained! ~ Kapil Sachdeva](https://youtu.be/9_eZHt2qJs4?si=SveGupYXpPJ5VOHe)
- Video: [Luis Serrano KL Divergence](https://www.youtube.com/watch?v=sjgZxuCm_8Q&list=WL&index=24&t=9s)
	- Actually explains the idea of why we use "negative lob probabilities" very intuitively!
- [Artem Kirsanov: They Key Equation Behind Probability Video](https://youtu.be/KHVR587oW8I?si=kPgUbkjrIk8cQ9ze)
- [RitVik Math: The KL Divergence](https://youtu.be/q0AkK8aYbLY?si=8tQ4N_k7NhMlFKHV)

![[Pasted image 20240703135248.png|300]]
(Above: Definition)

This measures how much $Q$ diverges from $P$, or how much "information" is lost when using $Q$ to approximate $P$. KL Divergence is in the \[0, $\infty$\], and 0 only if p=q.
Captures distances between probability distributions. It is a ==non-symmetric measure==.
Minimizing the Cross Entropy Loss is equivalent to minimizing the KL loss. Since the CE Loss has a simpler form, it's become (one of) the standard loss functions.

![[Pasted image 20240703140046.png]]
(Above: Definition; example in the case of probability density estimation, with our approximating distribution parametrized by some $\theta$)

Variants:
- ==Forward KL Divergence== (from p, the reference distribution, to q, the approximating distribution. This is the one we use in ML; it's mean-seeking)
- ==Reverse KL Divergence== (from q, the approximating distribution, to p, the reference distribution. Mode-seeking behavior.)

---

Say we have a five-sided die with the following probabilities over rolls.
![[Pasted image 20240703124320.png]]
![[Pasted image 20240703124425.png|300]]
Say we wanted to replicate some sequence of rolls; Intuitively, the second die seems most similar to the first die. But how can we actually quantify that? What's the probability of Die2 or Die3 of generating this sequence?
- We can take the probability of generating a single outcome, and consider that all the rolls in the sequence are independent... so we just get the product of all of the probabilities of each roll in the sequence. 
![[Pasted image 20240703131212.png]]
- - This results in a very small number, because we're multiplying many < 1 numbers together -- we might even underflow! 
- We can turn this product into a sum by taking the log of each probability and adding them! This is because of the identity $log(ab) = log(a) + log(b)$. Now we have a sum of log probabilities -- but the log of a small number between 0...1 is a negative number, so our sum is going to be negative. Because we'd prefer positives, let's just multiply the entire thing by -1 to get a positive number. Let's also divide by the number of rolls to get an average.
![[Pasted image 20240703131027.png]]
- In the bottom right, we're comparing Die1 with itself, and we get the [[Cross-Entropy]] H(P|P); in this case, we're comparing it with itself, so it's just the [[Entropy]] H(P). In general, the more spread-out the distribution is, the higher the entropy is.
Let's now do the same thing, but compare Die1 and Die2
![[Pasted image 20240703131238.png]]
See that we're considering the Cross Entropy from Die1 to Die2, in this situation. See that $-(1/n)log(probability)$ is equivalent to $-\sum_i{p_ilog(q_i)}$ -- think about why.
And let's do the same thing for Die1 and Die3 (where Die3 was more visually dissimilar to Die1 than Die2 was)
![[Pasted image 20240703131357.png]]
When we compare these figures
![[Pasted image 20240703131441.png]]
The [[Kullback-Leibler Divergence|KL-Divergence]] is between P and Q is simply the Cross Entropy H(P|Q).
- As you can see, the smallest KL divergence is between the distribution and itself, and the 
(Correction to table: For Die 1: 0.4^4 * 0.2^2 * 0.1^1 * 0.1^1 * 0.2^2 For Die 2: 0.4^4 * 0.1^2 * 0.2^1 * 0.2^1 * 0.1^2 For Die 3: 0.1^4 * 0.2^2 * 0.4^1 * 0.2^1 * 0.1^2)

This can also be extended from the discrete case to the continuous case:
![[Pasted image 20240703133113.png]]


- ---
Relationship between KL Divergence and [[Cross-Entropy]]
- Both metrics aim to measure how one distribution differs from a second, reference probability distribution. When applied to the problem of classification.
- When applied to the problem of classification, where we're comparing a predicted distribution $Q$ against a true distribution $P$ , the KL divergence becomes part of the cross-entropy formula!

Mathematically, we usually say:

$H(P, Q) = H(P) + D_{KL}(P||Q)$ 
*CrossEntropy(P,Q) = Entropy(P) + KLDivergence(P, Q)*

or equivalently
$D_{KL}(P||Q) = H(P,Q) -  H(P)$ 
*KLDivergence(P,Q) = CrossEntropy(P,Q)-Entropy(P)*

![[Pasted image 20240703135423.png|300]]
(Above: Consider how this formula falls out of the alternative phrasing of Cross Entropy minus Entropy, where cross entropy is $H(P,Q) = -\sum{P(x)log(Q(x))}$  and entropy is $H(P) = -\sum_{P(x)log(P(x))}$ 
- Consider the identity $log(a) - log(b)  = log(a/b)$ ... and we see how we get the log odds ratio.

Therefore, relative entropy (KL Divergence) can be interpreted as ==the expected extra message-length per datum that must be communicated if a code that is optimal for a given (wrong) distribution Q is used, compared to using a code based on the true distribution P: it is the _excess_ entropy.==

Given two distributions, this makes sense that ==minimizing the Cross Entropy and minimizing the KL Divergence both amount to doing the same thing.==

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


![[Pasted image 20250112235914.png]]
