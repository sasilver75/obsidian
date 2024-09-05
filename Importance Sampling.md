References:
- [Video: Mutual Information's Importance Sampling Introduction](https://youtu.be/C3p2wI4RAi8?si=HxU_JpVMBycDI_vP)
- Video: [Mutual Information's Monte Carlo and Off-Policy Methods (24:00)](https://www.youtube.com/watch?v=bpUszPiWM7o)


It's a [[Monte-Carlo]] method for evaluating properties of a particular distribution, while only having samples generated from a different distribution than a distribution of interest (intriguing!). Useful in the context of [[Off-Policy]] learning, where we have a *behavior policy* that's exploring the MDP, but we have a *target* policy that's of interest to us. Also used in [[Variational Autoencoder]]s (VAEs).

---
Let's assume that we want to calculate the expectation of some function $f(x)$, where x is subject to some distribution $x \sim p(x)$ .
We have the following estimation of $\mathbb{E}[f(x)]$ :
![[Pasted image 20240708161527.png]]
with the right side showing the Monte-Carlo expectation that approximates this expectation.

Monte-Carlo sampling involves repeatedly sampling $x$ from the distribution $p(x)$ and taking the average of all $f(x_i)$ to get an estimation of the expectation.

But what if $p(x)$ is very hard to sample from? Can we estimate the expectation based on some *other* distribution which is *known, and easily sampled from?* Yes!

We do a simple transformation of the formula:
![[Pasted image 20240708162057.png]]
We multiply by some $q(x)/q(x)$ (=1).
Just as the first integral is interpreted as "expectation of $f(x)$, where x is subject to the $p(x)$ distribution," we could interpret the latter one as "expectation of $f(x)(p(x)/q(x))$, where x is subject to the new $q(x)$ distribution...but we know this is equivalent to the former expectation, because all we did was multiply it by "one" (q(x)/q(x)).
- $q(x)$ is called the ==proposal distribution==.
- $p(x)/q(x)$ is called the ==sampling ratio== or ==sampling weight==, and acts as a correction weight to offset the probability of sampling from our proposal distribution.

The key ideas are:
1. We want to estimate $E[f(x)]$ under $p(x)$, but $p(x)$ is hard to sample from
2. We sample from $q(x)$ instead, because it's known any easy to sample for.
3. We correct for this difference by multiplying each sample by a sampling ratio $p(x)/q(x)$ that works as a correction factor. In a sense, we're estimating the expected value of a different function (f(x)(q(x)/p(x))) under a different distribution (q(x)), but we know that the result is equivalent to the expectation of f(x) under p(x).
This allows us to estimate expectations with respect to one distribution by sampling from another, which can be very useful in statistical applications.

In this way, we're able to estimate the expectation of {f(x), when x is subject to p(x)} distribution. The trick is that we equivalently estimate {f(x)(p(x)/q(x), when x is subject to q(x)}! So we we're able to calculate the expectation of our function in the context of a distribution.. by considering the function in the context of a different distribution! Feels impossible, but the magic works out! 😅

Another thing to talk about is the [[Variance]] of our estimation:
![[Pasted image 20240708165938.png]]
In this case, X is our $f(x)(p(x)/q(x))$. If our sampling ratio is large, this results in a large variance, which we hope to avoid -- our goal is to select a proper q(x) that results in even smaller variance!


---

![[Pasted image 20240626153912.png|300]]
Above: "b" is the behavior policy. 
The thing we're after is the expectation under b of the return... multiplied by the ratio of the probabilities of that return under the two policies. This ratio is called *rho*, and it turns out that it's equal to the product of the ratios of the two policies over the remainder of the trajectory.
We also require coverage, meaning if the target policy has a non-zero probability of taking some action in a specific state, then so must the behavior policy. $\pi(a|s) > 0 \implies b(a|s) >0$. This ensures that in the limit of infinite data, we won't have zero data in the places where the target ends up.


