References:
- [Video: Mutual Information's Importance Sampling Introduction](https://youtu.be/C3p2wI4RAi8?si=HxU_JpVMBycDI_vP)
- Video: [Mutual Information's Monte Carlo and Off-Policy Methods (24:00)](https://www.youtube.com/watch?v=bpUszPiWM7o)


It's a [[Monte-Carlo]] method for evaluating properties of a particular distribution, while only having samples generated from a different distribution than a distribution of interest (intriguing!). Useful in the context of [[Off-Policy]] learning, where we have a *behavior policy* that's exploring the MDP, but we have a *target* policy that's trying to learn the optimal policy. Also used in [[Variational Autoencoder]]s (VAEs).

When we want to calculate the expectation of a random variable $E_{x \sim p}[f(x)]$ , we can write it as $\int p(x)f(x)dx$, the probability-weighted sum of functions. This can also be written for discrete as $\sum{p(x)f(x)}$.

A lot of the time, the integral is impossible to calculate exactly. Typically because the dimensionality of x is high, so the space it lives in is exponentially huge, so we have no hope of adding up everything within it. 

This is where Monte-Carlo methods come in -- the idea is to merely approximate the expectation with an average:

$\mathbb{E_p}[f(x)] \approx \frac{1}{N}\sum_{i=1}^Nf(x_i)$ where $x_i \sim p(x)$ 

This says that we collect $N$ samples of $x$ from the distribution $p(x)$, plug those into $f$, and take their average. It turns out that as N gets large, this approaches our answer.
![[Pasted image 20250113145641.png]]
The area under this p(x)f(x) function is the truth that we're after (Area = 0.5406); but in the general case, it's not possible to approximate the area under this curve.

So instead, we sample $x$s from $p(x)$, and then plug them into $f(x)$. The average of many of these is an approximation of area under this $p(x)f(x)$ curve! 

Our sample average itself has its own distribution.
![[Pasted image 20250113145928.png|300]]
This MC estimate is ==unbiased==, because it's centered on the true expectation that we're after! Looking at this, we can see that the variance of our rollout return $s$, $V_p[s]$, matters a lot! It turns out that that the variance of $s$ is the variance of $f(x)$ scaled down by N!

$\mathbb{V_p}[s] = \frac{1}{N}\mathbb{V_p}[f(x)]$ 

Where did that come from? That's the [[Central Limit Theorem]], and it says that it doesn't matter what $p$ or $f$ are -- no matter what, as N gets large, it becomes closer and closer to a normal distribution!

We can write as:
![[Pasted image 20250113150529.png|350]]
This says that $s$'s distribution approaches the Normal as N gets large. 
- The mean of this normal is the expectation of the function that we're trying to calculate
- Its variance is the variance of f(x) scaled down by N.

Now we can learn about importance sampling!
- We introduce a new distribution q(x). 
- We're still interested in understanding $E_{x \sim p}[f(x)] = \int p(x)f(x)dx$ 

We take the right term and multiply it by $\frac{q(x)}{q(x)}$, which is just 1.

This gives us $= \int  q(x) \frac{p(x)}{q(x)}f(x)dx$ 
This is the probability-weighted average of a new function!

We can then rewrite this $\int q(x)$ as $\mathbb{E_{x \sim q}}$ so that the full thing can be written as 

$E_{x \sim p}[f(x)] = E_{x \sim q}[\frac{p(x)}{q(x)}f(x)]dx$ 

==So the p-probability-weighted average of f(x) is equal to the q-probability-weighted average of f(x) times the ratio of p and q densities!==

Recalling Monte Carlo, now, we can estimate this USING SAMPLES FROM q!

$\mathbb{E_q}[\frac{p(x)}{q(x)}f(x)] \approx \frac{1}{N}\sum_{i=1}^N{\frac{p(x_i)}{q(x_i)}f(x)}$ , where $x_i \sim q(x)$

In other words, we can approximate this expectation by using the Monte Carlo method, taking a bunch of samples of $x$ from the $q(x)$ distribution and seeing what the average return is, where the return is this $f(x)$ for the drawn $x$, but weighted by this ratio of $p(x)q(x)$.

So what's the advantage of using this $\frac{1}{N}\sum_{i=1}^N{\frac{p(x_i)}{q(x_i)}f(x)}$  , which we'll refer to as $r$ ?
This is still unbiased, and has a new, possibly-improved variance:

The expected value of $r$ under $q$ is 
$\mathbb{E_q}[r] = \mathbb{E_p}[f(x)]$

New, possibly improved variance:
$\mathbb{V_q[r]} = \frac{1}{N}\mathbb{V_q}[\frac{p(x)}{q(x)}f(x)]$

So the hope is that we can choose q such that this variance is less than the variance we dealt with earlier:
![[Pasted image 20250113160039.png|400]]
This is a key result that we aren't proving, but it makes sense when you think that we're trying to estimate the area under the p(x)f(x) curve!

How we chooose q(x) depends on p(x) and f(x). It's tricky, and the answers aren't very satisfying:
- Maybe we can choose a q that approximates p?
- It's very each to do a terrible job selecting q, especially in high dimensions. In these cases, the density ratio (p(x)/q(x)) will vary wildly over samples... and a majority of them will be very small. This means your average is effectively determined by a small number of samples, making it high variance -- not good!

Wrap up:
- When is Importance Sampling likely to be useful?
	- When p(x) is difficult or impossible to sample from
	- When we can *evaluate* p(x), meaning we can plug x in and get a probability.
	- q(x) is easy to evaluate and sample from
	- We can choose q(x) to be high were the |p(x)f(x)| is high
		- This is not necessarily an easy task.

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


