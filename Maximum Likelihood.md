---
aliases:
  - MLE
  - Likelihood
  - Log Likelihood
---
Maximum Likelihood Estimation (MLE) is a statistical method used to estimate the parameters of a probability distribution or statistical model.

The idea is to ==find the parameter values of your model that make the observed data most probable.==

$\mathcal{L}(\theta|x)$ represents the likelihood of the parameters $\theta$ given the observed data $x$
- This notation is meant to emphasize that we're considering the likelihood as a function of the parameters $\theta$, with the data $x$ held fixed.
- (Contrast this with Bayesian Inference, in which we actually do work with $P(\theta|x)$; this is different from the likelihood $L(\theta|x)$ used in frequentist inference. To avoid this confusion, some texts use notation like $L(\theta; x)$ instead of $L(\theta|x)$ to avoid conveying the idea of a conditional probability.)

This value is ***proportional*** ($\propto$) to the probability $P(x|\theta)$. As a result, this is often what we solve for.
- For conditionally-independent and identically distributed (IID) observations $P(x|\theta)$ = $p(x_1|\theta) * p(x_2|\theta) * ... * p(x_n|\theta)$.
- In an autoregressive language model, it might look something like $P(x|\theta) = p(x_1|\theta)*p(x_2|\theta, x_1)*p(x_3|\theta, x_1, x_2) * ... * p(x_n|\theta, x_1,...,x_{n-1})$ 

Instead of taking the product of multiple probabilities like in these examples (risking underflow by multiplying small numbers together), it's common to instead work with ==Log Likelihood==, because it's easier to to manipulate mathematically.
$\ell(\theta|x) = log(\mathcal{L}(\theta|x)) = \sum_i{log(P(x_i|\theta))}$
Note our use of $\ell$ for log-likelihood, rather than $\mathcal{L}$ for likelihood.

In the *Maximization* stage of MLE, we want to find parameters values $\theta$ that maximize the log-likelihood. This is typically done by taking the gradient/derivative of the log-likelihood with respect to $\theta$ and solving as a gradient-based optimization problem.
	
---
### Aside: Probability vs Likelihood
- ==Probability==: Measures the chance of an event occurring, sum to one. Answers "*Given these parameters, what's the chance of this data?*"
	- "If a coin has 60% chance of heads, what's hte probability of getting 7 heads in 10 flips?"
- ==Likelihood==: Measures the plausibility of parameter values, given observed data, and are not required to sum to one. Answers "*Given this data, how plausible are these parameter values?*"
	- If we observe 7 heads in 10 flips, how likely is it that the coin has a 60% chance of heads?"
----

