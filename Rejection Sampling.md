Popularized with [[LLaMA 2]]
The training-time version of [[Best-of-N Sampling]]

A sampling method for LLMs in which we sample a batch of K completions from a language model policy and then evaluate them across a reward model, returning the best one. If you retrain on the best few outputs via the reward model, your policy (LM) can improve.

Todo: See how it's used both in [[Speculative Decoding]] and in [[Medusa]]; we have some speculative artifact (a draft model, in SD, and multiple Medusa heads, in Medusa) that's generating a speculative forward-looking sequence of tokens that haven't yet been confirmed by the "main" model... We pass this speculative sequence through the main model to compare our per-token probabilities from our speculative process to the per-token probabilities as assessed by our main model. We accept tokens up until we reject one of the candidate tokens, which happens when both (in sequence) the $p_{model}(token) < p_{speculation}(token)$ *and* we then fail to otherwise accept it with $p_{accept} = p_{model}/p_{speculation}$ .

----
# Rejection Sampling in Probability
(NOTE: There seems to be a concept of [Rejection Sampling/Accept-Reject Sampling](https://en.wikipedia.org/wiki/Rejection_sampling) in probability, and this seems *not* to be the same thing as rejection sampling )

In *that* domain, it's seen as a basic technique to generate observations from a probability distribution (density; though it seems people do it with discrete anways). 
It's based on the observation that to sample a random variable in one dimension, we can perform some uniformly-random sampling of a two-dimensional Cartesian graph (x,y), and keep the samples in the region under the graph of its density function. This can be extended to N-dimensional functions.

To visualize the motivation behind rejection sampling, imagine graphing the probability density function (PDF) of a random variable onto a large rectangular board and throwing darts at it. Assume that the darts are uniformly distributed around the board. Now remove all of the darts that are outside the area under the curve. The remaining darts will be distributed uniformly within the area under the curve, and the 

![[Pasted image 20240708155545.png]]
An example of someone doing it in the discrete case.
![[Pasted image 20240708155615.png]]
Imagine just keeping the blue darts, and drawing a line down to the x axis for each dart. Note that x values near the mode will have the highest probability of being sampled.