https://www.youtube.com/watch?v=MAGBUh77bNg

The plan for today is to talk about latent variable models! 
- We previously saw the first family of generative models, ==Autoregressive models==
	- Using chain rule based factorization to describe our joint distribution as a product of conditionals... which we tried to approximate using some kind of NN.
	- Pros: Easy to evaluate likelihoods (you get access to the likelihood), and easy to train as a result (optimize the parameters of your model to maximize the probability of the dataset you're given).
	- Cons: Requires some ordering of conditionals (sometimes easy, sometimes tricky), generation is slow and sequential, cannot learn features in an unsupervised way.

----


The motivation for latent models
- When you're trying to model a complicated dataset (eg images of people), the problem is typically hard because there's a lot of variability that you have to capture; People might have different ages, poses, hair colors, eye colors, etc. These all lead to very different values for pixels in our dataset.

But this latent structure isn't annotated -- it's not obvious how to discover it and take advantage of it!

Idea: ==Latent Variable Models== essentially add a bunch of random variables $z$ that are supposed to capture all of these latent factor of variation. These "hidden" or "latent" random variables are not directly observed in the dataset.
- Advantages: Extracting these latent variables given pixel values... yield useful features for doing other things (eg training a classification model on these small number of latent variables, as opposed to the raw high-dimensional data/pixel values.)


![[Pasted image 20240703155924.png]]
We want a joint probability distribution between our observed $x$ and our latent variables $z$ (which we don't get to observe for each datapoint; we have no annotations).
- These latent variables correspond to high-level features
	- ==If z is chosen properly, p(x|z) could be much easier than the simple marginal distribution p(x)!==
	- For some new datapoint x, we can identify/infer these features as $p(z|x)$!
		- eg p(EyeColor = Blue | x)

It might be hard to specify these conditionals by hand (as above, as something like a bayesian network) -- so as usual, we'll instead use a neural network to model the conditionals.

![[Pasted image 20240703161311.png]]
We have a set of observed variables $x$ and a latent variable $z$ , but we won't have anything interpretable in terms of how the random variables are related to eachother, or even what they mean.
- We'll assume that we have a set of random variables $z$ that are somewhat simple (eg distributed via a gaussian distribution)
- We model the conditional distribution of x given z using some kind of NN... We have a simple distribution like a gaussian, but the parameters of the distribution depend in some way on latent variables through a couple neural networks.
	- The parameters of our neural network intuitively give us an estimate of x as a function of z... and we don't know what z was, so intuitively we sort of have to guess a value for the latent variables, and then try to fit the model.

We'd never expect that a simple gaussian is enough to measure something interesting... but if we were able to cluster our datapoints in an interesting way, then within the cluster, we might be able to get away with a simple distribution (?).


The goal is always to just model p(x)
- The motivation for using this z variable is that it might make your life easier! If you're somehow able to cluster the data using these z variables, then your life might be useful.
- Inferring the latent variables might be useful if you're not (eg) trying to generate images, but instead... understanding what sort of latent factors of variation exist in your dataset.



As a warmup, let's look at the simplest kind of latent variable model you can think of, which is a mixture of gaussians... no NN involved.
[[Gaussian Mixture Model]]
![[Pasted image 20240703163227.png]]
- z is a categorical random variable.. say there is k mixtures here.
- p(x | z = j) is a gaussian, and we have some table that tells us what the mean and variance is for some mixture k. This defines a generative model.
So we sample by sampling a mixture component z (eg by uniform random), then sampling an x from a gaussian with a corresponding mean and covariance.
(Picture assumes x is two-dimensional)


![[Pasted image 20240703164646.png|400]]
We can see that there's some kind of relationship between these two variables.
- We could try to model it using a single gaussian... but you can see that it's not going to do a good job, because we'll put a lot of probability mass in the middle where there's not a lot of data.
- It really seems like there's two types of eruption -- if we use two gaussians, it looks much better. If we can infer the z variable, given the x (figuring out which cluster it belongs to), then we can give a much better prediction of the distribution.

![[Pasted image 20240703164955.png]]
We can see that with unsupervised learning, we're able to do some reasonable clustering.

Q: How do we actually learn these posteriors?
A: We haven't talked about it! We'll go through it in this class!


![[Pasted image 20240703165516.png]]
Using latent variables allows us to have this powerful mixture model behavior... you can kind of see in our mixture of gaussian example that if we have three gaussians (blue, orange, green) that if we think about the corresponding marginal distribution over x, it has this nice red shape.
- We can see here that we have a discrete number of latents, and we're basically summing these distributions.

Even though p(x|z) (these gaussians) are really simple, the combined marginal that you can get is pretty powerful. This is one of the reasons why VAEs are so powerful!
- ==But now $z$ isn't just a categorical random variable (we don't have a finite number of mixture components)... now $z$ can take on an infinite number of different values (think of it as an infinite number of mixture components; so now we have something like a mixture of an infinite number of gaussians).== 

![[Pasted image 20240703165709.png]]
z is basically continuous now (the prior we're setting is that it's going to be a unit normal distribution; it can be something else, though), instead of only taking k different values as it did in the previous example.
- The process to sample would be the same as before (in the GMM, we sample a value of z (now it's a gaussian), do our "lookups" to get the mean and variance (now, the mean and standard deviation don't come from from some lookup table as before, but instead from two neural networks), and sample from the resulting gaussian that's parametrized by our predicted mean and variance.
The key here is about the neural networks that learn to map the latent variables to the parameters of this simple distribution that we sample our p(x|z) from.
The last point speaks to the previous picture where we were basically "combining/mixing" gaussians into a more complicated marginal distribution.

Q: What's the dimension of z in practice, relative to x?
A: Typically z is a smaller dimensionality than x; the idea is to discover some number of latent factors of variation that well-describe your data.

Q: Is it possible to put more information into the prior, instead of sampling from a gaussian?
A: Yeah, you could do something like put an autoregressive model over the latent variables to get an even more flexible distribution. 

![[Pasted image 20240703172801.png]]

The problem is that basically we have some missing values. What happens is that we have something like this:
![[Pasted image 20240703172958.png]]

![[Pasted image 20240703173907.png]]
So to predict the probability of a datapoint, we need to integrate over the probability of observing that datapoint over all z's, parametrized by parameters theta.

We have a dataset, but for every datapoint, we only get to see the x variables, and the z variables are missing; they're unobserved.
- Maximum log likelihood; But we still want to maximize the probability of our data by varying our parameters!
![[Pasted image 20240703174357.png]]
The problem is that evaluating the probability of our datapoint under mixture models is expensive, because we have to sum over all possible values that the z variable can take, for that  datapoint. This can even be intractable even for reasonably small $z$!

So how can we approximate?
We can try a [[Monte-Carlo]] thing; Instead of summing over *all possible values of z*, can we try just summing over *some values of z* to get an approximation?
![[Pasted image 20240703180035.png]]
Approximating a sum with a sample average
- We randomly sample a bunch of values of z
- We approximate the expectation using a sample average
	- (Bottom): We check how likely these completions are under the joint, and rescale appropriately.
So we just need to check k completions, instead of all the possible completions we'd have to deal with.
- UNFORTUNATELY because the k space is so large (and because probability is pretty sparsely distributed amongst it), Monte Carlo sampling with reasonably-computed k doesn't really work well.

==We need a better way of selecting z's!==

![[Pasted image 20240703182711.png]]
Let's try to use something like [[Importance Sampling]], where, instead of sampling uniformly at random, we try to sample important completions more often.
- Note: we want $p_{\theta}(x)$ , which is also the sum of $p_{\theta}(x,z)$ over all possible latent variables. ==Now, we can multiply and divide by this q(z), where q is an arbitrary distribution that we can use to choose values for the latent variables.== (this is just a "1" term)
- But now again, we have an expectation with respect to q(z)... of this ratio of probabilities. So with the definition of expectation, we make that last transformation. The probability of the true model compared to the probability under our proposal distribution.
	- This is the expectation of p₀(x,z) / q(z) where z is sampled from the distribution q(z).
	- ![[Pasted image 20240703181933.png|400]]
	- This is useful because it allows us to estimate $p_\theta$(x) by sampling from this $q(z)$  distribution, rather than summing over all z! We can choose q(z) to be some distribution that's easy to sample from; if q(z) is chosen well, it can reduce the variance of our estimate compared to naive Monte-Carlo sampling.

Now that we have an expectation, we can try to do the usual trick of sampling a bunch of z's (now, not uniformly, but from our proposal distribution q), and approximate our expectation using a sample average:
![[Pasted image 20240703183034.png]]
In terms of choosing our proposal distribution q(z)... we want it to frequently sample z's that are likely, given our X.

Now... the issue is that we don't want to determine the p(x), but the log probability of a datapoint log(p(x))...
- So we need to apply a log to the expression from earlier.
![[Pasted image 20240703183232.png]]
If we were to choose a single sample (if k=1, just sampling a single possible completion and evaluating the estimator that way)... it's just the ratio of the two probabilities (?)
- You can see that it's no longer unbiased.
- The expectation of the log is no longer the log of the expectation!

We can actually figure out what that bias is.
Recall: We want the log marginal probability, which we can write down as this importance sampling distribution:
![[Pasted image 20240703183328.png]]
We know that log is a concave function.
We can use this concavity property using the [[Jensen Inequality]]
- The log of the expectation of a function is at least as large as the expectation of the log.
Because of this concavity, we can basically work out what happens if we swap the ordering of logarithm and expectation (if we put the expectation *outside* the log, we get a bound on the quantity we want).
[[Evidence Lower Bound ]](ELBO)
![[Pasted image 20240703183509.png]]
((Teacher stops and considers that he's killed the class.))
![[Pasted image 20240703183744.png]]
Above: [[Evidence Lower Bound|ELBO]]: A lower bound on the probability of evidence (the probability of x)... x is the thing we get to see... the log probability of evidence s the thing we'd like to optimize, but it's tricky to evaluate that, so instead we have this evidence lower bound (ELBO) which is this thing we can actually compute and optimize.

Recall: We care about doing maximum likelihood... what we care about is
![[Pasted image 20240703183901.png]]
For all the datapoints, we want to evaluate the log probability of that datapoint.
The good news is that we can get a lower bound on that quantity through the machinery we describe on the ELBO slide above.
- The strategy is basically going to be: Optimize this lower bound.
- We'll see that the choice of q (how we decide to sample the latent variables) basically controls how tight this ELBO bound is. If q is well chosen then this becomes a very good approximation to the quantity we care about, which is the log marginal probability p(x).

![[Pasted image 20240703184707.png]]
ELBO: For any choice of q, we have a nice lower bound on the quantity we care about.
- If we expand this thing, we get a decomposition where we use the log of the ratio as the difference of the logs, and the entropy H(q) of q falls out!
- We can also rewrite this expression as the sum of two terms: the average log joint probability, under q ... and the entropy under q.
"Essentially, the best way of guessing the z variables is to actually use the posterior distribution according to the model" 🤤

((I'm pretty jealous of the students that are able to follow this material live in the course and still ask good questions!))















