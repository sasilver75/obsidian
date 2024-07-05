https://www.youtube.com/watch?v=8cO61e_8oPY
-----

Variational Autoencoder:
![[Pasted image 20240705153047.png]]
- We first sample a simple latent variable z (eg by drawing from a simple gaussian distribution)
- We pass this sample z into our two neural networks that give us the parameters of another gaussian distribution (a mean vector and a covariance matrix).
- We actually generate our datapoint by sampling form this resulting conditional distribution, p(x) given z.

The building blocks are simple, but the marginal distribution over x that you get is very flexible/general.
- Can think of it as a mixture of a mixture of a very large/infinite number of gausians.
	- For every z, a gaussian
	- An infinite number of zs, so an infinite number of gaussians.

If we wanted to know p(x), we'd have to integrate over all possible values of z, seeing the p(x|z) of each.

Recap:
![[Pasted image 20240705153450.png]]

But there's no free lunch
- The price you pay is that these models are more difficult to learn compared to fully-observed autoregressive models, because p(x) is hard to evaluate (and optimize), because you have to essentially check all values of z that have generated a datapoint x
	- This means you can't *really* evaluate likelihood of datapoints... this makes training pretty hard; because how do we optimize the probability of producing the dataset?
	- This is different from autoregressive models where it was trivial to evaluate likelihood p(x) because we could just multiply a bunch of conditional probabilities together p(x1)px(x2|x1)p(x3|x2x1) to get the joint probability.


So how do we train these latent variable models?
- It relies on [[Variational Bayesian Inference|Variational Inference]], where we have an auxiliary model that we'll try to use to infer latent variables.
	- This auxiliary model is also going to be a NN in this course
- We're going to jointly train the generative model and an auxiliary inference model that we'll use to reduce the problem to one that we've seen before, where both the x and z part is observed.

![[Pasted image 20240705155251.png]]
This is built on the result of an evidence lower bound [[Evidence Lower Bound|ELBO]]
- We've seen that we can obtain a lower bound through [[Jensen's Inequality]] on a quantity that we want to optimize (logp(x;theta)) using an auxiliary *proposal distribution q* to infer the values of latent variables.
	- Remember the challenge is we only get x, but don't get to see z, so we have to infer it. We use a ==proposal distribution== q to infer the values of the z variables, when only x is observed. It construct a lower bound on the marginal likelihood, which is the quantity we want to optimize as a function of theta.
We can decompose that objective into two pieces
- Average log probability when when both the x part are observed, when you infer the z part using this q model. (As if both x and z were fully observed, but we infer the latent part using the q model)
- Another piece that doesn't depend on p at all; it only depends on q. It's the expected value under q of log q, which is what we call the [[Entropy]] of q. It's essentially a quantity that tells us how "random" q is; how uncertain we should be about a drawn sample from q.
The higher the sum of these two terms, the closer we get to the evidence (the true value of the marginal likelihood).
We also briefly discussed that if we were to choose q to be the posterior distribution of z|x under your generative model, then the above inequality becomes an equality, and there's no longer an approximation involved, and the ELBO becomes *exactly equal* to the marginal likelihood. Cool! 
- This has the flavor of an EM algorithm where we have a way of filling in missing values using the q distribution, and we pretend that all of the data is fully-observed (and then there's the entropy term).


So how do we do this...
![[Pasted image 20240705155716.png]]
If we work out the [[Kullback-Leibler Divergence|KL-Divergence]] between the q distribution and the optimal wa of inferring the latent variables (the conditional distribution of z, given x):
- We see that this expression (with algebra) is equal to the thing on the right 
	- Average log joint probability when we use q(z) to infer the value of the z variables.
	- Marginal likelihood of data
	- Entropy of q
- The key takeaway is that KL divergence is known to be non-negative for any choice of q.
So if we re-arrange these terms, we re-derive the ELBO in a slightly different way!
Again, this bound holds for *any choice of q*
But we pick q = p(z|x ;theta), then the inequality becomes an equality (and the KL divergence becomes 0, since both sides are the same), and we can rearrange terms to get:
![[Pasted image 20240705155841.png]]
In practice.... the p(z|x;theta) object is too expensive to compute. If you could compute the posterior, then you could also just compute logp(x;theta)! 
So in [[Variational Bayesian Inference|Variational Inference]], we try to optimize over q(z) to try to find the tightest possible lower bound. ==We have a separate NN playing the role of q, and we'll try to jointly optimize both p and q to try to optimize our [[Evidence Lower Bound|ELBO]].==
- ==Decoder in VAE (p)==
- ==Encoder in VAE (q)==
Will both work together to maximize the ELBO as much as possible!
(The optimal q would be the true p(z|x;theta))


So... the ==problem as we've brought up is that these posterior distributions p(z|x; theta) is intractable to compute!==
![[Pasted image 20240705160543.png]]
- In a VAE, this corresponds to "inverting" the decoder
	- The conditional distribution of x|z is relatively simple (a gaussian), but the gaussian depends on two NNs (mu, sigma).
	- So when we try to compute z|x, we're basically trying to invert these two neural networks. ("Given an x, what zs were used to produce it?")

![[Pasted image 20240705160952.png]]
So the idea is that we're going to ==approximate== this intractable posterior p(z|x;theta).
- We're going to define a family of distributions q over the latent variables, which are also going to be parametrized by some set of parameters phi (variational parameters).
	- Maybe one part of phi denotes the mean, and the other denotes the variance, and we want to set these parameters to get as close as possible to the true posterior we want to get q(z; phi)
- So we'll then jointly try to optimize q and p to maximize the ELBO.
So in variational inference, we want to optimize our variational parameters phi so that q(z; phi) is as close as possible to p(z|x; theta), which is a thing we don't know how to compute.
- So we have a true conditional distribution p(z|x) which we show as a mixture of two gaussians in blue above; Let's say we want to approximate this distribution using a Gaussian... we might have something like the orange distribution (mean 2, variance 2 for phi1 phi2). So in a VAE we try to optimize these parameters phi to try to match the true posterior distribution p(z|x) as well as we can.

![[Pasted image 20240705161642.png]]
We want to jointly optimize the right hand side of the inequality to make the ELBO as low as possible to the thing we care about (log p(x)); by optimize our probabilities, we push up the lower bound on the marginal likelihood of x, which is what we want!
- The decoder is basically theta, and the encoder is basically phi, and these two things will work together to try to optimize the evidence lower bounder (ELBO).
The gap between the ELBO and the TRUE marginal likelihood is given by the KL divergence in the second equation; the better we can estimate p(z|x; theta), the smaller the divergence between the proposal distribution (q) and the true optimal one (p(z|x)) becomes.... How big the KL divergence is determines how much slack there is between the lower bound and the blue line.

Q: We're trying to compute log p so we can do [[Maximum Likelihood|Maximum Likelihood Estimation]], right? So there are two objectives here, we're trying to optimize logp(x;theta) as well as the ELBO?
A: The dream would be that we could just optimize logp(x;theta); if we *could* do that directly, that would be awesome. But we don't really know how to evaluate that quantity. But we can get a LOWER BOUND with the ELBO by optimizing phi and theta. So instead of optimizing the marginal probability logp(x;theta), we optimize the lower bound.


![[Pasted image 20240705162929.png]]
So we know for any choice of q, we can compute the associated ELBO
And that we'd like to optimize the average log probability logp(x) for every x in our dataset.
- So what we can do... given that we know how to get a lower bound for any log probability for any x, through the ELBO... we can then get a lower bound on the log likelihood assigned to our *dataset* by taking the sum of ELBOs for each datapoint.

![[Pasted image 20240705163205.png]]
The main complication is that we're going to need different qs for different datapoints!
- If you think about it -- the posterior distribution, even for some same choice of theta, is going to be different across datapoints.... so we might want to choose different variational parameters for different datapoints. 
- So while we have a single choice of theta for our generative model, we might have a different variational parameters phi for different datapoints.
	- We'll see that this isn't scalable, so we'll have to introduce additional approximations to make things more scalable.


![[Pasted image 20240705164545.png]]

So how do we actually do this?
![[Pasted image 20240705164745.png]]
The simplest version would be to do gradient descent on the ELBO function (expanded)
- For every datapoint x_i we have a expectation with respect to the variational distribution q... and have the log probability in the fully-observed  case, and then this term which is kind of like the entropy of q (expected log probability under q).
- We initialize our optimization variables somehow 
- We randomly sample a datapoint
- Try to optimize our loss as a function of the variational parameters.
	- Inner loop finds the best lower bound
- We take a step on our theta after computing the gradient
	- Takes a step on that optimal lower bound

Q: It seems like each step is going to take a lot of computation.
A: We need to see how to even compute these phi gradients in this inner loop. Turns out it's not so expensive, but there's definitely a question of how many steps we should take on theta before taking a step on phi.

![[Pasted image 20240705165625.png]]
![[Pasted image 20240705165643.png]]

![[Pasted image 20240705165705.png]]

![[Pasted image 20240705165732.png]]

![[Pasted image 20240705165803.png]]

![[Pasted image 20240705165813.png]]






