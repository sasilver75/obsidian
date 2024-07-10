---
aliases:
  - VAE
---
References:
- Video: [DeepBean's Understanding Variational Autoencoders (VAEs)](https://youtu.be/HBYQvKlaE0A?si=PIMO3X3rEC-oqAa4)
- Video: [Umar Jamil's Variational Autoencoder - Model, ELBO, loss function and maths explained easily!](https://www.youtube.com/watch?v=iwEzwTTalbg)

Variants: VQ-VAE (Vector Quantized-Variational AutoEncoder, when you want the latent to be discrete)

---
Explanation from Deep Bean's "Understanding Variational Autoencoders"

A generative model that learns a distribution of data p(x) from the dataset it's trained on.
$p_{data}(x) \approx p_{\theta}(x)$ , where we tune the parameters $\theta$ of our model to maximize the likelihood of our model generating our data ([[Maximum Likelihood|MLE]]).

Say we want our model to learn the distribution of circles
![[Pasted image 20240703201842.png|300]]
- We assume that there are some underlying factors of variation (position, radius) that guide the generation of this thing we call "circle"
- We call these underlying factors ==latent variables==, because we never directly see them; we just see the circle itself, represented by some pixels in an image; but we get the sense that these latent variables capture the structure in the data, and the data representation we're given is actually just their *shadow.*
	- (To tell our friend how to draw one of these circles, we would tell them what the position and radius of the circle, rather than describe the value for each pixel. In a sense, these latents are a maximally-efficient compression of the meaning of the data)
- So every datapoint $x \in D$ has both a *data representation* and a *latent representation*.
- So how do we learn a ==latent representation== that that captures the factors of variation that are inherent in our data?
![[Pasted image 20240703203836.png|300]]
- We would like to:
	- Given an a datapoint $x$, *infer* its latent $z$ by learning $p_{\theta}(z|x)$
	- Given a latent $z$, *generate* the datapoint $x$ by learning $p_{\theta}(x|z)$
Essentially, we want to find the *joint distribution* $p(z,x)$, which, using the [[Chain Rule of Probability]], we can write as $p_{\theta}(z)p_{\theta(x|z)}$. We don't really have any prior knowledge of how p(z) should be distributed, so in practice we just make up some distribution (eg a multivariate unit gaussian) for our p(z).
![[Pasted image 20240703203806.png|300]]
All we have to do now is optimize theta; but to do this, we have to maximize our original target, p_theta(x)... which can only be found by marginalizing over z -- by integrating over the joint probability distribution for all possible values of z.
![[Pasted image 20240703204019.png|300]]
Unfortunately, this integral can't be easily estimated (because our data only sparsely populate the z space, so MC sampling isn't easy).
- Realize: We don't need to compute p(x); we just need some way of ***increasing*** it as much as we can, for our dataset examples. We do this in a Bayesian framework.

If we have a model with parameters H that in a sense form some *hypothesis*, Bayes' rule can be used to estimate the probability of a particular hypothesis H, given some observations w call the evidence E.
![[Pasted image 20240703204349.png|300]]
It's called the marginal likelihood because it's the probability of observing the evidence regardless of the hypothesis; marginalized over the hypothesis.
We can adapt it to our situation:
![[Pasted image 20240703204405.png|300]]
Ideally, we want to maximize the model evidence $p_{\theta}(x)$; we can then rearrange this:
![[Pasted image 20240703205117.png|300]]
Dang, a lot of intractabilities! But we know that we can target the right side -- can we shift our attention from $p_{\theta}(x)$ to instead trying to directly approximate the posterior, $p_\theta(z|x)$ .
- It turns out that we can do this with the method of ==Variational Inference!==
![[Pasted image 20240703205359.png]]
==[[Variational Bayesian Inference|Variational Inference]]== (here, "amortized VI," technically) is the process of approximating some target distribution p, with an approximation distribution q, parametrized by phi. If p represents a bayesian distribution, this is often called "variational bayes" or "variational bayesian inference."
- This q should be a sufficiently flexible family of distributions, such that by fiddling with phi, we can push these distributions as close together as possible.
A common objective for variational inference is the [[Kullback-Leibler Divergence]]. It's a sort of relative information entropy that we incur by diverging from the true distribution p to q.
![[Pasted image 20240703210019.png|400]]
So let's find the KL divergence from our target to our approximation:
![[Pasted image 20240703210043.png]]
- Use the definition of expectation to rewrite the integral as an expectation over the $q_\phi$ distribution.'
- Use the identity of log(a/b) = log(a) - log(b) to split the equation.
- Using the chain rule of probability (rearranged) to turn p(z|x) into p(z,x)/p(x)
- Again using the log identity to split the equation
- Rearrange the terms in the same expression
	- Recognize that $E_{q_\phi}logp_\theta(x)$  on the right side doesn't depend on z.
	- Rearrange the terms, grouping $E_{q_\phi}logq_\phi$  and $E_{q_\phi}logp_\theta(z,x)$  together (both depend on z and are under the same expectation; ==the expectation of a difference is the difference of expectations==.)
Easy, right? 🤪
- Immediately we see that this KL divergence can't be computed directly, because we know that the marginal likelihood of x is intractable. 
Let's reshuffle the terms!
![[Pasted image 20240703222418.png]]
We know that the KL divergence is non-negative, so we know that our log likelihood on the left side of the equation is always greater than or equal to the following term:
![[Pasted image 20240703222444.png]]
This essentially sets a lower bound on the value of the log likelihood. Since in a Bayesian context this p(x) is called the "model evidence," we call this term the [[Evidence Lower Bound]] (ELBO). 
- We can alternatively derive the ELBO using a mathematical result called [[Jensen's Inequality]], which basically states that the expectation of a concave function is always less than or equal to that function *applied on the expectation*.
	- (Any function with a line connecting two points on a concave line always lies below the curve -- see [[Jensen's Inequality]] for more)

![[Pasted image 20240703223428.png]]
Anyways, using this result, we can think of the KL Divergence from p to q as the *gap* between the ELBO and the *actual* log likelihood.
- A lower KL divergence increases the tightness of this bound.
- Maximizing the ELBO by optimizing Phi and Theta will do two things:
	- Maximize the $logp_\theta(x)$ 
	- Minimize the $D_{KL}(q_\phi||p_\theta)$ from the true posterior p to the approximation q.
In other words, both the generative model and the inference model are simultaneously optimize, all without having to explicitly calculate p(x) itself!

So how do we maximize the ELBO of a large set of datapoints?
- We could use stochastic gradient descent with respect to phi and theta, sampling mini-batches and performing gradient descent over the parameter space.
![[Pasted image 20240703223650.png]]
 We can easily differentiate the ELBO function with respect to theta to get a gradient, but it's problematic to differentiate it with respect to phi, because phi itself defines the distribution under which the expectation is taken. Let's analyze the gradient to understand:
 ![[Pasted image 20240703223756.png]]
 (Notice for the last part we removed the log q term because it doesn't depend on theta)

Let's try it for Phi... 😅
![[Pasted image 20240703223836.png]]
It turns out that this second term isn't an expectation and so can't be estimated easily, and we can't directly compute it either, since in the general case we don't have an analytical solution to the phi gradient of q.
- It turns out the solution here is to REPLACE Q with some equivalent distribution that's NOT parametrized by phi!
- We express z as some function g of phi, x, and a random variable eta, which is sampled from p(eta):
![[Pasted image 20240703223954.png|200]]
The trick is to define g such that phi and x influence g deterministically, and all of the stochasticity comes from the eta variable, whose distribution remains constant throughout training; we've externalized the randomness by transferring it from z to eta. This is called the ==Reparameterization Trick==, which lets us efficiently estimate the gradient of the ELBO
![[Pasted image 20240703224248.png|400]]
- Now, the expectation is taken not over q, but over p(eta), which allows us to estimate it using Monte-Carlo sampling, since the gradient of the expectation is the expectation of the gradient. 
- This whole scheme of estimating the ELBO using the Reparameterization Trick is called the "Stochastic Gradient Variational Byes (SGVB) estimator"

Phew! 😮‍💨
![[Pasted image 20240703225438.png]]
How do we actually compute the log likelihood itself?
- It depends on the type of distribution we choose to model $p_\theta(x|z)$ 
	- For binary-valued x's, like in B&W MNIST, we might use a [[Bernoulli Distribution]]
	- If x is expressed in real numbers, we might choose to use a [[Gaussian Distribution]].

If we assume that the generative model (p(x|z)) outs follows a Gaussian distribution parametrized by some mean and standard deviation for every data variable.
![[Pasted image 20240703230108.png]]
We want to maximize this thing.
- We can see that we want to minimize this mean squared difference between x and the mean of the distribution, as well as minimize the standard deviation sigma.
- There are difficulties faced by optimizing both mu and sigma!
	- You could get a high log probability by generating mu values very close to x, and pushing sigma down to some low value. But this can lead to numeric stability in the objective, and lower performance by encouraging the model to focus on particular data variables. It might generate some pixels very well, while performing poorly on the rest of the image.
	- So practitioners usually forego trying to learn sigma, and usually set it to 1.
	- So the first term vanishes, and we're left trying to just minimize the MSE between x and our mean:

![[Pasted image 20240703230349.png]]
y_mu is now the only output of the generative model, which is now a deterministic model on z.

Now let's sketch out the operation of the overall model (assuming gaussian prior and posterior):
![[Pasted image 20240703231018.png]]
- In this scheme, we have one deterministic model parametrized by phi that maps some input x to two outputs, mu and sigma. Some stochastic variable eta is sampled from a fixed distribution, and we use these all to sample the latent variable z.
- Another deterministic model parametrized by theta maps z to an output y.
- To ensure maximum flexibility in our two models, we want a highly-expressive model like a NN.
Our model is an [[Autoencoder]] that learns to reconstruct an input through a low-dimensional bottleneck that captures the "important" information in the input. But instead of deterministically mapping x to z like in a normal autoencoder, we sample  z from a stochastic model on x that's been learned through [[Variational Bayesian Inference|Variational Inference]]; so we have a [[Variational Autoencoder]].

Our regularized latent space has been trained to have two important properties:
1. ==Continuity==: Smooth transitions in latent space correspond to smooth transitions in data space. The decoder is encouraged to reconstruct the same output regardless of which z is sampled from the posterior during training ((?)); all points in the vicinity of the reference point are encouraged to reconstruct similar-looking outputs; so a point between two reference points could be expected to have a reconstruction that looks like a mix of the reconstructions produced by the two reference points.
2. ==Completeness==: There's a complete mapping between the prior in latent space and the target distribution in latent space.

We can imagine various losses to use for our VAEs
![[Pasted image 20240703231516.png|400]]
You don't want to put too much weight on either our reconstruction loss or our KL loss. 
The KL divergence pulls them together, and the reconstruction loss pushes them apart; we want to strike a balance between the two (this doesn't mean that alpha and beta have to be the same). Theres some benefit to having a Beta greater than Alpha (see: [[Beta-VAE]]).

---
Explanation from MIT 6.S191 (2024)
![[Pasted image 20240702201359.png]]
No matter how many times you put 2 in, you'll get that same reconstruction out. This isn't super useful if we want to generate new samples, because all the network has learned is to reconstruct -- we want some more variability -- some more randomness -- a real probability distribution!

In Variational Autoencoders, we introduce stochasticity/sampling to the network itself!
- They're a probabilistic twist on autoencoders, where we sample from the mean and standard deviation to compute latent samples!
![[Pasted image 20240702201502.png]]
In the bottleneck/latent layer, instead of learning those latent variables directly, let's say we have means and standard deviations for each of the latent variables that now let us define a probability distribution for each of these latent variables.
- Now we've gone from a vector of latent variables $z$ to a vector of means $\mu$ and a vector of standard deviations $\sigma$.
Now we can use this to sample from these probability distributions to generate new data instances.

We assume the distributions are normal -- why is that a reasonable assumption?
![[Pasted image 20240702202849.png]]
The new regularization term is introduced because we need to make some kind of assumption about what z's probability distribution actually looks like (we chose normal, parametrized by mu and stdev).
- The regularization term says "okay, we'll take this prior and compare how good the representation learned by the encoder is, relative to that prior... how close does it match our assumption about (eg) the normal distribution". You can say it's capturing the difference between the inferred learned distribution q, and some prior/guess we have about what those latents should look like.

So how do we choose that prior?
- Commonly, we assume a normal gaussian distribution (mean 0, stdev/var 1)
- The rationale is that it encourages the model to distribute the encodings of the latent variables roughly evenly around teh center of teh latent space.
![[Pasted image 20240702203159.png]]
![[Pasted image 20240702203210.png]]
Briefly, to touch on the form of what the regularization term looks like. The [[Kullback-Leibler Divergence|KL-Divergence]] is (sort of) a measure of distance between distributions. When we assume gaussian, the KL divergence takes a pretty clean form shown here.
![[Pasted image 20240702203517.png]]
If we enforce regularization, points that are close in latent space have a similar meaning after being decoded.
![[Pasted image 20240702203808.png]]

Skipping the "==reparametrization== trick" which helps us train the model end to end.

![[Pasted image 20240702204311.png]]
We want our latent variables to be as uncorrelated with eachother as possible, to maximize information in the latents. It's a common question to ask how we can disentangle latent variables like head pose and smile.

A straightforward solution to encourage is to relatively weight the loss's reconstruction term and the regularization term.. to encourage disentanglement
![[Pasted image 20240702204415.png]]
![[Pasted image 20240702204434.png]]
Above: [[Beta-VAE]]

![[Pasted image 20240702204644.png]]

----

Explanation from Umar Jamil's video

![[Pasted image 20240710150033.png]]
The z learned for a cat might be very similar to the z learned for a pizza! There's no semantic relationship captured in the code.
![[Pasted image 20240710150119.png]]
The latent  space captures the semantic relationship of the data too! So all of the food pictures will have a similar representation, all of the animals too, etc. 
- Importantly, we want to be able to *sample* from this latent space to generate new data!

![[Pasted image 20240710151906.png]]
Why do we call them *latents*, and *latent space?*
- We model our data (random variable x, observable) as if it were conditioned on some other random variable z that we cannot observe (latent means hidden). 
- We model this hidden variable as a multivariate gaussian parametrized by means and variances.

First, a Math Pep Talk

[[KL Divergence]] allows you to measure the difference between two probability distributions; It tells you how "far" two distributions p and q are from eachother. But it's not a true distance metric, because it's not symmetric.
- But just like any distance metric, it's always >= 0, and \==0 only if the two distributions are the same.

So we want to define our data as coming from some random distribution x, which is conditioned on some hidden data z.
![[Pasted image 20240710152936.png]]
We want to find p(x), but we don't have the ground-truth p(z|x), the probability of the latent given the real data. So we can't use this relationship either!

We have a chicken and egg problem!
![[Pasted image 20240710153038.png]]
How do we come out of it? If we can't find something we want, we usually try to find out how to approximate it.
![[Pasted image 20240710153141.png]]
We want to find a *surrogate* for $p_\theta(z|x)$ ... and we'll assume that this surrogate is parametrized by its own parameters $\phi$.

![[Pasted image 20240710153437.png]]
We can do some maths:
- Start with log likelihood of its data equal to itself
- We can multiply by one (the integral over the domain of a probability distribution is always equal to one)
- We can then bring the log(x) quantity inside the integral because it doesn't depend on the variable being integrated
- Using the definition of expectation, we can rephrase...
- Inside of the expectation, we can apply the equation given by the chain rule of probability
- We can then multiply and divide by the same quantity of q(z|x)/q(z|x)
- We can then split the expectation, because the log(ab) = log(a) + log(b), and then split the expectation
- We see that the second term is the KL divergence, which is always >= 0

What can we infer?
![[Pasted image 20240710154941.png]]
We call this the [[Evidence Lower Bound|ELBO]]
What can we infer from this? As an example, the salary equation below. Without knowing anything about your bonus, you know that your total compensation is going to be *at least* your base salary! So your total compensation >= base salary. This is the same for our situation:
![[Pasted image 20240710154929.png]]
We can say this, without caring about what happens to our KL divergence property. So if we *maximize* this ELBO term, we also *maximize* the log probability of our data.

If we want to maximize this [[Evidence Lower Bound|ELBO]], let's look closer into it:
![[Pasted image 20240710155151.png]]

![[Pasted image 20240710155647.png]]

![[Pasted image 20240710155815.png]]
There exists a very high-variance estimator for the ELBO. We can't use this high-variance estimator in practice... how do we run backpropagation on a quantity that's stochastic? How do we calculate the derivative of a sampling operation? Pytorch can't do that. So we need a new estimator!

![[Pasted image 20240710155945.png]]
==Reparametrization Trick==: We move the source of randomness outside the model!

![[Pasted image 20240710160001.png]]
We create a new variable $\epsilon$ that's our new stochastic node, and we combine the sampled information from it with the mu and sigmas of our multivariate gaussian we're tryin to learn... then run backpropagation through it.

![[Pasted image 20240710160037.png]]
![[Pasted image 20240710160212.png]]
![[Pasted image 20240710160305.png]]
- We found something called ELBO, which, if we maximize it, we'll learn the latent space.
- We found an estimator of ELBO that allows backpropagation to be run.
Above: Why are we learning $log(\sigma^2)$ instead of $\sigma^2$? If we learned sigma squared, we'd force our model to learn a positive quantity, because sigma squared cannot be negative. So we pretend to learn sigma squared by learning log sigma squared, and then later exponentiating it to recover the sigma squared.

Imagine now we have a picture
- We run it through the encoder; given our picture, it gives us the latent representation.
- We sample from the noise source outside the model, and combine it with the parameters learned by the model.
- We pass this latent to the decoder, to reconstruct an image.
- Determine a reconstruction error comparing our input image and reconstructed image.

![[Pasted image 20240710160443.png]]
The loss function has two components:
- Tells how far our learned distribution is from what we want our distribution to look like.
- The quality of the reconstruction (boxed).