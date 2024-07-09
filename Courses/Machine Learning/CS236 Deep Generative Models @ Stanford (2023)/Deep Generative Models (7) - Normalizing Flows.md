https://www.youtube.com/watch?v=m6dKKRsZwBQ&list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8&index=7&t=4282s

We're going to finish up our slides we didn't finish for VAEs and then start talking about flow models.

----


Let's talk about the interpretation of a VAE as an AutoEncoder...
- We derived it from the perspective of there being a latent variable model, and that there's a [[Variational Bayesian Inference|Variational Inference]] technique for training the model, where we have a decoder defining the generative process p, and an encoder network q that's used to output the variational parameters that are supposed to give you a good approximation of the posterior p(z|x)...

![[Pasted image 20240708200315.png]]
We have a training objective that depends on the parameters of the decoder parametrized by theta and the decoder parametrized by phi, and we saw that this function was a lower bound to the true marginal probability of a datapoint p(x) ([[Evidence Lower Bound|ELBO]]), so we tried to jointly maximize this function as a function of theta and phi.
- For every datapoint x, we use q to guess z for x. (we do an expectaiton with respect to this)
- And then we look at the log likelihood of the datapoint after we've guessed what we don't know (the z), and... if we just optimized this, then q would be incentivized to try to find completions that are most likely under the original generative model...
- so instead, we also want to add this -logq term, which acts a a regularizer, where we also look at the probability of the completions, under q. This corresponds to the entropy of the variational distribution q term, and kind of encourages the distribution to spread out the probability mass (not just finding the most likely z, but the most possible zs that are consistent with the x we have access to)
If you q is sufficiently flexible and approach the true p(z|x), then the objective function becomes exactly the log probability(x), which is the traditional [[Maximum Likelihood|MLE]] objective.

![[Pasted image 20240708221735.png]]
If we change our formula by just adding and subtracting the log probability of z... then you divide by log(pz)  (dividing the joint by the marginal gives the conditional)... and you end up with line 3, with the [[Kullback-Leibler Divergence|KL-Divergence]] between the inference distribution and the prior.
- If we were to evaluate this objective and do some MC approximation, we would take some datapoint (eg an image) x, map it to a z by sampling from our q(z|x) encoder. This encoder output might be something like a mu and a sigma, if you're using a gaussian distribution. Then we could sample from that gaussian distribution.
- We then use our decoder to reconstruct x by sampling from p(x|z). We sample our x from a Gaussian with parameters *decoder*(z).

In a typical Autoencoder, you'd take an input, map it to a vector in a deterministic way, and try to map it back to the data.
This is kind of like a stochastic autoencoder, where we map an input to a *distribution over latent variables*. The latent variables that we sample from the distribution should be useful/good to reconstruct the image.

The first term is like an AutoEncoding loss, because we're producing z|x, and then asking about the log probability of producing x|z.
KL divergence term:"Not only should you be able to reconstruct well, but the kind of latent variables you use should be distributed as a gaussian random variable (like p(z))"

![[Pasted image 20240708223838.png]]

Q: When you're training, how do you compute the Divergence? Doesn't it require p(z)?
A: Yeah, just look at the first line of that series of equations from the previous slide; in that one, every item is computable.


![[Pasted image 20240709004051.png]]
Flow Models are another way about going about intractability of the marginal probability (of p(x)?) in a latent variable model
- Autorgressive models are nice because we directly have access to the probability of the data (via calculating out the chain of conditionals), so we don't have to deal with variational inference and encoders/decoders.
- Variational autoencoders are nice because we can actually define pretty flexible marginal distributions... can generate in one shot, too.
	- Problem is that we can't evaluate the marginal probabiliy (p(x)), so we have to come up with the ELBO...

Flow Models are types of Latent Variable Models (kind of like VAEs) but have a special structure so that we don't have to do [[Variational Bayesian Inference|Variational Inference]], and can train them by [[Maximum Likelihood]]

