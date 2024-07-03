---
aliases:
  - VAE
---
Variants: VQ-VAE (Vector Quantized-Variational AutoEncoder, when you want the latent to be discrete)

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
