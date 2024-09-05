---
aliases:
  - LCM
---
References:
- Blog: [Explained: Latent Consistency Models](https://naklecha.notion.site/explained-latent-consistency-models-13a9290c0fd3427d8d1a1e0bed97bde2?pvs=97#48d3f1efa9e64e269b42b0e2b22242a2)

Traditional [[Latent Diffusion Model]]s (LDMs) like [[Stable Diffusion]] are very slow.

![[Pasted image 20240710034750.png]]
Above: The architecture of [[Stable Diffusion]]
- Given an input image $x$, it's encoded using an encoder $\mathcal{E}$ which produces $z$, a latent space vector with a much smaller dimensionality than the image. Think of it as a compressed vector representation of the input image $x$.
- Now we add gaussian noise $T$ times to the vector $z$ , after which we're left with a noised $z_T$ vector. This process of adding noise is called the (forward) diffusion process. Note that we retain partially-noised $z_i$ vectors from each step.
- After a large enough number of steps, the noisy latent vector looks like purely random noise:
	![[Pasted image 20240710035320.png|100]]
- If we have a text input along with our image, we separately encode that into a same-dimensioned latent vector, which is then *added* to our noisy latent vector $z_T$ in order to guide the (backward) diffusion process.
- Now, we train a model that can *denoise* the noisy vector $z_t$, using a model called a [[U-Net|UNet]].
	- When correctly trained, the denoising UNet model should be able to accurately predict the *noise* that was added to $z_{t-1}$ to produce $z_t$.
- This denoising step is repeated T times, feeding the output from 1 step into the input for the next step. At the end of the T denoising steps, our denoised $z_t$ vector should hopefully now be a close approximation of the original latent vector $z$.
- The new denoised $z$ is then *decoded* into an $\hat{x}$ , which should look very similar to the original image.
- After training, when we later want to generate a new image using diffusion, we generate a random noisy vector $z_t$, denoise this $z_t$ vector to receive a noiseless vector $z$, and then decode that $z$ latent into an image.


### Why are Latent Diffusion Models so slow?

These are the number of iterations required (?) by Stable Diffusion, and how long single iterations take:
![[Pasted image 20240710141051.png]]
So vanilla Stable Diffusion on a powerful GPU can generate 5-10 images per second -- something like 100 images/second would be nice!

Could we cut down the time it takes to generate an image by reducing the number of iterations required? Perhaps even to the point where we remove all of the noise from the image in a single step? This is what [[Latent Consistency Model]]s (LCMs) do!

Our LCM:  $f_{\theta}(x, t) \implies Image$
where $x$ is a noisy latent vector and $t$ is the amount of noise that was added.

The authors propose the following equation:
![[Pasted image 20240710141728.png]]
$f_{\theta}(x,t) = c_{skip}(t)x + c_{out}(t)F_{\theta}(x,t)$
- $c_{skip}(t)$ is a differentiable function controlling the proportion of the original noisy image
- $c_{out}(t)$ is a differentiable function that controls the proportion of neural network $F_{\theta}$
- $F_{\theta}$ is a deep neural network that captures the complex patterns and relationships in images, allowing it to effectively denoise images even at high noise levels.

Our goal is to learn the three functions ($c_{skip}$, $c_{out}$, and $F_{\theta}$)

Note that in an ideal world where $f_{\theta}$ is a perfect single-step denoising function, the following equation is satisfied:
![[Pasted image 20240710142653.png]]
This is called the ==Self-Consistency Property==. This means that the function f produces the same output, regardless of the specific time step (t or t'), as long as the input x is also appropriately time-indexed.
- So if I say "Hey, denoise this image (x_2, 2), which has had two noisings applied to it" or "Hey, denoise this image (x_8, 8), which is the same image but with eight noisings applied to it", then we should get the same output image for each.
- The goal is to maintain consistency in the latent space representations across different time steps or noise levels in the denoising process.

Now in order to train the LCM model, let's consider the loss function:
![[Pasted image 20240710143217.png]]
- $\theta$: The parameters of the current consistency model (the weights and biases of the model we're training)
- $\theta^-$: The target model parameters - our updated version of $\theta$ using exponential moving average (EMA). "This might as well be $\theta$, but EMA is used to smoothen out the training process."
	- $\theta^- \leftarrow \mu\theta^- + (1-\mu)\theta$  where $\mu$ is the EMA decay rate
		- ((Nice choice of symbols idiot))
- $f_{\theta}$: Output of the consistency model $f_\theta$ for the current model
- $f_{\theta^-}$: Output of the consistent $f_{\theta^-}$ for the target model
- $d$: A distance metric; ideally, if the model is trained well, the distance between the two functions is zero. ==This ensures that the self-consistency property is satisfied.==

Training process:
- Initialize a $\theta^-$ using a copy of $\theta$
- Calculate the ==consistency loss== L($\theta$, $\theta^-$; $\Phi$) 
- Update $\theta$ using gradient descent to minimize the consistency loss L($\theta$, $\theta_-$; $\Phi$)
- Update $\theta^-$ using an exponential moving average:
	- $\theta^- = \mu\theta^- + (1-\mu)\theta$ 

We still have a few problems though:
1. We have to train this model from scratch
2. We need millions of labeled images to train
3. It's going to be extremely expensive
4. Probably won't be as good as stable diffusion

Soution:
1. We grab stable diffusion
2. Steal their weights and leech their knowledge via distillation
3. Finetune or train a smaller model
4. Boom, we have the self-consistency property!