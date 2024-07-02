https://www.youtube.com/watch?v=rNEujZmD2Tg&t=4186s


Agenda:
- What is a generative model?
- Representing probability distributions
	- The curse of dimensionality (the first challenge we run into; we'll talk about various ways to deal with it)
	- Crash course on graphical models (Bayesian Networks)
	- Generative vs Discriminative models
	- Neural models

---

![[Pasted image 20240702102739.png]]

Assumption is that the data that we have access to are samples from some P_data representing some *unknown* common data-generating process.
- We don't have access to P_data, only samples that have been drawn from it.

We want to generate some good approximation to this P_data distribution; we can then sample from this distribution by sampling from it, generating new images or text!
- We define some model family (eg all gaussian distributions), and then search over this set to find some good approximation of the data distribution within this set.
	- ==It's not clear what the right model family is==
- We do this by defining some notion of distance/loss between our distribution and the data distribution -- what is it that we ***care*** about?
	- ==It's not clear what the right loss/distance is==
- Then it becomes an ***optimization problem*** where we try to find the point in our model family that minimizes this loss!


We want to learn a probability distribution p(x) over images x such taht:
- ==Generation==: If we sample some x_new ~ p(x), then x_new should look like a dog (sampling)
- ==Density Estimation==: p(x) should be high if some input x looks like a dog, and low otherwise (eg for *anomaly detection*)
- ==Unsupservised representation learning==: We should be able to learn what these images have in common, e.g. ears, tails, etc. (features) In order to do well at our task, you have to understand what all of these images have in common -- so we learn "what does a dog look like," etc.


So... our first question: How do we *represent* p(x)?
- How do we actually come up with some reasonable set of models that we can optimize over with respect to some loss between a candidate model and our P_data data-generating distribution.

---

If we're dealing with low-dimensional data, the problem isn't so hard:
- If we have a single discrete binary random variable, it might not be so hard to assign probabilities to these events.
![[Pasted image 20240702104322.png|400]]
For a Bernoulli distribution, we just need a single parameter to tell us the probability of heads. This isn't going to be enough for distributions of images or text, though.
![[Pasted image 20240702104409.png]]
If we have k different outcomes, then we often have a categorical distribution of some sort -- we can use this to model rolling a die. The challenge is that if we have k different things that can happen, we need to specify a probability for each one of them (and they have to sum to one).

We can combine these two building blocks to model more interesting objects!
- Say we want to do a probability distribution over many images!
	- We'll have to model many single pixels
	- For a single pixel, we'll have to specify three numbers, (red, green, blue channel intensities).
![[Pasted image 20240702104545.png|300]]
Now we can describe the many colors we can get for a specific pixel.
- Sampling from the joint distribution (r,g,b) ~ p(R,G,B) randomly generates some color for the pixel.
- But how many parameters do we need to specify the joint probability distribution p(R=r, G=g, B=b)?
	- How many things can happen? 256x256x256-1 colors... and we have to specify a non-negative probability to each one of them... and it all has to solve to one... but it's still a reasonably high number.
	- The space of possible things that can happen grows exponentially when we have many random variables to model.

For a random distribution over images (say, black and white ones)
- One random variable for every pixel (say it's a 28x28 image), and each pixel by itself is a bernoulli random variable that's either on or off, white or black.
- Say we have an MNIST training set... and we want to learn a probability distribution over these images!
- ![[Pasted image 20240702104825.png|400]]
We can think about how many different images are there with n pixels; $2^n$ possible images, where n is the number of pixels in any image.

Sampling from p(x1, ..., x_n) will generate an image
- If you've done a good job, the image you sample will be like those in the training set

But how many parameters do you need to specify the joint distribution p(x1, ..., x_n) over n binary pixels?
- $2^n - 1$   ... There are $2^n$ possible things that can happen...
This is more than the atoms in the universe, even for a small number of n... so we need some kind of tricks/assumptions to deal with this complexity.


One way to make progress is to assume something about how random variables are related to eachother.
- A strong assumption is that all of these random variables are *independent* to eachother! If they're independent, then the joint distribution can be factored as a product of the marginal distributions.
If we make this assumption, then how many different images are there?
- Still 2^n possible images... We still have a probability distribution over the same space... 
- But we can drastically reduce the number of parameters! Just $n$ ; we just need to be able to store each one of these entires; each one of these marginals -- these are just bernoulli random variables, so a single parameter for each.
![[Pasted image 20240702112513.png|300]]
The challenge is that this independence a



