https://youtu.be/tRArbBf-AbI?si=muv-W5sDNSR6KJ6Z

Autoregressive models will be our first family of generative models that we'll consider in the class.

Recap of previous lecture:
![[Pasted image 20240702180557.png|400]]
We're going to be thinking about probability distributions; we want to learn some parameters setting within our model family that results in a function that helps us approximation the P_data distribution as we see it from the $x \in X$ in the empirical distribution that we drew from true data distribution.

Once we have this probability distribution, we can:
- ==Generation/Sampling==: We can sample new $x_{new} \sim p(x)$, where the x_new should look like a dog.
- ==Density estimation==: Given an x, p(x) should be high if x looks like a dog, and low otherwise (useful for anomaly detection)
- ==Unsupervised representation learning==: We should be able to learn what sort of things these images have in common (eg tails, ears, etc) (features)

----

![[Pasted image 20240702181820.png|300]]
- The autoregressive models we're going to talk about today are basically build on the Chain Rule of Probability, where we can take any joint probability distribution and factor is as a product of conditional probabilities. Fully general; No assumptions are needed, and you can use any ordering you want. 
- Bayes Nets essentially exploit the idea of causality, and try to make progress by simplifying our conditionals. Makes some conditional independence assumptions between certain random variables to cut down on the number of parameters.
- Neural models assume some specific functional form for our conditionals. If you can use very deep neural networks, there's a good chance you can come up with some good approximation of your conditionals; used a lot in practice.

![[Pasted image 20240702182222.png]]

Let's say you wanted to learn a generative model over images; Say we wanted to work with binarized MNIST, where each image has 28x28 pixels.
![[Pasted image 20240702182352.png]]
==There are still a lot of possible images, and we want to be able to assign a probability to each one!==
RECIPE:
1. Define a model family parametrized by theta
2. Search for model parameters theta in an optimization process based on some loss/distance function, using our training data.

To use an autoregressive model to define this probability distribution... we're going to be using the chain rule, and we have to pick an ordering (because there are multiple different ways to express a joint probability as a product of conditionals). For these images, there's no obvious ordering that we should pick for our conditionals; any ordering works in principle.

In "raster-scan ordering," we go from top-left to bottom-right... and apply chain rule in that order.
![[Pasted image 20240702182556.png]]
So we've broken down a large joint probability distribution into a series of conditional probability distributions.
But some conditionals are too complex to be stored in tabular form (so a bayesian network is probably out of consideration), so we'll model these conditional probabilities using some sort of (eg) neural model that let us map the different pixels we're conditioning on to a probability distribution over the next pixel...
- A simple thing we could do would just be use logistic regression
![[Pasted image 20240702183546.png]]
(Ignore the CPT thing)
- Notice that we don't have a single classification problem, we have a *sequence* of classification problems
	- Being able to predict the second pixel given the first one
	- Being able to predict the third one given the first two
	- ...
In general, we can even use different parameters or models for each one of these classification problems.
- Here, there's a different vector of coefficients $\alpha$ for each classification problem

![[Pasted image 20240702183748.png]]
We're using parametrized functions (eg logistic regression) to predict the next pixel given all of the previous ones. We call this an ==autoregressive== model because we're trying to predict parts of the datapoint given other parts of the datapoint.

Here's an early example of this type of generative model:
![[Pasted image 20240702183915.png]]
- When we think about chain rule... we have all of these individual pixels we're modeling on those that come before it in our order...
- When we model p(X_i| x_1...x_i-1; alpha), which we often denote as p(X_i=1|x_<i ; alpha)... in the case of logistic regression, that's given as;
- ![[Pasted image 20240702184031.png]]

==So if someone gives us a datapoint and wants to know how likely a datapoint is, given our model:==
- Evaluate p(x_1, .... x_784)
We would have to go back to chain rule.... and multiply together all the conditional factors!
If we just had four pixels, and we're observing 0110:
- Multiply together all of these values, which are basically the predicted probability that a pixel gives a certain value... and these predicted probabilities depend on the values of the previous pixels in the order.

![[Pasted image 20240702184249.png]]
predicted $\hat{x_i}$ depends on all of the pixels that come before it in the ordering.
(he's doing 1-xhat for the ones that are 0, since p(x_i) is the probability that x_i is 1/black, in this case)

==How do we sample==  from our model?
- Pick some first xbar_1
- Then sample xbar_2 as p(x_2|x_1=xbar1)
- Then sample xbar_2 as p(x_3 | x1=xbar1, x2=xbar2)

![[Pasted image 20240702184532.png|300]]
It's nice that sampling is to some extent easy... it's not great that we have to sequentially go through each random variable that we're working with... but better than using something like a MCMC method or other complicated techniques.

How many parameters in all of the alpha^i vectors? 1+ 2+ 3 + .... + n ~= (n^2)/2

So this type of autoregressive generation for images doesn't actually work that well:
![[Pasted image 20240702184813.png]]
Samples are eon the left, and the generations are on the right -- very blobby!
- This is because the logistic regression model isn't able to describe the relatively complicated dependencies that these pixels values have on eachother.

Can we make things better by using a deep neural network?
![[Pasted image 20240702185056.png]]
![[Pasted image 20240702185323.png]]
Tying the parameters to reduce the number of parameters and speed up computation... (tying together all these A1, A2, A3 matrices)
- And just tie together all the weights we use in the prediction problem -- treat it as selecting some progressively larger slice of a big matrix? So we're always trying to extract the same features from (eg) x1, and using them amongst all of the classification problems in the conditionals.

Q: is the bias vector c shared among all layers?
a: Could be, doesn't have to be

![[Pasted image 20240702190345.png]]


==So what do we do if we want to model *non-binary* discrete random variables==? Like if wanted to do a color image, instead of a black and white one where the pixel values are 0,1?
![[Pasted image 20240702190735.png]]
- We get a hidden vector or latent representation h
- Instead of applying/mapping down to the parameters of a bernoulli random variable, we use some sort of softmax output layer to map it down to a probability distribution over the k outputs that we care about.
	- This is the natural generalization of the logistic function we saw earlier for the binary classification case.



==But now what if we want to model continual data that isn't natural to discretize?==
![[Pasted image 20240702190924.png]]
The solution again could be to... have the output of the neural network be some continuous distribution (eg a gaussian, a logistic, or some continuous probability density function that we think should work well for our dataset)
- We could use a mixture of k gaussians, for instance!
- We need to make sure that the outputs of our NN give us the parameter of our K different gaussians, which are then mixed together to obtain a relatively flexible probability density function.
![[Pasted image 20240702191033.png]]
EG it's a mixture/average of K gaussians, each with their own mean, standard deviation.
- So we need k means and k standard deviations output by our model.

## Masked Autoencoders
![[Pasted image 20240702193943.png]]
For every datapoint, we train our autoencode to minimize some reconstruction loss (could be a loss like MSE, eg. for continuous...)
- Note also that our encoder and decoder are constrained so that they don't just learn identity mappings
- The hope is that e(x) is a meaningful, compressed representation of x
==But a vanilla autoencoder is not a generative model -- it doesn't define a distribution over x where we can sample from to generate new datapoints==

But if we enforce some sort of ordering on the autoencoder, we can get back an autoregressive model!
- If we put constraint on the network so that there is a correspondign bayes network or chain rule factorization, we can get an autoregressive model.
- The issue is that we need a way of generating data sequentially... One way to do it is to set  up the computation graph so the first reconstructed random variable doesn't depend on later inputs.
	- So we can come up with the first output ourself
	- And then feed the predicted first random variable into the network again; it's fine if the predict value for the second random variable depends on x_1.
![[Pasted image 20240702195025.png]]
TLDR if we mask the weights in the right way, we can get a single autoregressive neural network that does the whole shebang 🤔

But wait, how do we get all of the parameters in a single pass?
- The way is to *mask*; we enforce some kind of ordering, masking out connections so that we can generate data... this ordering can be anything.
![[Pasted image 20240702195410.png]]
Above: a [[Masked Autoencoder]]
- We mask out the network so that generating x3 is only allowed to depend on x2, and generating x1 only depends on x2,x3.
We essentially turn our auto-encoder into an autoregressive model by masking.

Q: Is this masking done during training?
A: Yeah, you have to set up an architecture that's appropriately masked, so it can't cheat when training.

Q: Is the benefit only at training time or inference time?
A: At inference time you still have a sequential thing too; that's unavoidable.

Q: How do you choose the ordering (of the chain rule breakdown of conditional probabilities)
A: That's hard - if you know there's structure in some way, then perhaps go that way... otherwise probably try many orderings, or choose one at random.

----

## RNNs
- An alternative way to approach the autoregressive problem is to use an RNN; some recursive style of computation to predict the next random variable, given the previous ones.
![[Pasted image 20240702200613.png]]
![[Pasted image 20240702201106.png]]
The key thing is that there are a small number of parameters, and we use the hidden state of the RNN to get the conditional probabilities we need in an autoregressive factorization.
![[Pasted image 20240702201151.png]]
Slow during training because we have to unroll the recursion to compute probabilities.

