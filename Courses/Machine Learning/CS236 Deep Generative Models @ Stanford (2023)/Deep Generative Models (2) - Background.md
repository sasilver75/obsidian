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
The challenge is that this independence assumption is too strong -- we can't just choose the values of pixels independently and get anything that looks like our p_data distribution.


![[Pasted image 20240702112625.png]]
[[Chain Rule of Probability]]
- Basic idea is that you can always write down the probability of a bunch of events happening the same time as a product of conditional probabilities. 
[[Bayes Rule]]
- Allows us to write the conditional probability of one even given another one as a function of the prior probability...and the likelihood of s2 happening, given s1.


Using chain rule, we can always take a joint distribution over n variables, and write it down as a product.
- This is the kind of factorization that's used in autoregressive models, which are the first class of models that we'll talk about.
- We write down the probability of observing a sequence of words in a sentences as the probability of observing the first word, times the probability of observing the second word given the first, times the probability of seeing the third word given the first and second, ...


![[Pasted image 20240702114858.png]]
How many parameters do we need, if we do this sort of factorization?
- We still need an exponentially large number of params, unfortunately... not free lunch. We haven't made any restrictive assumptions to get this factorization, so we can't get any savings. 
- 1, 2, 4, ...

What if we assumed conditional independence?
- That the value of the i'th plus one word is conditionally independent... only on the i'th word? ([[Markov Assumption]])
- If you're willing to make this assumption, you get big savings!
![[Pasted image 20240702115011.png]]
Becomes probability like p(x_3|x_2)
Generally, $p(x_n|x_{n-1})$   ... we're always conditioning on at most one variable!
So we need $2n - 1$ parameters -- linear! That's a much more reasonable model than the full-independence model.
These Markovian models are quite useful, but if you think about language or images, you're probably not going to do a great job if you think about it -- autocomplete just based on the previous word? It's going to be okay... but not great -- you need more context to make a good prediction.


----

![[Pasted image 20240702115340.png]]
One way to generalize this idea is to use a ==Bayesian Network==
- We write down the joint as a product of conditionals, but instead of it being one variable given another variable, we use conditional distributions where the i'th variable depends on another set X_A_i of random variables.
- Intuitively, we try to write down the joint as a product of conditionals, but now the conditionals are a little more complex.
	- Each variable is allowed to depend on some 1+ subset of variables.
Idea: Because we're using chain rule, as long as there's some kind of ordering that you've used to come up with this joint distribution... then it's guaranteed to correspond to a valid model.

More formally...
![[Pasted image 20240702115501.png]]
A Bayesian Network is a data structure we can use to specify a probability distribution -- it's a graph-based data structure where there's an underlying DAG that gives us the ordering in our chain rule factorization
- One node in the graph for each random variable we're modeling
	- One word for every pixel
	- One word for every text token
For every node in the graph, we specify its conditional probability distribution specifying the variable's probability conditioned on its parents value.
*Given a DAG, we can always come up with an ordering (eg topological sort on the graph), apply chain rule to factorize with respect to the ordering, and simplify the conditionals with some conditional independence assumption.*

![[Pasted image 20240702130726.png]]
We only need to work out how these random variables are related to eachother locally, with respsect to the graph.
- We break down the complexity of the joint probability in terms of small local interactions of the random variables.

By making this assumption that global dependencies can be broken down into smaller local ones... you benefit! Because these smaller conditional probabilities are often easier to represent.

It turns out that assuming the above is ~the same as assuming conditional independencies?
![[Pasted image 20240702130946.png]]
If we assume that the intelligence of the student doesn't depend on the difficulty of the exam, then we can start simplifying the conditionals, and they become more like the one seen above (the first one, rather than the second one).

[[Bayesian Network]]s are ways to simplify complicated probability distributions based on conditional independence assumptions between variables (which is more reasonable 
than assuming *full independence* of the variables)

![[Pasted image 20240702131112.png]]
In this class, the graphical models that we use will be relatively simple, involving only 2 or  3 random variables/vectors.
- Instead, we'll make softer notions of conditional independence, which will be the ideas of using neural networks to represent how different variables are related to eachother.

---

[[Naive Bayes]] example
- Simple generative model to solve a discriminative task (to then contrast with neural newtorks): Trying to predict if an email is spam, given features X_i that are binary (say, on or off depending on if a word in the vocabulary is in the email)

We assume that there's some relationship between our label Y and the X's in the email
We can approach this by building a bayesian network; this is a basic Naive Bayes classifier.
- We want to model this joint distribution by making a conditional independence assumption and assume that the joint can be modeled by this DAG:
![[Pasted image 20240702131643.png]]
This means that the features (words, X_is) are basically conditionally independent (from eachother), given Y.
- We can then factorize the joint as a product of conditionals:
![[Pasted image 20240702131710.png]]
According to this very simple model of the world, we can generate a sample by:
- First choosing whether it's spam or not
- Then choosing the different words (in any order, since the aren't conditionally independent on eachother) in the email, conditioned on whether it's spam or not.

Once we have this kind of model, we can do things like:
- Estimate the parameters of this model (the probabilities) by looking at data
- Can do classification on new data!

![[Pasted image 20240702132018.png]]
Above: Given a new email, compute the probability Y by using [[Bayes Rule]]
- In this case, that's P(x,Y)|p(x), where x here is the whole sequence
	- ((In this case, the numerator is the probability of generating a sequence, given that's is spam, and the denominator is the probability of generating a sequence regardless of whether it's spam or not. This intuitively gives the probability that the label is spam, given the generated text))

This model might perform reasonably well at predicting the label Y, given features X
- But these conditional independence assumptions between the words that we're using aren't really realistic, are they? Never the less, this tends to work *okay* in practice.

But how does this all fit into the discriminative/generative taxonomy?
![[Pasted image 20240702132410.png]]
These are both equivalent bayesian networks, using the chain rule of probability.
- At the end of the day, if all we care about is P(Y|x), then the second one on the right is usually what we're going to do.
- In the left model, we need to specify/learn both P(Y) and p(X|Y), then combine P(Y|X) via Bayes rule... while while in the right model, it suffices to estimate just the conditional distribution P(Y|X)... and there's no reason to learn p(X).

![[Pasted image 20240702132623.png]]
A generative model, in contrast, is able to reason about its inputs too -- the full relationship between x and y.
- But there's no free lunch! In both case you end up with some conditionals that are pretty complicated:

![[Pasted image 20240702132805.png]]
In the discriminative world, we still have to model how Y depends on Xs, and Y has a lot of parents...
For the generative model, we generally make assumptions of conditional independence between the Xs (eg Naive Bayes)
![[Pasted image 20240702132843.png]]
Ending up with something that's much simpler.


![[Pasted image 20240702141055.png]]
Repeating these nonlinearities multiple times can help us get more expressive ways of approximating p(Y|x).
- We can use chain rule and neural networks to represent our conditionals.

![[Pasted image 20240702141326.png]]
- Chain rule factorization is *fully general*; with no assumptios, we can always write a joint as the product of conditionsals..
- With a bayesian network, we simplify these conditionals by assuming some variables are conditionally independent.
	- This is usually too strong an assumption, and doesn't work well on high-dimensional datasets like images
- In Neural models, we replace the conditionals we don't know how to deal with with neural networks.
	- We use it to predict (eg) "Whats the fourth word, given the first, second, third"
	- We assume that these conditional distributions can be captured by a neural network, which may or may not be the case in practice.


What if we're dealing with continuous random variables?
![[Pasted image 20240702142456.png]]
For continuous ones, we can't use our tables -- so we have to use the probability density function of our specific distribution.
- We can use any sort of distribution, or way of modeling distribution -- eg mixing multiple gaussians.
![[Pasted image 20240702142651.png]]







