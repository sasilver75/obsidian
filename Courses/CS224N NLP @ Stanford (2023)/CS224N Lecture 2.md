#lecture 
Link: https://www.youtube.com/watch?v=gqaHkPEZAew&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=2

---

Agenda: Word Vectors, Word Senses, and Neural Netwrok Classifiers
- Course Organization
- Finish looking at word vectors + ==Word2Vec==
- Optimization Basics
- Can we capture the *essence of word meaning* more effectively by *counting?*
- The ==GLoVe== model of word vectors
- Evaluating word vectors
- Word senses
- Review of classification and how neural nets differ
- Introducing neural networks

Goal: To be able to read word embedding papers

----


![[Pasted image 20240401123856.png]]
Review:
- We start with random word vectors, we have a big corpus of rtext
- For each sequence, iterate through, and try to predict what words surround our center word, and we do that with a probability distribution that's defined in terms of
	- The dot product between the center word and the context words
- Actual words *did* occur in the context of "into" in the above situation
	- we want to make it *more likely* that words like `banking, crises, turning, problems`, are predicted as context words
- ==Doing no more than this simple algorithm, we learn word vectors that capture similarity and meaningful directions in a word space!==



![[Pasted image 20240401124216.png]]
- Precisely, the only parameters for this model are the *word vectors!*
	- We have a "center" and "outside" vector for each word
- We use the dot product and softmax transformation to turn these into probabilities.
- This is a ==Bag of Words model==, which means that it doesn't pay attention to word order -- it doesn't matter if the context word is on the left or right -- it doesn't matter if the word is to the left or right, the probability is the same
	- This seems like a crude model of language that will offend any linguist
	- And it is! We'll move on to better ones as we move on
	- But even this crude model of language goes a long way to learn about the properties of words.

With this model, we wanted to give reasonably high probabilities to the words that *do* occur in the context of the center word... but obviously lots of different words *can* occur -- So we shouldn't have probabilities like .4 or .5 -- more like probabilities like .01, and things like that.

How do we achieve this in the Word2Vec model?
- We place words that are similar in meaning *close to eachother in this high-dimensional space* (so their dot-product is maximized)

![[Pasted image 20240401124649.png|350]]


So we mentioned doing learning -- but ==how do we *learn* good word vectors?==
- We didn't quite touch this in the last class

![[Pasted image 20240401124716.png]]
- Start out with random word vectors (small numbers near zero)
- Use a ==Gradient Descent algorithm:== We figure out the gradient of the loss/objective function with respect to the parameters
	- We make small steps in the directions that minimize the loss (in the direction opposite the gradient)
	- One of the parameters we can futz with is the ==learning rate== in our neural network


![[Pasted image 20240401124934.png]]
Above:
- Our new parameter values are the old parameter values, and then we step in the negative direction of the gradient, with the learning rate modifying our  step size.

==The above is the vanilla basics, but it's not what's actually used!==
- The problem is that above getting $\nabla_\theta J(\theta)$  is *very expensive to compute*, because it's a function of *==all==* records in the dataset!
- ==We could wait a *very long time* before even making a *single update! bad!*==

![[Pasted image 20240401125133.png]]
Instead, we use something like ==Stochastic Gradient Descent==
- We repeatedly sample windows of some size (maybe 1, for SGD, or mini-batches of some N < dataset size) and update after each one

Basically 100% of the time, we don't use Batch Gradient Descent, we instead use some variant of Stochastic Gradient Descent

![[Pasted image 20240401125629.png|450]]
Aside: The gradient update information is pretty sparse, when we only do minibatches of (eg) 50, when we have huge vocabularies!

![[Pasted image 20240401125654.png|500]]
- If you're thinking systems optimization, we might think then that we only want to update the parameters for a few words, and there have to be (and are) much more efficient ways that we can do that.
- Aside
	- Up until now, we've representing word vectors as column vectors, and that makes the most sense if you think about it as a piece of math -- but actually, in all common deep learning packages (Eg PyTorch), word vectors are actually represented as *row* vectors.

Having a single vector for word makes it very complicated
- Sometimes you'll have the same word as the context and center word
- When you're doing your calculs then, you have a messy case where just for that word, you're getting a dotproduct of x\*x term, which makes it much messier to work out, so that's why we have a simple optimization of using two vectors per word.

The actual Mikolov 2013 Word2Vec paper had two variants:
- [[Skip-Gram]] model: The one explained to us (Predict the context based on the center word)
	- "More natural, in many ways" <-- I disagree, lol
- Continuous Bag-of-Words ([[Continuous Bag of Words|CBOW]]): Predict the center from the context words.

Additional efficiency in training:
- ==Negative Sampling==
	- We were presented the naive softmax equation, which is a simple but relatively expensive training method; this isn't what they suggest using in the paper. An acronym you'll see is SGNS (Skipgram Negative Sampling)
	- This is part of homework 2!

![[Pasted image 20240401130512.png]]
Above:
- If you use the naive softmax, working out the denominator is pretty expensive
	- You ahve to interate over every word in teh vocab and work out these dot product -- if you havea  100k word vocab, you haev to do 100k dot products, a shame!
- Instead of that, we use negative sampling
	- Instead of using this softmax, we're going to train a binary logistic regression model for both
		- The true pair (center word and word in context window)
		- Several noise pairs (the center word paired with some randomly word(s))

![[Pasted image 20240401130619.png]]
Overall:
- We still want to optimize the loss for each center word
- When we want to figure out the loss for each center word... and each particular window... we take the dot product as before of hte center word and the outside word (this is the main quantity)
- But now instead of using that in softmax, we put it through the ==logistic function (sigmoid function)== -- This is a handy functino mapping any real number to a [-1, 1] interval -- if the dot product is large, the logistic will be virtually 1.
- We'd like on average the dot product between the center word and words we chose randomly to be *small*
	- The trick of how this is done... the sigmoid function is *symmetric*, so if we want the probability to be small, we can take the *negative* of the dot product (we want it to be over on the left side, a negative number)...
	- the way they're presenting things, they're maximizing the quantity
	- Let's make it closer to how we do it:

![[Pasted image 20240401130832.png]]
- Above:
	- We take the negative log likelihood of the sigmoid of the dot product
	- We use the same negated dot product through the sigmoid
	- We work out the quantity for a handful of random words
	- ...

## What else could we do?

There's this funny iterative algorithm to give us word vectors...
- if we have a lot of words in the corpus, it seems an obvious thing we could do is look at counts of how words occur with eachotehr and build a co-occurrence matrix 
- We could define some window size
![[Pasted image 20240401131208.png]]
Symmetric just like our word2vec algorithm
- The counts in the cells are just how often these things co-occur in a window size of size=1
	- eg "deep learning" occurs once, so we get a 1 in each of those cells.
- This gives us a representation of words as co-occurrence vectors
	- We take a row $i$ and say: "my representation of the word $i$ is this row vector!"

![[Pasted image 20240401134923.png]]

![[Pasted image 20240401134949.png]]
From linear algebra:
- You might have found Singular Value Decomposition
	- Which has various math properties we won't talk about here
	- retaining some optimality, producing a reduced-size number of matrices that let you recover the matrix
- The idea: Take any matrix (like our count matrix) and decompose it into *three* matrices (U, Sigma, VTranspose)
- In these matrices, some parts of them are not used...
	- So we can delete out some singular values... which means that in this product, some of U and some of V is not used.. as a result we get lower dimensional representations for our words which still do as-good-as-possible a job

![[Pasted image 20240401135502.png]]


[[GloVe|Global Vectors for Word Representation]] (GLoVe)

![[Pasted image 20240401135715.png]]
- The linear algebra methods had advantages for fast trainin
	- There had been efforts to capture word similarity with them, but they hadn't worked that well
- The neural models seemed like... that perhaps you're inefficiently using statistics vs a co-occurrence matrix
	- But it's easy to scale to a very large corpus by trading time for space
	- The Neural methods seemed to just work better for people

What we realized we needed:
- If you want to have vector subtractions and additions *work* for analogies, etc. -- the property that you want is for *meaning components* (something like going from male to female, king to queen, truck to driver), those components should be represented as ratios of co-occurrence probabilities!
![[Pasted image 20240401135807.png]]
If we're trying to get the spectrum from *solid to gas*, as in physics
- You can get the solid part by asking: "Does the word co-occur with Ice"
	- The problem is that the word *water* occurs a lot with ice :O 

In contrast
![[Pasted image 20240401135846.png]]
If you look at words co-occuring with steam:
- Again it doesn't quite


to Get what we want, look at the *==ratio==* of these co-occurrence probabilities!
![[Pasted image 20240401135915.png]]
- If you count them up in a large corpus, you might get:

![[Pasted image 20240401140003.png|400]]

How can we capture these ratios of co-occurence probabilites as linear meaning components so that in our word vector space we can just add and subtract these components

IT seems like the way we can achieve that is by building a log-bilinear model so that the dot product between word vectors estimates the log probability of conditional occurrence:

![[Pasted image 20240401140344.png]]
![[Pasted image 20240401140409.png]]
The GLoVe model wanted to unify the thinking between the linear algebra and neural models
- Neural, but Calculated on top of a co-occurence matrix count
- Explicit loss function
	- We want the dot product to be similar to the log of the co-occurrence
	- We want to not have very common words dominate; we capped the effective high word counts using this $f$ function shown here
	- Optimize this $J$ function directly on the co-occurrence count matrix

This algorithm worked very well!


![[Pasted image 20240401140557.png]]

==Intrinstic Evaluation==
- Evaluating directly on the specific or intermediate subtask that you've been working on
- Normally fast to compute, help you understand the component you're working on
- Simply trying to optimize this component may or may not have a good effect on the overall system you're building

==Extrinsic Evaluation==
- Take some *real task of interest to human beings* (web search, translation) and say that your goal is to improve performance on *that* task
- A real proof that this is doing something useful.


![[Pasted image 20240401142933.png]]
So many meanings for a word: Should we have a separate vector for each?

![[Pasted image 20240401143523.png]]
These people tried! You train a bunch of vectors for a word and average them
It sorta worked


























- 




