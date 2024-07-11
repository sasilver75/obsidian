#lecture 
Link: https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4

Agenda:
- Course
- Human Language and Word Meaning
- Word2Vec introduction
- Word2Vec objective function gradients
- Optimization basics
- Looking at word vectors

-------

We want to get a sense of how impressive word embeddings are: representing words as (perhaps dense) vectors of numbers, which flies in the face of many of the traditional NLP examples

Goals of the course:
- Effective understanding of the modern methods for DL as applied to NLP
	- Basics, then key methods used in NLP: Recurrent NNs, Attention, Transformers
- A big picture understanding of human languages and why they're difficult to understand and produce
- An understanding/ability to build systems in PyTorch re: major problems in NLP
	- Word meaning, dependency parsing, machine translation, question answering

Diving into human language
- It wasn't claws or speed that made humans dominant, it was the development of languages (1My ago) and of writing (5ky ago) that made it so.

![[Pasted image 20240331233415.png]]
We call them distributed representations because the "meaning" of banking is spread over the (eg) 300 dimensions in the high-dimensional vector space!

We can visualize these high-dimensional vector spaces in 2 or 3 dimensions and see that "related" words are often in clusters with eachother, and that directionality has a semantic meaning.
(But note that when you crush 300 dimensions down into 2, you're going to lose a lot of information and have things placed together that don't really make sense together, along with the ones that do)

Contentful verbs
Copular verbs
# [[Word2Vec]]
Introduced in 2013 (Mikolov et al) as a framework for learning word vectors

`Word2Vec Idea`:
- We have a large *corpus* of text
- Every word in a fixed-size vocabulary (eg 4,000) is represented by a *vector*
- Go through each position *t* in the text, which has a center word `c` and a context/"outside" words `o`
- Use the similarity of the word vectors for `c` and `o` to calculate the probability of `o`, given `c` (or vice versa)
	- eg: "Given the surrounding words, what's the probability of the center word?" ([[Continuous Bag of Words]] (CBOW)) or "Given the center word, what's the probability of the outside words?" ([[Skip-Gram]])
- Keep adjusting the word vectors to maximize this probability

Below: Using the ==Skip-Gram== formulation (Which is harder/slower to train but represents rare words better)
 ![[Pasted image 20240331235410.png]]
Above:
- Given a piece of text
- Pick a center word
- Say: "If we have a model of predicting *context words*, given the center word
	- This model, we'll come to in a minute
- Get a probability that it gives...to the words that actually occurred in the context of our center word
- Given the loss, determine that the probability should be higher; how do we change our word vectors to raise these probabilities?
	- We do some calcualtions, and make updates to vectors (maybe later)
- Go on to the "next" center word in the sentence

What are we doing for figuring out the probability of `o` given `c`?
What's our objective function?

![[Pasted image 20240331235544.png]]
Above:
- For each position in our body of text, we want to predict context words in a window of fixed size (j +- m). We want to become good at doing this, so we want to provide high probaility to words that appear in the context of our center word!
- We have a formal likelihood of the data likelihood as to how good we are at predicting words...
- That likelihood is defined *in terms of our word vectors*, the parameters of the model
	- using each word as the center word, the product of each word in the window around it... of predicting that context word, given the center word.

So we want to learn to adjust our parameters (our word vectors) in order to maximize the likelihood of the context that we see around center words.

(Second half of above slide): Following standard practice, we slightly fiddle this, because rather than dealing with products, it's easier to deal with sums, so we do the ==log likelihood==, because $log(ab) = log(a) + log(b)$
- We also work with the *average* log likelihood ($1/T$) -- we like to (for no reason) *minimize* our objective function rather than maximize it, so we stick a minus sign in there.
- By minimizing that $J(\theta)$ function, we end up maximizing our predictive accuracy.

But how do we calculate the probability of a word in the context, given the center word?

# Word2Vec: Objective Function

- We want to minimize the objective function:
![[Pasted image 20240401000100.png|400]]

Question: How do we calculate our {probability of context word, given center word and theta parameters?}
![[Pasted image 20240401000122.png]]
Answer: We will use *two vectors* for every word `w`:
1. $v_w$ when `w` is a center word
2. $u_w$ when `w` is a context word

Then, for a center word $c$ and a context word $o$ :

![[Pasted image 20240401000248.png]])
Above (Probability of "outside" given the "center" word):
- We have vector representations for each word... and we work out the probability simply in terms of the word vectors.
- We're going to give to each word ==two== word vectors!
	- One for when it's used as the center word
	- Another for when it's used as the context word
	- ==REASON==: This just simplifies the math and optimization -- it's a little ugly, but makes building word vectors much easier!
- Once we have these word vectors, then we can give the probability of context/outside word, given a center word

Let's pull this equation apart a bit more:
![[Pasted image 20240401000719.png]]
Above:
- Given words c and o, look up the *appropriate* vector representations of each word, $u_o$ and $v_c$ 
	- (Remember that each word has a "center" and "context" representation)
- Take the dot product of both measures
	- A natural measure of similarity between words
- We're sort of really doing nothing more than saying: "We want to represent dot products for word similarity, so let's do the dumbest thing we know how to make this into a probability distribution." We can't have negative probabilities, and they should sum to one
	- We exponentiate, so we know things are positive
	- We normalize over the entire vocabulary ((This seems expensive!))

The [[Softmax]] function
- Takes anything into a range of (0,1)
	- It's sort of like a MAX (because we exponentiate, it really emphasizes the big contents in the different dimensions of calculating similarity)... most of the probability goes to the biggest things.
	- It's "soft" because it doesn't do the maxing absolutely; it gives *some *probabilitiy to things that are similar


So how do we get our word vectors?
- Fiddle our word vectors such that we minimize our loss, maximizing the probability of the context words, given the center word
![[Pasted image 20240401002705.png]]
Theta represents all of our model parameters in one very long vector
- For each word, we have two vectors 
	- Context vector
	- Center vector
- And each is a d-dimensional vector -- we end up with something $2dV$ long (where V is the vocab size) 
	- (This is a one-dimensional vector, not a matrix! The picture makes that only kinda clear)

With our objective function, can we work out derivatives.. and work out the gradient that we can walk downhill and minimize loss. 
We can progressively walk downhill, adjusting our vector $\theta$ and improving our model.

At this point, we want to show a little more about how we can do that (some math)


















