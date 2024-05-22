Video: https://www.youtube.com/watch?v=qESAY4AE2IQ&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=3

----

# Topic: Neural Language Models

![[Pasted image 20240521210315.png]]
Recap: Language Models allow us to estimate the probability of a piece of text by factorizing that probability into the product of conditional probabilities preceding tokens, giving *their* preceding tokens.

For our [[N-Gram]] models, we used the Markov Assumption, where the next state depends only on the previous state (where we extended previous state to include one token (Unigram), 2 tokens (Bi-Gram), ... all the way up to N-grams).

![[Pasted image 20240521210529.png]]
Above:
- We count up the number of occurrences of "students opened their $w_j$"
- The denominator is crucial because it normalizes the count, ensuring that the probability is conditioned on the prefix. It effectively answers the question: out of all the times the prefix "students opened their" occurs, how many times does it continue with the specific word $w_j$

If we were using a *Bigram* model, we would ignore "students opened" and just look at P(w_j|their)

There's a reason few problems with using N-Gram models:
1. Sparsity problem; Let's say that "Students opened their laptop" never occurred in the training set? Then it'll be 0 in our numerator from the above picture. But what if it's a reasonable sentence? This can be somewhat ameliorated using [[Label Smoothing]]. There are problems with this too -- if you don't observe instances of "Students opened their laptop," you really have no idea whether the word is being used correctly or not; there's only so much you can do if you know nothing about a word or the context in which it occurs. This is also true of word types that just don't occur in the vocabulary. A "solution" there is to use an UNK token.
2. They're not great at modeling long-distance dependencies in language if the dependency is long than N tokens away (in an N-gram).
3. The storage cost of an N-gram model! For a bi-gram model, we have these big tables where the rows are the prefix, and the columns are the next word in the sentence. Every time we want to add more context (eg by going from a bigram to a trigram), that table size increases exponentially!
4. We treat all words/prefixes independently of eachother! "Students opened their {blank}" and "pupils opened their {blank}" have nothing to do with eachother, even though they're very similar, semantically!

On the plus side, N-gram models are very interpretable.

For *neural* language models, to combat issues like the ones above, we almost necessarily *have* to trade off a measure of interpretability.

![[Pasted image 20240521213240.png]]
N-gram models rely on the ==Bag of Words Assumption==, meaning that every single word type in the vocabulary and every single prefix are treated as one-hot vectors, where the length of our vector is the size of the vocabulary. Only one element o the vector is 1, and the rest are 0. If you think about an n-gram model, this makes sense; *Movies* and *Film* are completely independent of eachother, so maybe they should just be dissimilar in this way! But wait... ==all of these on-hot vectors are equally (dis)similar to eachother,== even when words like *Movies* and *Film* are actually kind of similar to eachother!

==What we want is a representation space in which words, phrases, sentences, etc. that are semantically similar also have similar representations to eachother!==

![[Pasted image 20240521214147.png]]
We don't want to use this bag of words assumption, treating every word as its own independent entity.


![[Pasted image 20240521214236.png]]
The common solution is to use lower-dimensional, dense token [[Embedding]]s, rather than sparse, high-dimensional one-hot vectors.
These dense vectors encode some notion of semanticity, such that there are vectors that have "meaning".

The dimensionality of this vector is a hyper-parameter that can be experimented with in a trial-and-error manner.

![[Pasted image 20240521220205.png]]
[[Softmax]] layer: 
- Convert a vector representation into a vector that represents a probability distribution (over the entire vocabulary (in the sense that every element is positive and sums to one))







