#lecture 
Link: https://www.youtube.com/watch?v=q6KvtdJzXlQ&list=PLWnsVgP6CzaelCF_jmn5HrpOXzRAPNjWj&index=2
Lecturer: Mohit Iyyer

---

# Topic: N-Gram Langauge Models

We're going to discuss simple n-gram language models
- Counting up the occurrences of particular words, and normalizing these counts by a certain value.

Let's say we want to train a model for [[Sentiment Analysis]]
- In the past, we might have simply trained a supervised model in labeled sentiment examples (eg review text of products, score pairs from IMDB, etc.)

We've discussed the advantages of Transfer Learning, though
- Where we have a pretrained model trained (perhaps) by a self-supervised training objective (eg SSL language modeling, with next-token prediction).
- We can then specialize this generalized model to the specific downstream task that we care about, and take advantage

This was popularized by things like [[ULMFiT]] (Jan '18), [[Bidirectional Encoder Representations from Transformers|BERT]] (Oct '18)

Note: A language model like ChatGPT isn't *just* trained to predict the next word; there are other steps after that like [[Instruction-Tuning]] and [[Reinforcement Learning from Human Feedback|RLHF]] that align the outputs of our model with what humans actually want to be seeing.

---

But let's come down to earth with [[N-Gram]] models, which don't have that much to do with modern language models, but *do* introduce us to some of the tasks we care about, and architecture-agnostic things like evaluation.

What is language modeling?
- We've talked about predicting the next word, given context
- This is the same task as assigning the probability of any given piece of text!
- ==A language model allows us to get the probability of any arbitrary piece of text.==
	- Say you're doing [[Machine Translation]] and we're translating an english sentence into Chinese. Let's say we have two candidate translations, and we know that candidate translation A is much less likely than candidate translation B -- that would be useful information, right?

![[Pasted image 20240521192443.png]]
- We could take every sequence of N words, and count up how frequently some specific sequence of N words appeared in the dataset, and divide over the total sequence of N words.
	- But there's a sparsity problem -- if N=20, we might not EVER observe any sequence of words that has the words in the order as our current sentence.

As we'll see, there are some tricks to get around this sparsity problem, and make our language model a lot better.

Let's say our words 1...5 are: P(its, water, is, so, transparent, that)
and we want to compute the probability of this 5 word sequence

When we use the Chain Rule of Probability, we can factorize the joint probability `P(A,B,C,D)` into these joint probabilities! `P(A)P(B|A)P(C|A,B)P(D|A,B,C) ....`

![[Pasted image 20240521193154.png]]

But how do we estimate the probability of P(its)? Maybe we take a huge collection of text, count the number of times that every word occurs in the text, and normalize by the length of the text? This is called a ==Unigram== model.

The ==Unigram== probability is just the freuqency of a single word in a body of text.

What about P(water | its)?
- We count up all of the occurrences of "its water", and divide by the total number of "its {anything}".
	- ((WAIT, I don't think this is right? See the first picture in the second lecture; it seems like we normalize by the count of the PREFIX?))
	- GPT-4: The denominator is crucial because it normalizes the count, ensuring that the probability is conditioned on the prefix. It effectively answers the question: out of all the times the prefix "students opened their" occurs, how many times does it continue with the specific word w_j

These all assume that we have access to some giant collection/corpus of text, and we're able to count up the frequency of different words/phrases in that corpus. We're actually forming the [[Maximum Likelihood]] estimate of these various probabilities.

This is essentially how N-Gram models work.

So if we can compute all of these individual probabilites
`P(its)`, `P(water|its)`, `P(is|its water)`, ...
We can then multiply them together to get the probability of a sequence
`P(its water is so transparent)`

When we get to neural language models, we're not jus taking into account taht these words occurred, but also their semantics
Maybe "its water is so transparent" is similar to "its fluid is so transparent"

Q: GPT-4 is an N-Gram model, right?
A: No, it's a neural language model! It's a different type that we'll start covering next week.

Terminology
$P(w_i|w_1w_2...w_{i-1})$ 
Above, the $w_1w_2...w{i-1}$ is called the "prefix"

In a unigram model, we're going to consider absolutely NO context! 
The prefix just IS NOT THERE, and we just count up the frequency of the word's occurrence in the corpus, regardless of what comes before it.

![[Pasted image 20240521193958.png]]
Above: [[Markov Assumption]]: Next state depends only on the previous phrase
(The second example in a ==Bi-Gram== assumption, which isn't the usual markov assumption, IMO) 

What's a downside of making these Markov assumptions?
- We lose "long-term" context! We don't know what the subject of a phrase is, what's come before, etc. 

As we increase the prefix...
- Trigram (considering three previous three words)

.... Our models get more performant! 😮

If we have probability of a word $y$ given all the words before it, how can we use this model to generate some text?
- We can sample some next word from the probability distribution over next words (using whatever strategy we want; say, using the highest-probability word, in greedy decoding)
- Repeat the forward pass and get a new distribution over next words
But wait, how do we know when to stop generating?
- In both N-Gram models and in models like ChatGPT, there's a special EOS token (end of sequence) attached to the end of every document/text in the training data. This means that the model then has some probability of producing an EOS token to terminate its generation.

What do we mean when we say that we sample from a distribution?

![[Pasted image 20240521200335.png]]

All words in our vocabulary are assigned some probability. We select some word from the vocabulary based on this probability distribution.

![[Pasted image 20240521200516.png]]
This is a cool example of using 1-2-3-4-gram models to approximate shakespeare! See that they improve in quality.

So increasing the length of this N-gram prefix is good, but if we increase it too much, we don't have enough occurrences of some prefix to estimate the distribution well. So N-gram models are a balance: You want as much context as you can get, without running into a problem where you have (eg) 0-few occurrences of your prefix.

![[Pasted image 20240521202114.png]]
Recall: We count the empirical frequenies of these words/phrase in the dataset that we're given.
The Maximum likelihood estimate of the bigram model of the dogwalkers.com is probably going to give us a higher probability of "I walked the dog" than if you estimated using twitter data; The data that you use really defines the distribution of language that you learn.
- As a result, people like big, diverse corpuses of text to estimate these models. We avoid both bias and sparsity.

![[Pasted image 20240521202411.png]]

Important terminology:
- ==Word Type==: Refers to any unique word in our vocabulary
- ==Word Token==: An occurrence of a word type in a dataset

![[Pasted image 20240521202853.png]]
When we multiply a bunch of small probabilities, we get underflow!

To solve this, we deal with ==log probabilities instead of raw probabilities==!

![[Pasted image 20240521202926.png]]
The raw probability of this is a vanishingly small number
The log probability is -14.33! This is easier to deal with/interpret/optimize than the raw probability.

The Jan 2024 ==Infinigram== [paper](https://arxiv.org/abs/2401.17377) generalized N-grams to... much longer prefixes!
![[Pasted image 20240521203429.png]]
If you actually have occurrences of a very long string, it might be the case that you can *exactly* occur what happens next.

Evaluation of models
- We evaluate the performance of our language model on some held-out data -- a good language model should assign "correct" probabilities to heldout text too! 
	- We may have either overfit or underfit.

![[Pasted image 20240521204100.png]]

![[Pasted image 20240521204130.png]]
The equation for perplexity is basically just exponentiating the negative log likelihood (per-token likelihood) on some held-out text corpus.

![[Pasted image 20240521204202.png]]
If we think about Perplexity; assume we were given a prefix "I want the" and we want to know how many different paths/next words the model thinks are equally likely, given the prefix.
- If the model had NO IDEA what cames after, then maybe it assigns roughly equal probability to aLL words in the vocabulary, meaning the model is very confused -- it's highly *perplexed*
- If the model is very good, it know that *dog*, *streets*, etc. are likely -- some small number of words. ==So you can view this as a sort of branching factor -- given a prefix, how many words does the model think are reasonable given the prefix?==

Above: What's the perplexity of a sentence of 10 random digits?
- 10!
- The probability of any sequence is (1/10)^N
- Perplexity then exponentiates this by -1/N
- This result in 10; it means that 10 numbers are equally likely given any prefix, given that this is a totally random model. 

In practice, we use negative log likelihood as our training signal as the thing we're trying to maximize

![[Pasted image 20240521204521.png]]

Se if we have the log probability of word i given previous context, we sum these log probabilities over the entire sequence (because when we use logs, we sum, rather than multiply), and then normalize by the number of words in the sequence; we're getting the average token-level probability of the next word under the language model.


If we had an absolutely perfect LM that gave us a P(1) for the correct next word in the sequence, this perplexity would be 1 (because the model thinks there's only 1 path that's likely). So 1 is the absolute minimum perplexity we could get.

What if our LM considers all words as equally liekly, in a vocabulary of 100,000? It's 100,000! (Garbage)
So the perplexity is between 1...VocabularySize, hopefully as close to 1 as possible.













