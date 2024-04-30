#article 
Link: https://thegradient.pub/understanding-evaluation-metrics-for-language-models/
( This is an old article, but it's mostly about metrics like perplexity, cross-entropy, etc. Evergreen topics )

-----

![[Pasted image 20240221205802.png]]

Traditionally, language model performance is measured by [[Perplexity]], [[Cross-Entropy]], and [[Bits Per Character]] (BPC). 

As language models are increasingly used as pre-trained models for other NLP-related tasks, they're often evaluated based on how well they perform on specific downstream tasks.

The [[GLUE]] benchmark score is one example of a broader, multi-task evaluation for language models.

Counterintuitively, having *more metrics* makes it *more difficult* to compare language models -- especially as many of these benchmarks are unreliable (leaked answers, etc).

==One of Chip's favorite interview questions is to explain [[Perplexity]] or the difference between [[Cross-Entropy]] and [[Bits Per Character]]==

We aim to minimize perplexity, but is it possible to get a perplexity of zero? How good is a *good* perplexity, what's the optimal value?

In metrics like accuracy where it's a certainty that 90% accuracy is better than 60% accuracy on the same test set, arguing that a model's perplexity is smaller than that of another doesn't signify a great deal unless we know how the text is pre-processed, the vocabulary size, the context length, etc.
- Perplexity for a LM at character-level might be much smaller than the perplexity of another model at word-level; it doesn't mean that the character-level LM is better than that of the world-level one.

# Understanding Perplexity, Bits-per-Character, and Cross-Entropy

- Consider an arbitrary language model $L$. The model assigns probabilities to sequences of arbitrary symbols, such that the more likely some sequence $(w_1, w_2, ..., w_n)$ is to *actually exist* in the language, the higher the probability our model will give it.
	- Symbols in this context could be characters, words, or sub-words.
- Most language models estimate this probability as a product of each symbol's probability, given its preceding symbols.

$P(w_1, w_2, ..., w_n) = p(w_1)p(w_2|w_1)p(w_3|w_1,w_2) ... p(w_n|w_1, ... w_n-1)$
$=\prod_{i=1}^{n} p(w_i|w_1, .... , w_{i-1})$ 

![[Pasted image 20240429170900.png]]

Alternatively, some other language models estimate the probability of each symbol given its *neighboring* symbols -- this is also known as the *Cloze* task.


# Entropy
- The goal of any language is to oncvery information.
- To measure the average amount of information conveyed in a message, we use a metric called "[[Entropy]]". 

> "*The entropy is a statistical parameter which estimates, in a certain sense, how much information is produced on the average for each letter of a text in the language. If the language is translated into binary digits (0, 1) in the most efficient way, the entropy is the average number of binary digits required per letter of the original language.*"

We know that for 8-bit ASCII, each character is composed of 8 bits; but this isn't the most efficient way to represent letters in English language, since all letters are represented using the same number of bits, regardless of how common they are! (We should probably use fewer bits for *o* or *e*, in exchange for more bits for rare letters like *z* or *q*). Under a more efficient encoding we might use less than 8 bits, on average.

Shannon approximates any languages entropy $H$ through a function $F_N$ which means the amount of information (in other words, entropy), extending over N adjacent letters of text.

# Cross Entropy
- Owing to the fact that there's an infinite amount of text in the language $L$, the true distribution of the language is unknown!
- A language model aims to learn, from sample text, a distribution $Q$ close to the empirical distribution $P$ of the language.
- To measure the "closeness" of the two distributions, metrics like [[Cross-Entropy]] are often used.

Mathematically, the cross-entropy of $Q$ with respect to $P$ is defined as follows:

$H(P,Q) = E_P[-logQ]$ 

When $P$ and $Q$ are discrete, this becomes:

![[Pasted image 20240429171749.png]]

Where $D_KL(P||Q)$ is the [[Kullback-Leibler Divergence]] of Q from P.
This term is also known as the relative entropy of P with respect to Q.

### ==TAKEAWAY==
==Therefore==, the [[Cross-Entropy]] of Q with respect to P is the sum of the following two values:
1. The average number of bits needed to encode any possible outcome of P using the code optimized for P (which is H(P) - the [[Entropy]] of P)
2. The number of EXTRA bits required to encode any possible outcome of P using the code optimized for Q. (D_KL(P||Q), the [[Kullback-Leibler Divergence|KL-Divergence]] of Q from P).

==It should be noted that since the empirical entropy H(P) is unoptimizable, when we train a language model with the objective of minimizing the cross-entropy loss, the true objective is to minimize the KL divergence of the distribution==, which was learned by our language model from the empirical distribution of the language.


# Perplexity

*A measurement of how well a probability distribution or probability model predicts a sample.*

Intuitively, [[Perplexity]] can be measured as a measure of uncertainty.
- Consider an LM with an entropy of three bits, in which each bit can encode two possible outcomes of equal probability.
	- This means that the LM has to choose among $2^3 = 8$ possible options.Thus, we can argue that the language model has a perplexity of 8.

Mathematically, the ==perplexity== of an LM is defined as:

$PPL(P, Q) = 2^{H(P,Q)}$ 

Where again, $H(P,Q)$ is the cross entropy of approximated distribution Q with respect to the reference distribution P.
- ((This seems to make sense that ))


# Bits-per-character and Bits-per-word
- [[Bits-per-Character]] (BPC) is another metric often reported for recent language models.
- It measures exactly the quantity that it's named after: ==The average number of bits to encode one character==. 
- This leads to revisiting Shannon's explanation of entropy of a language:
> "If the language is translated into binary digits (0 or 1) in the most efficient way, the entropy is the average number of binary digits required per letter of the original language."
- By *this* definition, *entropy* is the average number of BPC!
- While entropy and cross entropy are defined using log base 2 (with "bit" as the popular unit), popular ML frameworks (eg PyTorch) implement Cross Entropy loss using the natural log, because it's faster to compute.
	- In theory, the log base doesn't matter, because the difference is a fixed scale.

Keep in mind that BPC is specific to character-level language models. When we have *word-level* language models, the quantity is called [[Bits-per-Word]] (BPW) -- the average number of bits required to encode a word. 

# Reasoning about Entropy as a Metric
- Since we can convert from perplexity to cross entropy, and vice versa, let's examine only cross entropy.
- For many metrics used for ML models, we generally know their bounds. For example, the best possible value for accuracy is 100%, while that number is 0 for word-error-rate and mean-squared-error.
- When we argue that a LM has a cross-entropy loss of 7, we don't know how far this is from the best possible result, if we don't know what the best possible result should be!
	- Not knowing what we're aiming for can make it challenging in regards to deciding the amount resources to invest in hopes of improving the model.


# Mathematical bounds
- It's imperative to reflect on what we know, mathematically, about Cross Entropy

==Firstly==: We've seen that the smallest possible Cross Entropy for any distribution is zero, but the entropy of a language can only be zero if that language has exactly one symbol, which wouldn't be very useful.
- If a language has *two* characters that appear with equal probability, a binary system for instance, its entropy would be:

$H(P) = -0.5 * log(0.5) - 0.5*log(0.5) = 1$

==Secondly==, we know that the entropy of a probability distribution is maximized when the distribution is uniform!
- This alludes to the fact that for all the languages that share the same set of symbols (vocabulary), the language that has the maximal entropy is the one in which all the symbols appear with equal probability.

![[Pasted image 20240429174107.png]]

According to an average 20-year-old American knowing 42,000 words, their word-level entropy will be *at most* log(42,000) = 15.3581 (if the distribution was considered to be uniform)

==Thirdly==, we understand that the cross entropy loss of a language model will be at least the empirical entropy of the text that the language model is trained on.
- If the underlying language has the empirical entropy of 7, the cross entropy loss will be at least 7.

![[Pasted image 20240429174256.png]]

==Lastly,== remember that according to Shannon's definition, entropy is $F_N$ as $N$ approaches infinity. We will show that as $N$ increases, the $F_N$ value decreases.
- This makes senses, since the longer the previous sequence, the less confused the model will be when predicting the next symbol.
- This means that ==with an infinite amount of text, LMs that use longer context length in general SHOULD have lower cross-entropy values==, compared to those with shorter context lengths! So ==when we report either perplexity of cross-entropy for an LM, we should specify the context length!==

![[Pasted image 20240429174433.png]]


# Estimated Bounds
- Since the year 1948, when the notion of information entropy was introduced, estimating the entropy of the written English language has been popular musing among linguists, information theorists, and computer scientists.
- ((Omitted some historical attempts at doing this))


(Skipped the last 10%)