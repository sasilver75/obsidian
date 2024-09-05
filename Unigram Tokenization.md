References:
- [ARTICLE HuggingFace NLP Course Unigram Tokenization](https://huggingface.co/learn/nlp-course/chapter6/7?fw=pt)
- 

The unigram algorithm is often used in [[SentencePiece]], which is the tokenization algorithm used in models like AlBERT, [[T5]], mBART, Big Bird, and [[XLNet]].

Compared to [[Byte-Pair Encoding|BPE]] and [[WordPiece]], Unigram works in the *other direction,* starting with a ***big vocabulary*** and ***removes tokens*** from it until it reaches the desired vocabulary size.

There are several options to use to build that base vocabulary:
- We can take the most common substrings in the pre-tokenized words, for instance, or apply BPE on the initial corpus with a large vocabulary size.

At each step of training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. For each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was removed, and looks for the symbols that would increase it the least.
- These symbols have a lower effect on the overall loss over the corpus, so in a sense they are "less needed" and are the best candidates for removal.
- This is all a very costly operation, don't just remove the *single* symbol associated with the lowest loss increase, but the `p` (being a hyperparameter, usually 10-20%).
- This process is repeated until the vocabulary has reached the desired size.
- NOTE that we never remove the base characters, to make sure any word can be tokenized.

Regarding the loss used during training
- At any given stage, this loss is computed by tokenizing every word in the corpus, using the current vocabulary and the Unigram model determined by the frequencies of each token in the corpus.
- Each word in the corpus has a score, and the loss is the negative log-likelihood of those scores -- that is, the sum for all the words in the corpus of all the `-log(P(word))`.

