From 2012's *Japanese and Korean Voice Search (Schuster et al.)*

WordPiece is a *subword tokenization algorithm* used in [[Bidirectional Encoder Representations from Transformers|BERT]], [[DistilBERT]], and [[ELECTRA]], among others.

It's *very* similar to [[Byte-Pair Encoding|BPE]].

WordPiece first initializes the vocabulary to include every character present in the training data, and progressively learns a given number of merge rules.
- In contrast to BPE, WordPiece doesn't choose the *most frequent symbol pair,* but instead ==chooses to merge the pair of tokens that *maximizes the likelihood of the training data, once added to the vocabulary==.*
- Maximizing the likelihood of the training data is equivalent to finding the symbol pair whose probability divided by the probabilities of its first symbol, followed by its second symbol, is greatest among all symbol pairs.
	- e.g. "u" and "g" would only be merged if the probability of "ug" divided by "u", "g" would have been greater than any other symbol pair.
	- Intuitively WordPiece is different from BPE in that it evaluates what it *loses* by merging two symbols to ensure that it's worth it. 



