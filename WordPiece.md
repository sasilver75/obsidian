References
- [VIDEO: HuggingFace NLP Course Wordpiece Tokenization](https://www.youtube.com/watch?v=qpv6ms_t_1A)

From 2012's *Japanese and Korean Voice Search (Schuster et al.)*

WordPiece is a *subword tokenization algorithm* used in [[BERT|BERT]], [[DistilBERT]], and [[ELECTRA]], among others.

It's *very* similar to [[Byte-Pair Encoding|BPE]].

WordPiece first initializes the vocabulary to include every character present in the training data, and progressively learns a given number of merge rules.
- In contrast to BPE, WordPiece doesn't choose the *most frequent symbol pair,* but instead ==chooses to merge the pair of tokens that *maximizes the likelihood of the training data, once added to the vocabulary==.*
- Maximizing the likelihood of the training data is equivalent to finding the symbol pair whose probability divided by the probabilities of its first symbol, followed by its second symbol, is greatest among all symbol pairs.
	- e.g. "u" and "g" would only be merged if the probability of "ug" divided by "u", "g" would have been greater than any other symbol pair.
	- Intuitively WordPiece is different from BPE in that it evaluates what it *loses* by merging two symbols to ensure that it's worth it. 



![[Pasted image 20240623225032.png]]
Above: The three main subword tokenization algorithms, [[Byte-Pair Encoding|BPE]], [[WordPiece]], and Unigram

----
From HuggingFace NLP Course

- Like [[Byte-Pair Encoding|BPE]], WordPiece starts from a small vocabulary including the special tokens used by the model and the initial alphabet. It identifies subwords by adding a prefix (like ## for BERT), so each word is initially split by adding that prefix to all characters inside the word.
	- So "word" is split into `w ##o ##r ##d`
- Thus, the initial alphabet contains all characters present at the *beginnings* of word, and the characters present *inside words* preceded by the WordPiece prefix.
- ==Like BPE, WordPiece learns merge rules -- but the main difference is the way the pair to merged is selected.==
	- WordPiece computes a score for each pair, using the following formula:
	- ![[Pasted image 20240623232407.png]]
	- By dividing the frequency of the pair by the product of frequencies from the parts, the algorithm prioritizes merging pairs where the individual parts are less frequent in the vocabulary.
		- So it won't necessarily merge `("un", "#able")`, even though the pair might occur very frequency in the vocabulary, because the two tokens `"un"` and `"#able"` will likely appear in a lot of *other words* -- they're seen with high frequency! In contrast, tokens like `("hu", "#gging")` perhaps will be merged faster, since "hugging" likely appears often in the corpus, but `"hu"` and `"##gging"` are less likely to be frequent individually.
- Test-time Tokenization in WordPiece differs from BPE, because WordPiece only saves the final vocabulary, and not the merge rules that are learned!
	- Starting from the word to tokenize, WordPiece finds the *longest subword that's in the vocabulary, then splits on it.* So if we're tokenizing "hugs" and the longest subword starting from the beginning that is inside the vocabulary is "hug", we might split there and get  `("hug", "##s")`. 
	- In contrast, with [[Byte-Pair Encoding|BPE]], we would have applied the merges learned in order, and perhaps tokenized this as `("hu", "##gs")`, so the encoding is different.
	- When the tokenization gets to a stage where it's *not possible* to find a subword in the vocabulary, the ==whole word== is tokenized as *unknown*/UNK.
		- So even for `"bum"` if `"b"` and `"##u"` are in the vocabulary, if `"##m"` is not in the vocabulary, the resulting tokenization for `"bum"` will just be `[UNK]`, NOT `("b", "##u", [UNK])`. This is another difference with [[Byte-Pair Encoding|BPE]], where we could classify only the individual characters not in the vocabulary as unknown.