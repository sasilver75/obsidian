---
aliases:
  - BPE
---
References:
- [VIDEO: HuggingFace NLP Course BPE](https://youtu.be/HEikzVL-lZU?si=VESfTlnUoacLgEJX)

Introduced in 2016's *Neural Machine Translation of Rare Words with Subword Units (Sennrich et al.)*

Used by [[GPT-2]], [[GPT-3]], [[GPT-4]], [[RoBERTa]], and more.
It's *very* similar to [[WordPiece]]

BPE creates a base vocabulary consisting of all symbols that occur in our set of unique words, and learns merge rules to form new symbols from two symbols of the base vocabulary. It does so repeatedly until the vocabulary has attained the desired vocabulary size (which is hyperparameter). 

1.  Start with a vocabulary containing only *characters* and an "End of Word" symbol
2. Using a corpus of text, find the *most common adjacent characters* (say, a+b) and add the concatenation as a subword (ab)
3. Replace instances of the character pair with the new subword; repeat until desired vocabulary size is reached.

What you end up with is a vocabulary of very commonly-occurring substrings, which can be used to build up words.

GPT-2 used *bytes* as the base vocabulary (Byte-level BPE), which was a clever trick to force the base vocabulary to be size 256 while ensuring that every base character is included in the vocabulary. GPT2's tokenizer can tokenize every text without the need for the `<unk>` symbol.

----

Given an initial set of words/frequencies in a dataset:

`("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)`

Consequently, our base vocabulary is 

`["b", "g", "h", "n", "p", "s", "u"]`

We can then split all words into symbols of the base vocabulary:

`("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)`

We can then count the frequency of each possible symbol pair, and pick the symbol pair that occurs the most frequently. Above, "hu" is present 15 times, but the most frequent symbol pair is "ug", occurring 20 times. So the first merge we make is to group all "u","g"s into "ug". 

Our words are now:
`("h" "ug", 10), ("p" "ug", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "ug" "s", 5)`

BPE then continues to identify the next most common symbol pair ("u", "n", becomes "un")

`("hug", 10), ("p" "ug", 5), ("p" "un", 12), ("b" "un", 4), ("hug" "s", 5)`

And on and on.
- Now, "bug" would be tokenized to `["b", "ug"]`, but "mug" would be tokenized as `["<unk>", "ug"]`, since `m` wasn't a symbol in the base vocabulary. In general, this wouldn't happen to single letters, but might happen for special characters like emojis.

---

![[Pasted image 20240623225042.png]]
Above: The three main subword tokenization algorithms, [[Byte-Pair Encoding|BPE]], [[WordPiece]], and Unigram

From Amherst Course:
![[Pasted image 20240523163656.png]]
How do we get a subword encoded? The BPE algorithm *start* with a character-level encoding, and forms words iteratively. 
- Given our toy dataset of "hug, pug, pun, bun, hugs" (and associated counts), we start with our base vocabulary of just each of the characters used in the dataset.

We want to form subwords that are combinations of these characters, but we want to do so intelligently based on usage in the corpus!
- We want to *merge* characters together that occur together very frequently in our training data

![[Pasted image 20240523163817.png]]
We take every *pair* of characters in our dataset and check how often they co-occur.
- ==We observe that *ug* seems like a character pair that occurs quite often, so we *add* ug to our vocabulary, and re-tokenize our dataset using *ug*. When doing this, we prefer to tokenization that results in the fewest tokens! So "hug" becomes "h"+"ug" rather than "h"+"u"+"g".==

We repeat the process!
![[Pasted image 20240523164259.png]]
This time, we're probably going to add "un" to our vocabulary, but the *next* time it's possible we'll see "hug" joining our vocabulary!

Basically, we keep repeating this process until we've reached a number of steps (eg 32k, 64k merges). Then you're left with a vocabulary that includes characters, pair of characters, full words, etc. We refer to these as subwords.

==One of the implications is that there won't be any UNK tokens in our training set, using BPE... and any sequence at inference time using the same base characters that we started with similarly won't have unknown tokens. ==
- Note that if we have to use a bunch of the base tokens (eg "a", "b"), then the inference is going to be slower, because we have to process more tokens as a result.

![[Pasted image 20240523205905.png]]
- It's actually 7x faster to use subword tokenization than byte-level tokenization!

---

From HuggingFace NLP Course
- BPE training starts by computing the unique set of words used in the corpus (after normalization and pre-tokenization steps are completed), then starting the vocabulary by taking all the symbols used to write those words.
	- "hug", "pug", "pun", "bun", "hugs" -> ["b", "g", "h", "n", "p", "s", "u"]
	- For real-world datasets, the base vocabulary will include all the ASCII characters, and probably some Unicode characters as well.
	- If, at test time, your tokenizer use a character not seen in the training process, it will be converted to an UNK token (this is a reason why many NLP models are very bad at analyzing content with emojis).
- After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning *merges* of tokens, where we produce a new token by combining two existing tokens. 
	- So at the beginning, these merges will create tokens with two characters, and then, as training progresses, longer subwords (or words).
	- Given counts of `("hug", 10), ("pug", 5), ("pun", 12), ("bun", 4), ("hugs", 5)`
	- We split each word into our characters
	- `("h" "u" "g", 10), ("p" "u" "g", 5), ("p" "u" "n", 12), ("b" "u" "n", 4), ("h" "u" "g" "s", 5)`
	- Then look at pairs; the "hu" is present in "hugs" and "hug", so 15 times in total -- but the most common blogs to "ug", present in "hug," "pog", and "hugs" for a grand total of 20 times. So our first merge rule is that "ug" is aded to the vocabulary.
	- We continue this until we reach the desired vocabulary size.
- ==An interesting thing to note== is that BPE saves not just the resulting vocabulary, but also the merge rules. Because at test time, we tokenize sequences by breaking the sequence into the base vocabulary, and then following the merge rules to tokenize the string! This differs from [[WordPiece]], which follows a similar merging process to define the vocabulary, but has a different way of tokenizing sequences at test time.
