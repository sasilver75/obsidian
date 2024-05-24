---
aliases:
  - BPE
---
1. Start with a vocabulary containing only *characters* and an "End of Word" symbol
2. Using a corpus of text, find the *most common adjacent characters* (say, a+b) and add the concatenation as a subword (ab)
3. Replace instances of the character pair with the new subword; repeat until desired vocab size is reached.

What you end up with is a vocabulary of very commonly-occurring substrings, which can be used to build up words.

---

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

