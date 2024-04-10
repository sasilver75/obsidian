---
aliases:
  - BPE
---
1. Start with a vocabulary containing only *characters* and an "End of Word" symbol
2. Using a corpus of text, find the *most common adjacent characters* (say, a+b) and add the concatenation as a subword (ab)
3. Replace instances of the character pair with the new subword; repeat until desired vocab size is reached.

What you end up with is a vocabulary of very commonly-occurring substrings, which can be used to build up words.