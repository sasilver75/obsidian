#lecture 
Link: https://www.youtube.com/watch?v=DGfCRXuNA2w

# Subject: Pretraining


![[Pasted image 20240409224210.png]]
Word Structure and Subword models
- When we were teaching Word2Vec, we assumed some finite vocabulary V that you defined after looking at some data.
	- You have words like "hat", "learn," etc.
- That's all well and good... we know what to do when we see those words, but when we see variations like (Stretched out "heey" or misspelled "laern" instead of "learn" or "Transformerify", which isn't a word but has an obvious meaning)
	- (Variations, Mispellings, Novel Items)
	- ==Language is always doing this== -- young people are always making new words 😜! It's a problem for your model, even though the meanings should be relatively well defined!

So what do we do when we see them?
- Maybe we map them to some *unknown* token UNK? 
	- But this is bad -- we're losing *tons* of information!

In many other languages, this is a substantially larger problem than in English! 
In languages with more complex **morphology** (word structure) than English... like in Swahili, where you can conjugate a word more than 300 different ways! Should each of these get an independent vector under our model? That makes no sense, because they all obviously have a lot in common! This is a mistake for efficiency and learning reasons!

So what do we do?

### The Byte-Pair Encoding Algorithm

Let's look at subword modeling: We don't even try to define the set of words that exist - instead, we'll define our vocabulary to include *parts of words* that can be mixed and matched and combined! We split words into known subwords.

A simple and effective algorithm for this using [[Byte-Pair Encoding]] (BPE):
1. Start with a vocabulary containing only *characters* and an "End of Word" symbol
2. Using a corpus of text, find the *most common adjacent characters* (say, a+b) and add the concatenation as a subword (ab)
3. Replace instances of the character pair with the new subword; repeat until desired vocab size is reached.

What you end up with is a vocabulary of very commonly-occurring substrings, which can be used to build up words.

Sometimes you get complete words: "hat", "learn," and other times you get "la", "ern", or "Transformer"+"ify"
So subwords are <= words

