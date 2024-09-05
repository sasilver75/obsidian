Variants:
- Unigram
- Bigram
- Trigram
- Fourgram
- Fivegram


![[Pasted image 20240521200516.png]]
This is a cool example of using 1-2-3-4-gram models to approximate Shakespeare! See that they improve in quality.

So increasing the length of this N-gram prefix is good, but if we increase it too much, we don't have enough occurrences of some prefix to estimate the distribution well. So N-gram models are a balance: You want as much context as you can get, without running into a problem where you have (eg) 0-few occurrences of your prefix.

In general, this is an insufficient model of language becauesl anguage has long-distance dependencies!
- "The computer which I had just put into the machine room on the fifth floor crashed."
	- Any reasonable N probably won't connect `computer` and `crashed`


![[Pasted image 20240521202426.png]]
Above: these s symbols are start and end of sequences
You can do the same thing with a bigram/trigram, etc. We just count and then normalize.


There's a reason few problems with using N-Gram models:
1. Sparsity problem; Let's say that "Students opened their laptop" never occurred in the training set? Then it'll be 0 in our numerator from the above picture. But what if it's a reasonable sentence? This can be somewhat ameliorated using [[Label Smoothing]]. There are problems with this too -- if you don't observe instances of "Students opened their laptop," you really have no idea whether the word is being used correctly or not; there's only so much you can do if you know nothing about a word or the context in which it occurs. This is also true of word types that just don't occur in the vocabulary. A "solution" there is to use an UNK token.
2. They're not great at modeling long-distance dependencies in language if the dependency is long than N tokens away (in an N-gram)
3. The storage cost of an N-gram model! For a bi-gram model, we have these big tables where the rows are the prefix, and the columns are the next word in the sentence. Every time we want to add more context (eg by going from a bigram to a trigram), that table size increases exponentially!
4. We treat all words/prefixes independently of eachother! "Students opened their {blank}" and "pupils opened their {blank}" have nothing to do with eachother, even though they're very similar, semantically!