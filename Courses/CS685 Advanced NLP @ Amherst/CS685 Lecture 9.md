Video: https://www.youtube.com/watch?v=G9z6AZoJxYY&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=9

(Skipped 3-8 because they're basically all about Transformers and BERT; I might come back to them later, but I feel I have a handle on these.)

----

# Topic: Tokenization in Language Models

![[Pasted image 20240523161743.png]]
Above: We don't want to split O'Neill into O and Neill, but we DO want to split aren't into "are" and "not", right?

And what do we do when we see an unknown word, that we've never seen in our training data (or otherwise isn't in our vocabulary)?
- With word-level tokenization, we have *no way* of assigning an index to an unseen word! This means we don't have a word embedding for that word, and can't process the input sequence... Should we ignore the word, which could harm our performance?
- "Solution": We replace low-frequency words with a special UNK token, and use this token to handle unseen words at test time too.
	- Think: We also need UNK tokens during training time, so the model knows how to handle them! We do this by replacing low-frequency tokens with it, during training time, as discussed above.
	- We lose a lot of information about texts with a lot of rare words/entities, though!

![[Pasted image 20240523162355.png]]
You can see how we lose critical information about texts by using these unk tokens, in many cases.

Another limitation of word tokenization in general is that there's a lot of duplication information in your vocabulary -- consider "open," "opened," "opens," "opening", etc. All of these are different forms of the same word, but we have different vectors for them in the case of work tokenization, even if "open" occurs 50,000 times, and "opening" occurs 2 times, which probably isn't enough to get a good representation of opening (and the 50,000 occurrences of "open" don't help us get a good representation of "opening"'s vector.)

![[Pasted image 20240523162620.png]]
What about character-level tokenization?
- An effect of this is that our input sequences are now much much longer. Why might we not want such long input sequences?

Now, becaues of advances in GPU hardware, character-level tokenization isn't a completely unreasonable idea! Poeple have explored going beyond that tot even byte-level tokenization, where sequences get very long!

Subword tokenization is the most commonly used thing today, this is what we'll focus on most:
![[Pasted image 20240523162959.png]]
Strikes a good balance between character-level tokenization and word-level tokenization.
- Produced often by [[Byte-Pair Encoding]] (BPE)

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