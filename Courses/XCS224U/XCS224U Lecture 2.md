#lecture 
Link: [Lecture](https://www.youtube.com/watch?v=J52Dtu40esQ&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp&index=2)
Course Page: https://web.stanford.edu/class/cs224u/
- Mentions that there's a CS224U podcast

-------

Reviewing some things from last time
- We tried to immerse ourselves in this moment of AI and think about how we got here

The first two units of the course
- Transformers
- Retrieval-Augmented In-Context-Learning

But there is more to our field than just big language models and prompting, and many ways to contribute!

The third course unit: Compositional Generation
- The COGS benchmark
- A relatively new synthetic dataset trying to see whether LMs have systematic solutions to Language problems

![[Pasted image 20240422114600.png]]
It's basically a parsing problem  ((Not interested!))
Given a sentence, map it to the logical form (Lina gave the bottle -> * bottle ( x _ 3); give ...)

In this class, we'll work with an improved dataset, ReCOGS, to create a new dataset that we think is fairer -- but it remains incredibly hard. We have more confidence that it's testing something about semantics -- but it's still incredibly hard for our best systems -- there needs to be a breakthrough for our systems to do well, even on these incredibly simple sentences.

Some topics that we hope will improve our final probjects:

The first topic
- Better and more diverse benchmark tasks
	- We need measurement instruments to understand how well our systems are doing
	- "Water and air, the two essential fluids on which all life depends, have become global garbage cans" - Jacques Cousteau

==We ask a lot of our datasets:==
1. To optimize models
2. To evaluate models
3. To compare models
4. To enable new capabilities in models
5. To measure field-wide progress
6. For basic scientific inquiry into language and the world.



![[Pasted image 20240422115236.png]]
Above: benchmark saturation
Partially, what we're seeing is that our evaluations were really about "machine tasks"
- So we had humans do these "machine tasks" to get a benchmark of how good humans are at these machine tasks
Perhaps we need to change these.

Reference to: [[Dynabench]]
Reference to [[Dynaboard]]

![[Pasted image 20240422115547.png]]
^ Sort of reminds me of [[HELM]]


----

# Guiding Ideas

Static Vector Representations of Words
- Back in the old days, the way we represented examples was with feature-based sparse representations
	- Feature function ("Yes or no, referring to animate thing," "Yes or no ending in chracters -ing"), etc.

Other things like [[TF-IDF]]: Instead of writing feature functions, I'll just keep track of co-occurrence in large bodies of text.


1. ==Feature-based (sparse): Classical lexical representaitons==
	- The dawn of statistical revolution in NLP era; We represented examples with *feature-based sparse representations*: Feature function ("Yes or no, referring to animate thing," "Yes or no ending in characters -ing"), etc... and then composing a vector of these binary responses. You'd end up with long, hand-designed vectors of 0s and 1s.
2. ==Count-based methods (sparse): PMI, TF-IDF, etc.==
	- Up until just before Word2Vec, Glove. Used methods like Pointwise Mutual Information and [[TF-IDF]], things long used in the field of information retrieval. Just keeping track of co-occurrence patterns in large bodies of tents. Essentially counting and reweighting some of the counts.
3. ==Classical dimensionality reduction (dense): [[Principal Component Analysis|PCA]], [[Singular Value Decomposition|SVD]], [[Latent Dirichlet Allocation|LDA]], etc.==
	- Sort of co-occurred with #2 above, and often mixed with it.
	- Basically taking count data and giving us reduced-dimension versions of that count-data, allowing us to capture higher-order notions of co-occurrence (words that co-occur with words that I co-occur with, for example). Turns out to be very powerful -- lots of semantic affinity comes from this.
4. ==Learned dimensionality reduction (dense): [[Autoencoder]], [[Word2Vec]], [[GloVe]]==
	- You might start with some count data, but you have a ML algorithm that learns to compute dense learned representations from this count data. Sort of like step 3 but infused with more of what we know from ML

From here, we just do other methods on top of contextual representations.

![[Pasted image 20240424144447.png]]

==These above are still all ***static representations*** for words== -- you get one vector per word! Chris's experience is that words have many meanings, depending on context!
*The vase broke. Dawn broke. The news broke. Sandy broke the world record. Sandy broke the law. The burglar broke into the house. The newscaster broke into the move broadcast. We broke even.*

==Having fixed, static representations of words was NEVER going to work!==
The vision of contextual representation models is that we don't even TRY to do that -- we embrace that every word can take on a different meaning depending on everything that happens around it -- we'll just have a bunch of token-level representations that get transformed by their context.


A brief history of contextual representation
1. November 2015: Dai and Le (2015): Showed hte value of LM-style pretraining for downstream tasks
2. August 2017: McCann et al (2017) (CoVe): Pretrained bi-LSTMs for MT, and showed that this was a useful start-state for downstream tasks.
3. February 2018: Peters et al (2018): [[ELMo]] first showed how very large scale pretraining of bidirectional LSTMs can lead to rich multipurpose representations.
4. June 2018: [[GPT]]
5. October 2018: [[Bidirectional Encoder Representations from Transformers|BERT]]

As a field, we have been moving from smaller, more-structured/biased models, to larger, less-structured/biased models that impose essentially nothing on the world.

Attention is all you need: You don't even NEED to maintain a state as you roll across the sequence.

Notion: We should ==model the subparts of words==
- If you look back at ==ELMo==, what they did to embrace this insight is incredible
	- Character level representations
	- Convolutions on top of those character-level representations
	- Form a representation at the top that's the average+convolution of the convolution layers
	- ==Result: A vocab that has information about characters, subparts of words, and words in it (How talk is similar to talking, and talked) -- a simple unigram parsing would miss this!==
- The vocabulary for ELMo is a 100k words, quite large -- it's still the case that if you process real data, you ahve to "UNK-out" (mark out as unknown) many words we encounter.

The field says: Forget all that, instead: Tokenize our data so that we just split apart words into subword tokens.
![[Pasted image 20240424145940.png]]

We're talking about contextual models, so even though the "##nu" token doesn't itself mean very much, we know that the model is going to see the full sequence, and will reconstruct (we hope) the word from all the pieces.
==We take this for granted now, but it's incredibly insightful to think about it==.


Guiding idea: ==Positional Encoding==
- Instead of using a static embedding space like GLoVE, we'll also represent tokens with positional encodings, which just keep track of where the token appeared in the sequence we're processing.
- So "rock" in position 2 has a different representation from "rock" in position 47 in the string. ***It will be partially the same word, but partially a different word.***
- ==It's incredibly important about how we do positional encoding==


Guiding idea: Massive scale pretraining
- Have contextual models with all these tiny little parts of words (tokens) in them, and we train on sequences of them... and do this at an incredible scale -- some magic happens as you do this on more and more data, on larger and larger models.


Question from a student: Why don't tree-based models work as well for language models as attention-based models?
AnsweR: Chris's personal perspective is that all of the trees that we've come up with are kind of wrong, so we're making it harder for models by putting them in a bad initial state. What if we use a transformer to use explainability methods to see what sort of tree structure they induce -- because maybe those are closer to true tree structures for languages? And maybe there shouldnt' be one tree structure - instead, one for syntax, one for semantics, one for other things... We should learn the right structure, rather than impose them!


Question: How do we represent numbers with (eg) a WordPiece tokenizer?
Answer: Wonderful question -- he's sure that the low-level tokenization choice will influence your model's math abilities. A paper that evaluates this could really help us understand which model could be intrinsically limited (or not).













