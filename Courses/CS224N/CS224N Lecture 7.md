#lecture 
Link: https://www.youtube.com/watch?v=wzfWHP6SXxY&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=7


Today:
- Machine Translation Topics
Later this week:
- More talk on final projects, and more talk on practical tips for building NN systems


Agenda:
1. Introduce a new task, [[Machine Translation]] (15 mins)
2. A new neural architecture: Sequence to Sequence
3. A new neural technique: [[Attention]]


-------

# (1) Pre-Neural Machine Translation
- Where were we and what did we do in prehistory?

[[Machine Translation]]: The task of translating a sentence $x$ from a `source` language to a `target` language $y$

In the early 1950s, there started to be work in Machine Translation
- Most things with "Machine" in the name are old times. In the US, this came about in the context of the cold war; we wanted to keep an eye on what Russians were doing! Computers had *just* been good at codebreaking in WW2; maybe we can use them for translation?
	- Lots of talk of the "electronic brain!"
	- We realized that we hadn't reckoned with the ambiguity of language -- people didn't understand computers or linguistics very well, at the time 😄 -- We had very simple rule-based systems and dictionary word-lookup, but this didn't really work well enough -- human languages are much more complex; human words have many contextual meanings, we use idioms, metaphors, etc.

In the 1990s-2010s: Statistical Machine Translation

![[Pasted image 20240408202604.png]]
- Can we just start with data about translations (sentences and their translations) and learn a probabilistic model that can predict translations, given a sentence?
- Suppose we're translating French -> English
	- We want to find the best English sentence $y$ given French sentence $x$
		- $argmax_y P(y|x)$ 
			- "We want to maximize the probability of generating good translation y, given input text x"
	- We use Baye's Rule to break this down into two components to be learned separately
		- $argmax_yP(x|y)P(y)$
			- "If instead, we had probability of English sentences, and probability of a French sentence, given an English sentence, it turns out that we were able to make more progress. It's not immediately obvious why, but it allowed the problem to be separated into two parts, which made it more tractable."


So how do we learn P(x | y)?
- First, we need a large amount of *parallel data*  (meaning pairs of human-translated French/English sentences)
	- Fortunately, in the modern world, there are many places where parallel data is produced in large quantities! The European Union for instance produces a large amount of parallel text among different languages. The Canadian Parliament for instance produces a large amount of parallel text. The HK parliament produces much English/Chinese, etc. We can use all of this to build models!

![[Pasted image 20240408202921.png]]
But how do we do it?
- Question: How to learn translation model P(x | y) from the parallel corpus?
- Break it down further:
	- Introduce a latent $a$ variable into the model: P(x, a|y)
		- This *alignment variable* is a word-level correspondence between source sentence $x$ and target sentence $y$
		- If we could induce this alignment between the two sentences, then we could have probabilities of pieces of how likely a word/short phrase is translated in a certain way...

![[Pasted image 20240408203051.png]]
What is ==Alignment==?
- The correspondence between particular words in the translated sentence pair. It captures the grammatical difference in the languages.
- Typological differences in languages (subject before sentence, object before subject/object, etc.) lead to complicated alignments!
- Note: Some words have *no counterpart* in the other language! So we can have words that don't get translated at all!
	- eg in France, we say "The France" or "The Germany," so when we translate this to english the "The" bit goes away.


We can additionally have problems where there are many-to-one translations, where a single French word can get translated as *several* English words! For instance {French Word} --> "Aboriginal People"
Or the reverse, where multiple French words translate to a single english word! "mis en application" --> "implemented"

Or you can get even more complicated, where four english words are translated to two french words, but not in any obvious (2+2) way.
- "The poor don't have any money" -> "Las pauvres sont démuis" (the poor are moneyless)

![[Pasted image 20240408203738.png]]
Alignments are categorical things, they aren't probabilistic -- so we need to use special learning algorithms (eg EM Maximization) to learn these 🤔. In old versions of the class, we spent a lot of time talking about latent variable algorithms -- So you'll have to go off and see CS 228 (Probabilistic Graphical Models) if you want to learn more about that!

Note: SMT = "Statistical Machine Translation"


![[Pasted image 20240408204014.png]]
Naive: "What if we enumerate every possible $y$ and calcualte its probability? We can't possibly do that, there are way too many possible y's." We need a better way... for Language Models, we generated words one at a time and laid out a sentence -- but here we need to deal with the fact that things occur in different *orders* in source languages and translations! So we want to break it into pieces... but 

![[Pasted image 20240408204225.png]]
This is the way it was done in the past:
- Start with a source sentence (german)
	- As is standard in German, you get the second position as a verb; that's probably not in the right position for where it's going to be in the english sentence, so we might need to rearrange words.
- We have words/phrases that are reasonably likely translations of german words/phrases; these are effectively the lego pieces that we'll create the translation from.
- We'll generate the translation piece by piece, kind of like how we do it with our neural langauge models
	- Start with an empty translation
	- Use one of these lego pieces
	- Explore different possible ones... there's a search process.
		- We could translate er with "he", or start with translation the second word as "are"
		- It's probably much more likely to start with "he" than "are," though the latter isn't impossible.
		- We also record which german words we've translated


# 1990s-2010s: Statistical Machine Translation (contd)
- SMT was a huge research field
- The best systems were *extremely complex*
	- With many hundreds of important details that we haven't mentioned above
	- Systems had many separately designed subcomponents (language model, translation model, reordering models, ...)
	- Lots of feature engineering
		- Need to design features to capture particular language phenomena
	- Require compiling and maintaining extra resources
		- Like tables of equivalent phrases
	- Lots of human effort to maintain
		- Repeated effort for each language pair!

# (2) Neural Machine Translation (2014+)

![[Pasted image 20240408222242.png]]

People figured out how to do MT using large neural networks, which proved to be extremely successful and largely blew away everything that proceeded it.

## What is Neural Machine Translation?
- It's a way to do MT with a *==single end-to-end neural network==!*
	- What goes in is the source text, and what goes out in the translation text
	- We train this on parallel data (source, translation)
- The neural network architectures are called ==sequence-to-sequence (seq2seq) models==, and often involve *two neural networks* (eg 2 RNNs)
	1. One of the NNs encodes the source sentence
		- (We know from earlier that we can use LSTMs that start at the beginning, go through the sentence, and update the hidden state at each time. We used final state of the encoder RNN as an *encoding* of the source sentence)
	2. One of the NNs decodes the translation sentence
		- Langauge model that will generate the target sentence conditioned on the final hidden state of the encoder RNN (the hidden state for the decoder *starts* as the final hidden state of encoder RNN!)

![[Pasted image 20240408222624.png]]
Above:
This is the test time behavior
For the training time behavior, we use the same type of seq2seq model, but we do the decoder part just like training a language model, where we teach it to predict each word that's found in the translated sentence... (Just do it like you'd imagine it, in terms of replacing the already-generated sequence with the correct sequence at each timestamp)


Sequence to Sequence is versatile!
- Seq2Seq is useful for more than just MT
	- Summarization (long text -> short text)
	- Dialogue (previous utterances -> next utterances)
	- Parsing (input text -> output parse as sequence)

The sequence-to-sequence model is an example of a ==Conditional Language Model==
- ==Language Model==: Because the decoder is predicting hte next word of the target sentence $y$
- ==Conditional==: Because the predictions are also conditioned on the source sentence $x$
- Rather than starting at the beginning of the sentence and just generating a translation, we instead start with *something* -- an encoding of the source sentence that will help us determine what a good translation should look like!

In the NMT, we directly calculate $P(y|x)$:
- $P(y|x) = P(y1|x)P(y2|y1,x)P(y3|y1,y2,x)...$

With this, we'll often get Perplexities like 4, or 2.5! (Recall: This as if we rolled a 4 sided dice, in terms of "surety"/"surprise")


Question: How do we train an NMT system?
- Answer: Get a large parallel corpus (eg from the European Parliament proceedings)

![[Pasted image 20240408223413.png]]

Once we have these parallel sentences, what we do is take batches of source sentences and target sentences, encode the source sentence with our encoder LSTM, feed the final hidden state into a target LSTM, and then train word-by-word by predicting the next word and comparing it to the actual next word, determining some loss.


# Multi-Layer RNNs (stacked RNNs)
- RNNs are already "Deep" in one dimesnion (when we unroll them over many timesteps)
- We can also make them "deep" in another dimension by applying multiple RNNs! This is a multi-layer RNN!
- This allows the network to compute more complex representations
	- The lower RNNs should compute lower-level features, and the higher RNNs can compute higher-level feature.

Just like in other neural networks (FFNNs, CNNs), you get much greater power/success by having a stack of multiple layers.
- Think: "I could have a single LSTM with a hidden state of dimension 2000, or have four layers of LSTMs with hidden layers of 500 each" -- it makes a huge difference!


![[Pasted image 20240408224558.png]]
When we build one of these end-to-end Neural Machine Translation (NMT) systems, if we want them to work well, single-layer LSTM encoder-decoder systems just don't work well.
- But we can build something no more complex than the model we explained above by making it a multi-layer stacked LSTM NMT system!
- We have a multi-layer LSTM that goes through the source sentence.
	- At each point in time, we calculate a new hidden representation. 
	- Rather than stopping there, we feed it as input into another LSTM
	- Our representation of the source sentence of our encdoder is the rsult of the stack of three hidden layers, which are then fed in as the initial hidden layer into generating translations.
((Above: It's interesting to me that it seems like the encoder actually *does* spit out a token for the final translation!))


![[Pasted image 20240408225143.png]]

It's almost invariably the case that a 2 layer LSTM is better than a 1 layer LSTM; after that, things become less clear. 

# Decodcing Strategies
- The simplest way to decode is to take the most probable token in our output distribution; this is called [[Greedy Decoding]], and it's sort of the obvious thing to do, and doesn't seeeem like it could be a bad thing to do, but it turns out that it can be a fairly problematic thing to do!
- ==Problems with Greedy decoding==
	- You're takin what locally seems like the best choice, and then you're stuck with it, and there's no way to undo that decision later!
![[Pasted image 20240408225747.png]]

![[Pasted image 20240408225912.png]]
Far too expensive!

![[Pasted image 20240408230143.png]]
Beyond greedy decoding, the most important method used is [[Beam Search]] decoding
- The core idea: At each step of the decoder, we keep track of the $k$ most probable partial translations (which we call ==hypotheses==)
	- $k$ is the ==beam size== (in practice around 5 to 10)
It's much more efficient than exhaustive search!


Let's look at at example of Beam Search with a search of $k=2$  (usually it'd be much larger)

- We start with our start symbol
- What would be the two most likely words next, according to our language model?
	- He, I
	- Calculate log probabilities for each
- For each of these k hypotheses, we find the next k (2) likely words to follow them
	- He hit, He stuck, I was, I got
	- Calculate the log probabilities for each
	- At this poitns it seems like it will turn into some exponentially growing tree
- We look at the scores of our four hypotheses, and we *prune* down to the k with the highest score! We ignore the rest.
- For those two, we again generate $k$ hypotheses for the next word
	- He hit a, He hit me, I was hit, I was stuck
	- We again calculate the log probabilities for each
	- Keep the top $k$  hypotheses by score
- ...
- At the end... we trace back through the tree to get the full sentence

There's one more criteria: ==The stopping criteria==
- In greedy decoding, we usually decode until the model produces an EOS token
- In beam, different hypotheses might produce different lenght sequences
	- WE don't want to stop when we hit our first EOS token!
	- We put it aside and continue exploring other hypotheses via beam search
	- We stop when we hit some cutoff length or when we have $n$ complete hypotheses, when we'll look through the hypotheses we've completed, and pick the best one (by score).

So we have our completed hypotheses, and select the one with the highest score.

![[Pasted image 20240408230850.png]]
==NOTE==: Unfortunately, longer hypotheses generate lower probability scores (because we're multiplying < 1 values successively!)
- In a way, this appears unfair! Extremely large sentences are indeed less likely than short ones IRL, but not by much! We don't want to be doing 2-length sentences all day!
- As a compensation, we often ==normalize by length, so that we have a per-word log probability score==. In practice this works pretty well, even if it's theoretically kind of weird.

![[Pasted image 20240408231718.png|300]]

![[Pasted image 20240408231732.png|300]]


How do we evaluate Machine Translation?
- The BEST WAY is to show a human being who's fluent in both languages and get them to pass judgement on how good the translation is, but this is expensive and isn't always possible or scalable.
- In terms of "good enough,": Consider the ==[[BLEU]] Score==
	- Have a/several human translations of the source sentence, and compare a machine translation to those pre-given human translations
	- We score them for similarity by calculating ==n-gram precisions== (words that overlap between the computer/human translations -- using unigrams, bigrams, trigrams, 4-grams)...
		- Plus a penalty for too-short systems translations
	- Gives a score of 0-100 (100 = exactly producing one of the human translations, 0= not even a single unigram overlaps with any of the human sentences).

==BLEU is a *useful* but imperfect measure==
- There are many valid ways to translate a sentence
	- So a GOOD translation can get a *poor* BLEU score because it has a *low n-gram overlap* with the human translation :(

![[Pasted image 20240408232154.png]]
Above: Y axis = BLEU score
- It seemed that by 2017 or so, progress had stalled! 😔 -- People's big hope was that if we built a more complex model that *knew about the syntactic structure of language* and made use of tools like dependency parsers, we might be able to build better translations! 🤡 (Purple bar)
- As the years went by, it became obvious that this barely seemed to help
- The blue bars are NMT systems -- see how much they improved the field!
	- GPT2, GPT3, BERT

![[Pasted image 20240408232747.png]]











