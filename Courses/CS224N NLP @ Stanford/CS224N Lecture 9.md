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

![[Pasted image 20240410170529.png]]

Sometimes you get complete words: "hat", "learn," and other times you get "la", "ern", or "Transformer"+"ify"
So subwords are <= words

And we can sort of combine these, with each token having its own embedding.


![[Pasted image 20240410171115.png]]
Above:
> *"You shall know the meaning of the word by the company it keeps"*
> - (The Distributional Hypothesis, which powered things like Word2Vec)

Extension: The complete meaning of a word is *always contextual*, and no study of meaning apart from a complete contextual understanding can be taken seriously!
- The word *record*
	- Record an album? 🎤
	- Create a record? 💿
	- Set a record? 🏃‍♂️

So when we create a single representation of *record* with an embedding, the representation has to be 

We can build a better representation of words by having contextual representations!
(We use things like attention to build up contextual meaning)
![[Pasted image 20240410171527.png]]
We had pretrained word embeddings; we had a big box on top of it, like an LSTM, that was not pretrained
- We learn our word embeddings separately
- Then we have a task, as we initialize the parameters of our model randomly; we train to predict our label (updating the model parameters, but not the embeddings)

==What if we tried to train *all* the parameters, of both our model and our embeddings?==

![[Pasted image 20240410171652.png]]
Now, we can pretrain the entire structure together! The embedding and model parameters together!
- This gives us very strong representations of language.
- The meaning of "record" and "record" will be different int he contextual representations in the model that know where in the sequence it is and what words are co-occurring with it in the input.

This has also been good at building good parameter initializations for strong NLP models
- We usually say that we start from small, randomly distributed noise close to zero.
	- Here, instead, we way: Let's start our MT system from an initialization given to us from pretraining.

All of pretraining is centered around the idea of reconstructing the input
- A sequence of text that some human has generated
- Hypothesis: By masking out part of it and tasking the NN to reconstruct the input, the NN must necessarily learn a lot about both language and the world (\*) to reconstruct the sentence.
	- ((Or is it just learning a distribution over tokens, I don't know))

I drove to Stanford in \_\_\_\_\_ California
- Palo Alto. We're learning about the geography of CA!

The woman walked across the street checking for traffic over {mask} shoulder
- Her. We're learning about linguistic coreferences!

Overall, the value that I got from the two hours watching it was the sum total of the popcorn and the drink. The movie was {mask}
- Bad. We're learning sentiment!

Iroh went to the kitchen to make some tea. Standing next to Iroh, Zuko pondered his destiny. Zuko left the {mask}
- Kitchen. We learned that Iroh's in the kitchen, and Zuko is next to him. so Zuko is also in the kitchen!

-------

![[Pasted image 20240410172610.png]]
![[Pasted image 20240410172845.png]]
![[Pasted image 20240410173142.png]]
Why does pretraining "help?"
- Pretraining provides some parameters by minimizing loss on the pretraining dataset.
- Finetuning then minimizes loss on the finetuning dataset, starting at the parameters that were left from pretraining. It turns out that the starting point *really really matters*.

Pretraining is a big deal because there's an enormous amount of text on the internet -- much more than exists in your specific use case that you're actually trying to train a model for -- but we can learn useful things from this enormous generic blob of text that can then be used for our task!

![[Pasted image 20240410174300.png]]

# Encoders

![[Pasted image 20240410174357.png]]
We can't do language modeling with an encoder, because they get bidirectional context, and that would be cheating! It's easy to predict the next word if you can just look at the next word!

The idea with training encoders is that we mask out some tokens in the input and predict those words.
==We only add loss terms from words that are *masked out*==. This is called [[Masked Language Model]]ing
- You can see that this looks like we're doing something like masked language modeling, but with bidirectional context; we're ==removing some portions of the sequence, and asking the model to predict it.==
Learn $P_\theta(x|\widetilde{x})$ 
- Learn parameters that let us predict the original text $x$ given a corrupted version of the text $\widetilde{x}$ 
- Specifically we're learning the probability of the sequence given the corrupted sequence.

You can see that the red portion in the diagram above is the linear transformation $Ah_i + b$ 

Q: Do we choose words randomly to mask out, or is there a scheme?
A: Mostly there's random masking.

## BERT: Bidirectional Encoder Representations from Transformers

- In 2018, Devlin et al proposed the Masked LM objective and released the weights of a pretrained Transformer, a model they labeled BERT!
![[Pasted image 20240410174934.png]]
It's such a useful builder of representations in context!

- We tokenize our input, and then we predict a random 15% of (subword) tokens.
	- Sometimes we mask out the chosen token
	- Sometimes we replace the chosen token with a random token
	- Sometimes we don't leave the token changed at all and still predict it!
- ==Why do we sometimes use things other than a {MASK} token?==
	- If we had to build good representations to predict output words given a sequence that has these strange mask tokens in it... well... When we have to output something during real inference, the sequences that we're going to operate on definitely don't have the mask token in them!
		- "Oh, I have no job to do here, because I only have to deal with the mask token!"
			- Now it has to deal with issues where the word is not masked, or the word is wrong! Now it has the chance to be asked to predict anything at any time.
![[Pasted image 20240410175156.png]]
The folks at google had a separate task that's interesting to think about too:
- We had to predict whether one chunk of text follows another chunk of text, or is randomly sampled.
- ==Later work argued that this "next sentence prediction" is not necessary.==

![[Pasted image 20240410175737.png]]
BERT set a shit ton of benchmarks.

Q: So the encoder usually outputs some sort of hidden values... how do we actually correlate those values to stuff that we want to predict?
A: For each input, the encoder gives us a vector of that token in context. How do we take this representation turn it into answers for the task we care about? We basically just stick a classification/regression head -- meaning a linear transformation

![[Pasted image 20240410175949.png]]
- What is BERT good for? It's good for filling in the blanks, but it's much less naturally used for generated text; we wouldn't want to use BERT to generate a summary of something, because it's not really build for it; it doesn't have a natural notion of predicting the next token, given those before it. But we can use it (with a classification head) to do things like sentiment classification of a given sentence.

![[Pasted image 20240410180120.png]]
Note that it was only trained on something like ~1B words of text (the slides are wrong), and only had something like 340M parameters, at the time.
- At the time, this was considered to be an enormous model -- "Oh, only Google can do it!"
- Finetuning was still practical and common on a single GPU, though!


![[Pasted image 20240410180320.png]]
[[RoBERTa]] mentioned >>>>
It turns out that span-based masking might be a little better than random masking -- it's a harder problem.

![[Pasted image 20240410180354.png]]
RoBERTa does much better; they show that we should have trained BERT on a LOT more text (we hadn't invented scaling laws yet; it gave us intuition that we didn't really know a lot about training these things, or about how long we should train them for.)
- ((Basically always use RoBERTa, instead of BERT -- it's just better))

![[Pasted image 20240410180447.png]]
An alternative to finetuning ALL the parameters in the model is a form of [[Parameter-Efficient Fine-Tuning]], where we keep most of the parameters fix, and only finetune others.

We want to make the minimal change from the pretrained model to the model that does the thing that we want, so that we both save money and 

![[Pasted image 20240410180558.png]]
[[Prefix Tuning]]: We freeze all the parameters of the pretrained network itself, and never change any of them. Instead, we make a bunch of fake pseudo-word vectors that we prepend to the beginning of a sequence, and we just train them!
- These would have been inputs to the network, but we specify them as parameters, and just train the values/parameters of the fake words.
- This keeps all the generality of the model params, and is easier to do than finetune the entire network.

Another example of PEFT is [[Low-Rank Adaptation]] (LoRA)
![[Pasted image 20240410180932.png]]
- We have a bunch of weight matrices in our transformer
- We take each, and freeze it; we instead learn a very low-rank little *diff*, and we set the weight matrix value to be the original value plus the low-rank diff (rather than a full-size $\Delta W$ )

# Encoder-Decoders

![[Pasted image 20240410180947.png]]
For these, we could do something like language modeling, where we say that the encoder is a prefix for having bidirectional context, and then we could just predict all the words in the altter half of the sequence, like a language model -- and that would work fine.
- Take a long X, split into 2, give first half to encoder, generate second half with decoder

In practice, what works ***better*** is the notion of Span Corruption:
![[Pasted image 20240410181048.png]]
We mask out a bunch of words  in the input, and then for the output we predict the masked tokens.

This allows us to have bidirectional context (we get to see the whole sequence), and then generate the parts that were missing -- this feels a little like BERT -- but we generate the output as a sequence like we do in language modeling. This might be good for MT, or similar tasks.

![[Pasted image 20240410181211.png]]
A fascinating property of these models:
- [[T5]] was the model that was introduced with "salient span masking". At pretraining time we saw things like 
	- "FDR was born in ..." and you generate "1882"
It's sort of fascinating that in these Closed Book Question Answering tasks that models learn to access information stored in their parameters.
- Answers very often look reasonable, but are sometimes wrong 😄


# Decoders

![[Pasted image 20240410181558.png]]

![[Pasted image 20240410181832.png]]
- 2018: GPT was a big success with 117m parameters!

![[Pasted image 20240410183744.png]]
GPT-2 focused on the generative abilities of the network, and scaled to 1.5B parameters (~13x)
This size of model is still small enough that you can use it and finetune it on a small GPU. Its ability to generate long text at the time was notable.


![[Pasted image 20240410183847.png]]
GPT-3 was 175B (~115x) parameters and trained on 300B tokens -- this was very expensive!
- Emergent Ability: "Language Models are In-Context Learners"
	- It can figure out patterns that are given to it in its context, and continue that pattern.
	- You can give it a pattern of:

> thanks -> merci
> hello -> bonjour
> otter ->

And it will say
> loutre

What are its capabilities and limitations?
- This is STILL an area of open research!

![[Pasted image 20240410184338.png]]
The cost of training these are roughly the # of parameters times the # of tokens

The folks at [[DeepMind]] realized that GPT-3 was comically oversized -- their [[Chinchilla]] model was less than half the size, trained on more data, and was a better model than GPT-3.
- The Chinchilla paper established compute-optimality

![[Pasted image 20240410184432.png]]
[[Chain of Thought]]: The prefix can help specify what task to solve

![[Pasted image 20240410184653.png]]









