#lecture 
Link: [Link](https://youtu.be/JERXX2Byr90?si=xxTCdvm80D4wfF-G)

Topic: Contextual-based representations

----
# A series on Contextual-Based Representations
Subjects: 
1. Transformer
2. Positional Encoding
3. GPT
4. BERT
5. roBERTa
6. ELECTRA
7. Seq2Seq
8. Distillation
9. Wrapup


# (1/9) Transformers
## Simplified Transformer Block Walkthrough
Our sentence: "The Rock Rules"
![[Pasted image 20240425132718.png]]
- We've paired each of these with a ==token vector== representing the token in the string (ignoring subwords for now), along with a ==positional vector== for the position of the token in the sequence.
- These token representations are ==static word representations== that are very similar conceptually to those we had in the previous era with [[Word2Vec]] and [[GloVe]].
	- We do a similar thing with the positional tokens to get their representations.
- We then combine them by adding them together dimension-wise to get the representations we have in green, which you can think of as the first ==contextual representations== we have in the model.
- The next layer is the famous Attention layer, where we compute scores between our current vector and the other input vectors in the sequence, at this layer of the network. We use Key/Query/Value projection matrices to transform vectors in the sequence, comparing our vector's key vector to other vectors' query vectors via a scaled dot-product attention, return directional pairwise attention scores between our vector and other vectors in the sequence. We then do a attention-score weighted-sum of the value vectors in the sequence to compute the new representation of our vector. Attention doesn't change the dimensionality of our input vector ($d_k$)
	- Aside; Attention is All you Need? The authors saw what happened in the RNN era, and saw the recurrent mechanisms that *added* Attention mechanisms on top of the occurrences to connect everything to everything else. This paper said -- Get rid of those recurrent connections, Attention is all you Need!
- We then apply [[Dropout]]
- We then apply [[Layer Normalization|LayerNorm]]
- The next step is the Feed Forward Neural Network layer. We project our vector into a new (usually higher dimension) space, apply a nonlinearity with [[Rectified Linear Unit|ReLU]], and then project back down to the original dimensionality.
	- Most of the parameters in a Transformer are in the FFNN layers!
- Finally, we have a layer normalization step similar to what we had earlier.


We began with position-sensitive version of static word embeddings, we then piped it through an attention layer, and a feed forward layer. There were some normalizations and dropouts between there, but all the blocks generally follow this same rhythym.


Let's look again at the attention calculation:
![[Pasted image 20240425132818.png]]
These two calculations are actually the same, if you'll believe it!

![[Pasted image 20240425132834.png]]
Here's the same calculation, but in code. Look at it to convince yourself that this is the case!


## Multi-Headed Attention

- We're going to do three parallel calculations corresponding to our three heads
- We do our same dot-products as before, and it's the same calculation that leads to the result, but there are a few new parameters in the calculation:
![[Pasted image 20240425133013.png]]
We've introduced these new matrices to provide more represenetational power inside this attention block
- The subscript of 1 indicates that we're introducing parameters for the first attention head.

We do the same thing for the other attention heads, with parameters corresponding to the appropriate head
![[Pasted image 20240425133157.png|400]]

then we *re-assemble* the pieces
![[Pasted image 20240425133232.png]]

We can summarize... the one big idea worth repeating is that we typically stack transformer blocks on top of eachother. It's typical to have 12, 24, or even 100s of blocks stacked on top of eachother.

![[Pasted image 20240425133306.png]]
(Above: The orange representations are probably multi-headed attention)

![[Pasted image 20240425133431.png|300]]
In the encoder, we use [[Bidirectional Attention]], whereas in the decoder, we (when we do the self-attention, not the cross-attention) do [[Masked Attention]], where we do not attend to future tokens in the sequence.

# (2/9) Positional Encoding

- Transformers themselves have only a very limited capcity to keep track of word order
	- the attention ocnnections are not directional; just a bunch of dot products
	- There are no other interactions between the columns
- We're at risk of not knowing the difference between ABC and CBA!

- Positional encodings have also been used to keep track of hierarchical notions like "premise and hypothesis" in [[Natural Language Inference|NLI]].

There are many perspectives you could take on positional encoding, but let's focus on two questions:
1. ==Does the set of positions need to be decided ahead of time?==
2. ==Does the scheme hinder generalization to new positions?==

Rule: Modern transformer architectures *might* impose a max-length on sequences for many reasons related to how they were designed/optimized.
^ Let's set this aside and ask whether the position encoding *itself* causes it to be harder to generalize to longer sequences.


### 1) Absolute Positional Encoding
We have word representations, and have position representations that we've learned. We then simply add together our word vector and our positional vector.
How does this scheme do wrt our two questions?
1. The set of positions needs to be decided ahead of time! If we decide that there are going to be 512 positions, then what happens when we hit a 513th token in a sequence?
2. May hinder generalization to new positions, even for familiar phenomena. "The Rock" and "...The Rock" might both be represented meaningfully differently. "The Rock" is the same phrase, regardless of whether it's at the start of the phrase, the middle, or the end.
![[Pasted image 20240425134655.png|300]]
### 2) Frequency-based positional encoding 
- The essential idea her is that we define a mathematical function, where, given a position, we're given back a vector that encodes information about that position semantically in the structure.
- In the Attention is all you need paper, they used a sinusoidal function to encode this information. There are lots of other schemes that you can use, the essential thesis is that no matter what number position you give it (1, 5, 5,000,000), you get back a vector, and all of the vectors give some information about the position in the sequence.
- Aside: They pick this scheme based in frequency oscillations, based in Sin/Cos frequencies for vectors, where higher positions oscillate more frequently, and that information is encoded into the position vector we create.
How does this do WRT our two questions?
1. We can get a set of positions for any position, great!
2. This might still hinder generalization to new permissions, by virtue of the fact that we take word representations and add in positional vectors as equal partners... makes it hard for the model to know that the same phrase could appear in multiple places.
![[Pasted image 20240425134646.png]]
### 3) Relative Positional Encoding
- The most promising of the scheme we're going to discuss! 
![[Pasted image 20240425134805.png]]
In blue, we have key representations, and in the final step, we have these value representations that get added in. These are two sets of new crucial parameters that we learn and add in.
Having done this with all the position sensitivity encoded in the vectors, we don't NEED the positional information to be encoded in the GREEN boxes above, because the positional information is encoded in the attention layer.
- The powerful thing about this method is the notion of having a ==positional encoding window==
- With window size d=2:

![[Pasted image 20240425135055.png]]
(See that from position 4->1 still just gets us to -2, since we have the window size of 2.)
Represented in blue here, we just have a few vector: -2, 1-,1 0, 1, 2
We map our $a^K_{4,3}$ to just $w^K_{-1}$ 

![[Pasted image 20240425135200.png]]
We actually learn a relatively small number of position vectors, and what we do is give a small window-relative notion of position that's going to slide around and give us an ability to generalize to new positions.

Relative position encoding: Full definition with learned parameters:
- The cognitive shortcut is that it's just the previous attention calculation with new positional elements added in. In this mode we introduce positional relativityin the ATTENTION layer, not in the EMBEDDING layer.

![[Pasted image 20240425135350.png]]
How does it do on our two quetsions?
1. We don't need to decide positions ahead of time, just the window side! For a large string, we just slide around, using a relatively small number of positional vectors.
2. It shouldn't hinder generalization to new positions; "The Rock" involves the same positional vectors no matter where it appears in the string.

Idea: because we've overcome these two problems, relative positional encoding is a very promising manner of encoding position.


# (3/9) GPT


Let's start with the autoregressive loss function usually used for neural language modeling(think: RNN):

![[Pasted image 20240425140345.png]]
Numerator:
- At position t, we look up the token representation of our token in the embedding layer to get x_t
- We do a dot product of that representation with the hidden layer of our model that we built up at t-1
- The rest of this is softmax normalization, so we do the same thing for every item in our vocabulary
- We look for items that will maximize the log probability.
- The scoring is based on the dot product on the embedding representation of the token we want to predict and the representation that we've built up so far.
![[Pasted image 20240425140908.png]]
At each one of these time steps, we get a score that's proportional to the dot product of the embedding for that token with the hidden representation just prior to that point that we're making the prediction at (and then we exponentiate for the sake of the Softmax)


For GPT, we move to Transformers
- We look up our static representations for our tokens, add our positional encodings, and pass it through some transformer blocks.
- Eventually we get some output representations, and we add some language-specific parameters (eg a classification head) that lets us predict output tokens.

In essence, this is the same thing; because of the nature of the attention mechanisms, we need to do some masking to make sure we don't look into the future..

If we start at position a
The only point we can attend to it to ourselves; only self-attend.

When we then have position b, we can look at a,b
When we then have position c, we can look at a,b,c

"look-back attention" means that we shouldn't look at tokes in the future.

#### GPT: Training with Teacher Forcing
- [[Teacher Forcing]]: No matter what token we predict at this time step, we'll replace it with the actual token at the next timestep.

If the "actual" sequence is "I love hotdogs, they're great."

And we predict:
SOS -> I ✅
SOS I -> love ✅
SOS I love -> candy ❌ (Wrong, it should be hotdogs!)
SOS I love hotdogs -> they're ✅ (See we replaced the incorrectly-predicted "candy" with "hotdogs")

During inference time, though:
Imagine that the user has prompted model with the sequence

SOS The -> Rock
SOS The Rock -> Rolls
SOS The Rock Rolls -> Along

Notice that we don't have any sort of feedback as to whether we're right or wrong here; in generation, we don't have the possibility of doing teacher forcing because we're creating *new* tokens; this is called [[Student Forcing]].

Recall: The model doesn't predict tokens, the model predicts scores over the vocabulary, and we do some inferencing (and there are many schemes here to do this sampling, eg [[Beam Search]]) to choose what the next token will be. 


# (4/9) [[Bidirectional Encoder Representations from Transformers|BERT]]
- BERT is essentially just an interesting use of the Transformer encoder.
- In BERT, every sequence begins with a \[CLS\] token and end with a \[SEP\] token.
![[Pasted image 20240425142504.png]]
Along with our normal positional encoding, we also have a hierarchical positional encoding, given by this token Sent_A; for problems like [[Natural Language Inference|NLI]], we might have a separate token for the premise, and a separate token for the hypothesis, to help encode the fact that the encoding of a word in the Premise is slightly dfiferent than the encoding of the word in the Hypothesis.
![[Pasted image 20240425142622.png|300]]
We combine these representations into these green blocks and then pass them through some Transformer Blocks

[[Masked Language Modeling]] (MLM): Obscure some tokens in the sequence and have the model reconstruct the missing piece.
- We might replace "rules" with a \[==MASK==\] token, and have the model reconstruct that "rules" was the missing piece, using the full bidirectional context
- We could also do random word ==replacement/corruption==, replacing "rules" with some random word, like "every"; and have the model predict what the actual token was at that position. This also uses the full bidirectionanl context of the sequence to do this task.

We generally only do such things to a small percentage of the token.

The MLM loss function:
![[Pasted image 20240425142836.png|300]]
Numerator:
- We use the embedding representation of the token we want to predict, and do a dot product of that with a model representation of the FULL SURROUNDING CONTEXT, leaving out only the representation at T. (in contrast, for the autoregressive objective we looked at earlier, we could only use the previous context.)
- This $m_t$ indicator function is $1$ if we're looking at a masked token ,and $0$ otherwise; so we only get a learning signal from the masked/corrupted token.
	- Sort of inefficient; we do predictions for all time steps, but get an error signal for the loss function ONLY for the ones that are masked.


![[Pasted image 20240425143137.png|300]]
The BERT paper supplements the MLM objective with a Binary Next Sentence Prediction Task
- We create actual sentence sequences (labeld isNext), and create negative instances of randomly chosen sentences that are labeled as NotNext
- The motivation is to let the model learn some discourse-level information for learning about how to construct sequences.



When we think about Transfer Learning/ fine-tuning, there are a few approaches we can take:
![[Pasted image 20240425143345.png|300]]
The standard lightweight thing is to build some ==task-specific parameters==  above the final output representation that's produced by the transformer encoders above the CLS token.  We build a few dense layers on top of that, and do some classification learning there.

![[Pasted image 20240425143456.png|300]]
An alternative is to build an output above ALL of the positions, and pool together all of the output states, and build a classifier on top of that mean pooling/max pooling (or whatever you choose to use) -- this can be powerful as well, because you bring in a lot of information from across the entire sequence.


#### Tokenization and the BERT embedding space
- BERT has a pretty tiny vocabulary, so its uses [[WordPiece]] tokenization; this means that we have lots of word pieces like ##encode. The model rarely UNKs-out unknown tokens, instead breaking them into pieces... with the hope that the masked language model will learn the internal representation of words like "Encode" which get split into multiple separate tokens.

Known limitations (From Devlin et al, 2019, and Yang et al, 2019):
- The original BERT paper is admirably detailed, but it's still very partial in its ablation studies, and in studies of how to optimize the model; in the original paper, we're not looking at the best BERT possible.
- We're creating a mismatch between pretraining and finetuning, since the MASK token is never seen during finetuning.
- The downside of using an MLM that's only 15% of tokens are predicted in each batch. We turn off the modeling objective for the tokens that we don't mask, and we only mask a tiny number of them because we need the bidirectional context to make the prediction.
- BERT assumes that the predicted tokens are independent of eachother given the unmasked tokens, which oversimplifies, since high-order, long-range dependencies are prevalent in natural language. If you mask out both NEW and YORK, it's going to be hard to predict either of them. (Yang's ExcelNet brings this back in, to powerful effect.)



# (5/9) RoBERTa
- [[RoBERTa]] stands for "Robustly Optimized BERT Approach"

Addresses some of the known limitations with BERT, chiefly the observation that the BERT team did an admirably detailed, but still partial set of ablation and optimization studies. The RoBERTa team takes over, trying to do a more thorough exploration of the design space.
- Turns out we just scale the models bigger and nail the hyperparameters and we get a great BERT model!

At a meta level, this paper points to a shift in methodologies: RoBERTa team does a thorough examination of hyperparameters, but it's nowhere near the *exhaustive* hyperparameters sweeps before the BERT era, because it's just too expensive!

![[Pasted image 20240425145421.png]]
- BERT used a static masking approach, meaning they copied their training data some number of times, applying different masks to each copy. That set of copies of the dataset was then used repeatedly during epochs of training, so the same masking was seen multiple times by the model. We can get more diversity if we *==dynamically* mask examples== as we load individual batches, so that subsequent batches containing the same examples have different masking applied to them.
- For BERT, the inputs to the model were two concatenated document segments, which was crucial for their next-sentence prediciton task. RoBERTa incldued sentences that even span document boundaries. 
- Correspondingly, RoBERTa just ==dropped the NSP objective== on the grounds that it wasn't earning its keep.
- RoBERTa ==increased batch size== from 256 -> 2,000
- BERT used a WordPiece tokenizer, whereas RoBERTa used a character-level [[Byte-Pair Encoding|BPE]] algorithm.
- BERT trained only on BooksCorpus and English Wikipedia; RoBERTA added CC-News, OpenWebText, and Stories.
- BERT trained for 1M steps, and RoBERTa trained for 500k steps (with substantially larger batch sizes; the net effect being that ==RoBERTa saw many more instances==).
- BERT team thought they should train on short sequences first, in a curriculum learning fashion. RoBERTa trained only on full-length sequences, dropping that.


# (6/9) ELECTRA

Recall that we noted some limitations of BERT, which included (among others):
- A mismatch between pre-training and fine-tuning, since the MASK token isn't seen during fine-tuning.
- The second downside of using an MLM is that only 15% of tokens are predicted in each batch. Only 15% even contribute to the MLM objective, despite processing every item in the sequence -- this isn't very data efficient!

[[ELECTRA]] aims to improve on this:

![[Pasted image 20240425151728.png]]
- Given sequence $x$ : the chef cooked the meal
 - Create $x_{mask}$ , which is a masked version of the input sequence; can use the same protocol as BERT, say, by masking 15% of the tokens at random
- The generator, a small BERT-like model that processes the input and produces $x_{corrupt}$ , where we replace some of the original tokens not with their original inputs, but with tokens having probability proportional to the generator probabilities.
	- Sometimes we'll replace with the actual token, and other times we'll replace with some other token
- The discriminator, the heart of the Electra model, has a job that is supposed to figure out which tokens in the sequence are original, and which are replaced.
- we train the model jointly with the generator and the discriminator, and allow the generator to drop away, and focus on the discriminator as the primary pre-trained artifact produced by the process.

Includes rich studies about how to set up the generator

![[Pasted image 20240425175727.png]]
The best results come from having a generator that is small, compared to the discriminator.
The intuition here is that it's kind of good to have a weak generator so that the discriminator has a lot of interesting/hard work to do.

![[Pasted image 20240425175926.png]]
Lot better training efficiency, relative to competitors.

Further ELECTRA efficiency analyses


# (7/9) Seq2Seq Architectures

Lets talk about some tasks with a natural sequence-to-sequence structures to them:
1. [[Machine Translation]] (language to language)
2. [[Summarization]] (long text in, shorter out comes out, summarizing input)
3. Free-form [[Question Answering]] (Question+Context -> Answer)
4. Dialogue (utterance -> utterance)
5. Semantic parsing (sentence -> logical form)
6. Code generation (Natural language sentence -> program the sentence describes)

These are all special cases of the more-general *encoder-decoder* problems which is more agnostic about whether the encoding/decoding involve sequences or not.

#### From the RNN Era

![[Pasted image 20240425180436.png]]
- Class RNN formulation of a Seq2Seq problem; 
- Introduction of LSTMs to help the decoder remember what was in the encoding part; We see in the Transformer paper a full embrace of the attention mechanism as the primary mechanism, and removal of recurrent mechanisms.

![[Pasted image 20240425180513.png]]
- Left: Encoder-Decoder, where we fully encode the input on the left side, and then, perhaps with fully different parameters, do a decoding, attending in some way over the encoder state(s).
- Middle: We could also process these sequences with a standard Language Model; characteristic attention mask, where we can only attend to the past, even for the part that we think of as the encoded part.
- Right: We could do a full attention connections... where... we can have every element attend to every other element, and then when we start doing decoding, *that's* when the mask can only look into the past, not the future.

The middle and right options have become very popular lately.

## Encoder-Decoder Model: T5
- An encoder-decoder model that had extensive multi-task unsupervised and supervised training.
- An innovative thing that gives us a glimpse of what was *about* to happen with In-Context learning is provide instructions: "translate English to German: That is good"
![[Pasted image 20240425180815.png|300]]
- We express all these tasks as natural language, which guides the model's behavior as if those task instructions were themselves something like structured information.

## Encoder-Decoder Model: BART
- The essence is that on the encoder side we have a BERT-like architecture, and on the decoder side a GPT-like architecture.
- Unique about [[BART]] is the pretraining: Taking corrupted sequences and figuring out how to uncorrupt them.
	- Text-infilling (whole parts of input are masked/removed)
	- Sentence-shuffling
	- token masking
	- Token deletion
	- Document rotation
- The found the most effective was a combination of the text-infilling and sentence-shuffling, and having the model learn to uncorrupt the sequences.


# (8/9) Distillation
- Seeking models that are more efficient to use at inference time, but nonetheless performant -- [[Distillation]] is a set of techniques for doing that.

Name of the game:
- We have a ==Teacher model==, a performant, generally large model; the goal is to train a smaller ==Student model== that has very similar performance/behavior to the teacher, but is nonetheless much more efficient to use.

Various ways of doing it:
- Lightweight: Have the student mimic the teacher, in terms of its input/output behavior
- Deeper: Train the student to have internal representation that are similar to the teacher's.

Possible Distillation objectives, from least-to-most heavy duty (weighted averages of elements of this list are common):
0. Distill student by training on Gold data for the task
1. Train the student to have the same outputs as the teacher.
	- This doesn't actually require that you have the teacher at distillation time, just a dataset of labeled results.
2. Train the student to have the same *output scores* as the teacher
	- The centerpiece of one of the most famous distillation papers, *Hinton et al, 2015*
	- Requires the teacher at distillation time, because we require those score vectors.
	- ((This usually refers to the softmax output of the teacher model; the *probabilities*))
3. Train the student to have the same final *output states*
	- Requires much more access to the teacher at distillation time
	- From the [[DistilBERT]] paper
	- ((This usually refers to the raw logits from the final layer of the teacher model, before the softmax function is applied; more fine-grained information about the model's decision-making process. The cosine loss suggests that the student model is trained to match the *direction* of these logit vectors, rather than the exact values)).
4. Train the student to have similar hidden states and embeddings as the teacher
	- With an intuition that the student will be more powerful and alike the teacher if it has similar internal representations
	- Requires full access to teacher @ distillation time
1. Train the student to mimic the counterfactual behavior of the teacher under interventions -- instances where we change the internal state of the teacher, and do the same corresponding thing to the student, and make sure they have corresponding behavior.
	- This is relatively new, Chris Potts involved 
	- Requires full access to teacher @ distillation time


Modes of distillation:
1. Standard distillation: Teacher has frozen parameters, only the student parameters are updated
2. Multi-teacher distillation: Simultaneously try to distill various teachers into a single student that can perhaps perform multiple tasks
3. Co-distillation: Student and teacher and trained jointly. Also called "online distillation" (Anil et al, 2018)
4. Self-distillation: The objective includes terms that seek to make some model components align with others from the same model (Zhang et al. 2019)


Distillation has been applied in many domains; We can increase efficiency with almost no loss in performance. 
![[Pasted image 20240425183324.png|300]]
These converge on the same lesson: We can make BERT much smaller by distilling into a much smaller student that still performs very well on (eg) [[GLUE]]


# (9/9) Other noteworthy architectures
- [[Transformer-XL]] (2019): Long context via recurrent connections to previous (frozen) states
- [[XLNet]] (2019): Bidirectional context with an autoregressive language modeling loss, done via sampling different sequence orders.
- [[DeBERTa]] (2021): Separate representations for word and positions, with distinct attention connections. Decouples word and position by decoupling representations, and having distinct attention mechanisms to each part.

### BERT: Known limitations
![[Pasted image 20240425183814.png]]

### Current Trends
- Autoregressive architectures have taken over, but possibly because the field is so focused on generation.
- Bidirectional models like BERT might still have the edge over models like GPT when ti comes to representation.
- Seq2Seq models are still a dominant choice for tasks with that structure.
- People are still obsessed with scaling up to ever-larger LMs, but we're seeing a counter-movement towards "smaller" models (still ~10B parameters).
	- There are a lot of incentives that will encourage small models to become very god
		- Can deploy in more places
		- Can train more efficiently
		- Might have more control of them for things we want to do
