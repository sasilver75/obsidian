#article 
Link: https://jalammar.github.io/illustrated-transformer/

-------

[[Attention]] is a concept that helped to improve the performance of neural machine translation applications.
The [[Transformer]] takes advantage of it to great effect, beating previous state-of-the-art results in machine-translation!

### High-level look

![[Pasted image 20240328203958.png|300]]
Above: The black-box version of the Transformer, which takes in a sentence in (eg) French, and outputs a sentence in (eg) English.

If we peel back just a single layer, we see:
![[Pasted image 20240328204109.png|300]]
Above: We can see that the transformer has two components:
- Encoder
- Decoder

And if we peel back another layer, we see:
![[Pasted image 20240328204150.png|300]]
Above: We see that there are actually (in the paper) stacks of *six* encoders that make up the encoder component of the model, and the decoder component is similarly a stack of 6 decoders!
- The encoders are all identical in structure (but do not share weights)

Let's zoom into a single on of these encoder "blocks"!:
![[Pasted image 20240328204341.png|400]]
Above: We can see that the encoder block is a combination of two major components:
1. Self-Attention Layer
	- This helps the encoder look at other words in the input sentence as it encodes a specific word.
2. Feed-Forward Neural Network Layer

The decoder similarly has both of these layers, but between them is an *additional (cross-)attention* layer that helps the *decoder* focus on relevant parts of the *input sequence*, as understood by the *encoder*.
![[Pasted image 20240328205106.png|450]]
Above: See that the decoder blocks contain an additional layer, which performs [[Cross-Attention]] on the encoder block, rather than the [[Self-Attention]] that's performed in each block (the "bottom" layer, in the diagram).

### Bringing the tensors into the picture
- Let's start to look at the various vectors/tensors and how they flow through the components to turn the input of a trained model into an output.

- In the case of NLP applications, we begin by turning each input *word* into a *vector* by using an [[Embedding]] layer/algorithm.
![[Pasted image 20240328205358.png|450]]
Above: In this example, we're embedding words into a *dense* vector of size 512. This vector space has some sort of semantic meaning to it.

All of the encoders then receive a list of vectors having size 512 (this is a hyperparameter that can be set).
- In the bottom-most encoder, these vectors are the word embeddings produced by our embedding algorithm.
- In every later encoder block, it receives the output of the encoder that preceded it.

After we embed the words in our input sequence, each of them flows through each of the two layers of each encoder:
![[Pasted image 20240328205729.png|350]]
Above: 
- We can see a key property of the transformer, which is that ==each word in s sequence flows through its own path in the encoder!== 
	- This is great because it *==allows for parallelization of processing of a sequence==* (c.f. a recurrent neural network, where we must process each sequence in series, because we have to build up the hidden state that's used to generate outcomes). This is one of the reasons why Transformers are so powerful.

### Now we're encoding!
- The encoder receives a *list of vectors* as input, and processes this list by passing these vectors into:
	1. First, a self attention layer
	2. Then, a feed-forward neural network
- Finally, the output is sent on to the next encoder.

![[Pasted image 20240328210643.png|450]]
Above:
- See that each word vector in the input sequence of words is passed through the self-attention layer, producing a transformed vector which is then fed as input to a feed-forward neural network. The output of this FFNN layer is then passed along to the next layer.

### Self-Attention at a High Level

Let's consider the following sentence:

> "The animal didn't cross the street because it was too tired."

What does "it" in the sentence above refer to?
- To a human, this is simple, but it's not to a computer. ((There are sentences that have real ambiguities too!))
- When the model is processing the word "it", [[Self-Attention]] will help it associate "it" with "animal!"

==As the model processes each word in a sequence, self-attention allows it to look at other positions in the input sequence for clues that can help lead to a better encoding for this word!==
- Self-attention is the method the Transformer uses to bake the "understanding" of the relevant words into the one we're currently processing.
- In the encoder, we use "bidirectional attention", meaning that a word can "look" at all other words in the sentence. In the decoder, in contrast, we use a "causal" or "masked" self-attention, where we can only look at previous words.

![[Pasted image 20240328211528.png]]
Above:
- On the right, as the encoder considers a representation for the word "it", it determines an attention score for all (in this encoder's case; not in a *causal attention* in the *decoder*!) other words in the sequence

### Self-Attention in Detail
- Let's look at how to calculate self-attention using Vectors, then proceed to look at how it's actually implemented using Matrices!

The first step in calculating self-attention is to:
- ==Create three vectors from each of the encoder's input vectors== by multiplying the incoming vector by three matrices trained during the training process. 
	- Query vector
	- Key vector
	- Value vector

![[Pasted image 20240328212216.png|450]]
Above: 
- Notice that these new vectors are *smaller* (size 64) in dimension than the initial embedding vector (size 512). ==They don't *have* to be smaller, this is just an architecture choice to make the computation of [[Multi-Headed Attention]] (mostly) constant==.

So what are these "query," "key," and "value" vectors?
- They're abstractions that are useful for calculating and thinking about attention.

The *second step* in calculating self-attention is to calculate an ==attention score==.
- Say we're calculating the self-attention for the first word in the example above, "Thinking"
	- We need to consider each word in the input sequence, and *score it against this word.*
	- The score determines how much *focus* to place on other parts of the input sequence, as we encode a word at a certain position.

==This attention score is calculated by taking the *dot product* of the Query Vector of our current word with the Key Vector of the respective word we're scoring.==
- So if we're processing the self-attention for the word in position #1, the first score would be the dot product of q1 and k1; The second score would be the dot product of q1 and k2.

![[Pasted image 20240328213538.png|400]]
Above: See the calculation of attention scores between vectors in the sequence.


The third and fourth steps are to:
- ==Divide the scores by 8 (which is the *square root of the dimension of the key vector*== used in the paper, 64). *"This leads to having more stable gradients"*
- We then ==pass the result through a softmax operation==, which normalizes the scores so that they're all positive and add up to 1.

![[Pasted image 20240328213931.png|400]]
Above:
- The softmax score determines how much each word will be *expressed* at this position.

The ==fifth step is to multiply each value vector by its corresponding softmax/attention score== (in preparation to sum them up).
- The intuition here is to keep intact the values of the word(s) we want to focus on, and drown-out irrelevant words (by multiplying them by tiny numbers like 0.001).

"Attention combines the representation of input vectors' value vectors, weighted by the importance score (computed by the query and key vectors)"

==The sixth step is to sum up the weighted value vectors. This produces the output of the self-attention layer at this position.==

![[Pasted image 20240328214648.png|400]]
Above:
- See for our first word in our two-word sequence that we ==create query/key/value vectors for each word in the sequence==.
	- (Note that as we do the following calculations, we sort of "re-use" the key and value vectors; the only thing that changes as we process additional tokens is the query vector of the currently-considered token. *==This is why the KV vectors are cached==*, in what's usually called as a "KV Cache")
- For our currently-considered vector, ==we compute attention scores for each word in the sequence== by dot-producting the current vector's query vector with the key vector of each other word in the sentence (and dividing by the squareroot of the dimensionality of the key vector, then softmaxxing)
- ==Then we take the sum of all value vectors in the sequence, weighted by their attention score== (which is specific to the token that we're currently considering).


### Matrix Calculation of Self-Attention
- The first step is to calculate the Query, Key, and Value matrices.
- We do that by packing our embeddings (vectors) for each word in our sequence into a matrix $X$ , and multiplying it by the weight matrices we've trained ($W^Q, W^K, W^V$ )

![[Pasted image 20240328220038.png]]
Above:
- Every row in the $X$ matrix corresponds to a word in the input sequence! We see again the difference in size of the embedding vector (512, or 4 boxes in the figure), and the q/k/v vectors (64, or 3 boxes in the figure).

Finally, since we're dealing with matrices, we can condense steps 2-6 into a single formula to calculate the outputs of the self-attention layer:
![[Pasted image 20240328220157.png|450]]
Above:
- The creation of a matrix of attention-weighted value vectors for each vector in input sequence.
- Next, we would create the resulting vector by doing a column-wise sum of the $z$ matrix.


### The beast with many heads: Multi-headed Attention
- The paper then further refined the self-attention layers by adding a mechanism called [[Multi-Headed Attention]]!
- This improves the performance of the attention layer in two ways:
	1. It expands the model's ability to focus on different positions.
	2. It gives the attention layer multiple "representation subspaces"
		- With multi-headed attention, we have multiple sets of Query/Key/value weight matrices (the transformer uses eight attention heads, so we end up with eight sets for each the encoder/decoder).

![[Pasted image 20240328220804.png|450]]
Above:
- In multi-headed attention, we maintain separate $W^Q, W^K, W^V$  matrices for each attention head!
	- This means that we would maintain multiple KV-caches, one for each attention head.
- Like before, we multiply our $X$ matrix by each of the above matrices to produce $Q, K, V$ matrices.

If we do the same self-attention calculation we outlined above, just eight different times with different weight matrices, we end up with eight different Z matrices
![[Pasted image 20240328221456.png|450]]
- This leaves us with a bit of a challenge -- what do we do with all of these resulting $Z$ matrices, from our multiple attention heads? The feed-forward layer is not expecting eight matrices -- it's expecting a single matrix (a vector for each word)!
	- How do we do that? ==We concat the matrices then multiply them by an additional weights matrix WO====

![[Pasted image 20240328221659.png|450]]
Above:
- We basically concatenate the various resulting $z$ matrices, and then use a matrix to condense that long matrix back down into the "expected" size.

Here's all of multi-headed attention in one place:
![[Pasted image 20240328222352.png|450]]
Above:
- Given our input sentence, embed each word and stack them into an $X$ matrix.
- For each attention head:
	- Use our $W^Q, W^K, W^V$ matrices to compute our $Q, K, V$ matrices.
	- Calculate our matrix $Z$ (matrix of attention-weighted value vectors) using our Q, K, V matrices
- With our many $Z_{1..N}$ matrices, concatenate them into a wide $Z$ matrix
- Multiply that  $Z_{concat}$ matrix by a $W^O$ matrix to produce a final $Z$ matrix)

If we add all the attention heads to our previous picture, however, things can be harder to interpret:
![[Pasted image 20240328222946.png|300]]
Above: Sort of hard to interpret, huh 😄

### Representing the order of the sequence using Positional Encoding
- One thing that's missing from the model that we've described so far is a way to account for *the order of the words in the input sequence*.
- ==To address this need for positional information, the transformer adds a vector to each input embedding.==
- These vectors follow a specific pattern that the model learns , which helps it determine the position of each word.... or the distance between different words in the sequence
	- The ==intuition== is that adding these values to the embeddings provide meaningful distances between the embedding vectors once they're projected onto the Q/K/V vectors and during dot-product attention.

![[Pasted image 20240328223349.png|450]]
Above: To give the model a sense of the order of the words, we add positional encoding vectors to our initial embedding vectors.

If we assumed that the embedding has a dimensionality of 4, the actual positional encoding might look like this:
![[Pasted image 20240328223449.png|450]]

<<<<<<< HEAD
![[Pasted image 20240328231211.png]]
Above: A visualization of the positional encoding scheme applied during the original Transformer paper
=======
![[Pasted image 20240328223525.png|400]]
Above: A real example of positional encoding for 20 words (rows) with an embedding size of 512 (columns)... You can see that it appears split in half down the center -- that's because the value of the left half are generated by one function (which uses sine) and the right half is generated by another function (which uses cosine). They're then concatenated to form each of the positional encoding vectors.

The formula for positional encoding is described in the paper (section 3.5) -- it's not the only possible method for positional encoding, but gives the advantage of being able to scale to unseen lengths of sequences.

### The Residuals
- One detail that we need to talk about in the encoder that we need to mention is that each sub-layer (self-attention, FFN) in each encoder has a [[Residual Connection]] around it, and is followed by a [[Layer Normalization|LayerNorm]] step.

![[Pasted image 20240328231347.png|400]]
- Above:
	- See that both the Self-Attention and the Feed-Forward Networks layers are wrapped in a residual layer, followed by a layer-normalization step!

If we've going to visualize the vectors and the layer-norm operation associated with self-attention, it would look like this:

![[Pasted image 20240328231600.png|450]]

This goes for the sub-layers of the encoder as well.
If we were going to think of a Transformer as 2 stacked encoders and decoders, it would look something like this:

![[Pasted image 20240328231856.png]]
Above:
- {{It seems like all of the decoder blocks only attend to the output of the *last* encoder block. This probably makes sense.}}

### The Decoder Side
- Now that we've covered most of the concept on the encoder side, let's look at the how the components of the decoder side work as well, and how they work together.

- The encoder starts by processing the input sequence; the output of the top encoder is then transformed into a set of attention vectors K and V, when considered by the cross-attention layer of each of the decoder blocks.
- These help the decoder focus on the appropriate places in the input sequence (or rather in a transformer version of the input sequence)

![[Pasted image 20240328232606.png|500]]
Above:
- After we finish the encoding phase, we can begin the decoding sequence!
- The following steps repeat the process until a special symbol is reached, indicating that the transformer 

![[Pasted image 20240328232916.png]]
Above: (Watch the animation for this, if you can!)

The output of each step is fed to the decoder in the next step, and the decoders bubble up their decoding results just like the encoders did. 

==Just like we did with the encoder inputs, we embed and add positional encoding to those decoder inputs== to indicate the position of each word.  

The self-attention layers in the decoder operate slightly differently from the ones in the encoder.
- In the decoder, the self-attention layer is only allowed to attend to *earlier* positions in the output sequence! 
	- This is done by masking future positions (setting them to -inf) before the softmax step in the self-attention calculation.

The "Encoder-Decoder Attention" cross-attention layer works just like multiheaded self attention, except it creates its Query vectors using the output of the previous decoder block, and the Key and Value vectors using the output of the encoder stack.


### The Final Linear and Softmax Layer
- The decoder stack outputs a vector of floats -- how do we turn that into a word?
- The ==linear layer== is a simple fully-connected neural network that *projects* the vector produced by the stack of decoders into a much *larger* vector called a ==logits vector==!
- Let's assume that our model knows 10,000 unique English words (its ==vocabulary==) that it's learned from its training dataset -- that would make the logits vector 10,000 cells wide, with each cell corresponding to the score of a unique word -- that's how we interpret the output of the model followed by the linear layer.
- The ==softmax== layer then turns those scores into *==probabilities==* (all positive, all add to 1.0). The cell with the highest probability is chosen, and the word associated with it is produced as the output for this time step.

![[Pasted image 20240328234145.png|450]]
Above:
- Given the output from the Decoder stack, we pass it through a linear layer that projects it into a much larger vector, of size `vocab_size` -- these are called the [[Logit]]s
- Then we pipe that through a softmax function so that everything is positive and sums to one -- we consider these as probabilities.


### Recap of Training
- During training, an untrained model would go through a forward pass on a labeled training dataset, and we compare its output with the actual correct output.
- To visualize this, we assume that our output vocabulary contains only six words:
![[Pasted image 20240328235026.png]]

Once we define our output vocabulary, we can use a vector of the same width to indicate each word in our vocabulary -- this is known as one-hot encoding. So we could indicate the word "am" using the following vector:

![[Pasted image 20240328235107.png]]

Following this, let's talk about the loss function that we optimize during training.

### The Loss Function
- Let's say that we're translating "merci" into "thanks"!
- This means that we want the output to be a probabilitiy distribution indicating the word "thanks" -- but since the model isn't trained yet, that's unlikely to happen!

![[Pasted image 20240328235152.png|450]]
So now we have a (bad) output probability distribution from our model, and a "correct" output distribution from our dataset (one-hot encoded). How do we compare them? We simply subtract one from another
- For more, see [[Cross-Entropy]] and [[KL Divergence|Kullback-Leibler Divergence]]

Note that this is an oversimplified example -- we'll usually use a sentence longer than one word -- for example: input "je suis etudiant" and expected output "I am a student" -- So we want our model to successively output probability distributions where:
1. Each probability distribution is represented by a vector of width `vocab_size` (6, in our toy example)
2. The first probability distribution has the highest probability at the cell associated with the word "i"
3. The second probability distribution has the highest probability at the cell associated with the word "am"
4. ... And so on, until the fifth output distribution indicates the `<end of sequence>` symbol, which also has a cell associated with it in the 10,000-long vocabulary.

![[Pasted image 20240328235605.png|450]]

Now, because the model produces the outputs one at a time, we can assume that the model is selecting the model with the highest probability from the probability distribution, and throwing away the rest. That's *one way* to do it, in a method called [[Greedy Decoding]].
- We could alternatively consider, say, the first top two words, then, in the next step, run the model twice (one assuming "A" and the other assuming "B", in the previous step), and repeat this process... this method is called [[Beam Search]], where in our example `beam_size` was two and `top_beams` is also two.

### Go forth and Transform!
