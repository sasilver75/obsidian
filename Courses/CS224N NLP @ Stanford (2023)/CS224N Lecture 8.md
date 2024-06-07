#lecture 
Link: https://www.youtube.com/watch?v=wzfWHP6SXxY&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4&index=8

# Subject: Transformers

(Including here the end of the previous lecture 7, where they spent the last 3 minutes introducing Attention)

The problem with Seq2Seq models that use a multi-layer encoder-decoder RNN/LSTM-style architecture is that the encoder has to capture *all of the relevant information about the source sentence* in a single vector to pass on to the decoder!
This is referred to as ==The Botleneck Problem==
![[Pasted image 20240409175851.png]]

Introduce: [[Attention]] : 
- ==At each step of the decoder, we use a *direct connection to the encoder* that lets us focus on a *particular part* of the source sequence and use it to generate the words that come next.==

Here's a diagram, with no equations.
![[Pasted image 20240409180304.png]]
![[Pasted image 20240409180312.png]]
"Which of these encoder states is most like my decoder state?"
Based on this attention distribution over previous tokens, we're going to generate a new attention output -- this attention output is going to be an average of the hidden states of the encoder model, weighted by teh attention scores.
We then take this attention output, combine it with the hidden state of the decoder RNN, and together, the two of them are going to be used to predict the next token.
At that point, we chug along and continue doing these attention computations at each position.

![[Pasted image 20240409180324.png]]


\ --------
End of Lecture 7
\ --------

-----

Lecture 8 -> This is now jumping to 2023, rather than 2021

Today, we're going to talk about self-attention and transformers!

Plan:
1. From recurrence (RNN0 to attention-based NLP models
2. The Transformer model
3. Great results with Transformers
4. Drawbacks and variants of Transformers

![[Pasted image 20240409181115.png]]


Now, we have the same goals, but different building blocks!


Issues with recurrent models:

![[Pasted image 20240409182332.png]]
- "Linear interaction distance"
	- RNNs are unrolled "left to right" (or right to left) -- in encodes the notion of linear locality (nearby words often affect eachothers' meanings -- like "Tasty Pizza!")
	- Problem: RNNs take O(sequence length) steps for distance word pairs to interact
It can be quite difficult to learn that these things are related!

![[Pasted image 20240409182426.png]]
If you have many steps between related words, it can hard to learn the dependencies between them
- It's hard to learn these because of gradient problem!
- The linear order of words is "Baked in"; we already know that linear order isn't the right way to think about sentences, though!
	- It's hard to learn "the chef was" above, because there's this qualifying phrase between!

![[Pasted image 20240409182641.png]]
Lack of Parallelizability
- The forward and backward passes have O(sequence length) unparallelizable operations!
	- They're *recurrent!* They build up this hidden state and they roll across the sequence -- you can't compute the RNN hidden state for timestep 5 without computing timestep 4... 3... 2... 1... -- it's not parallelizable!
	- It would be awesome if we could use our very powerful GPU hardware!

GPUs can perform a bunch of independent computations at once -- not being able to use this inhibits training on very large datasets!


Recapping:
- The two problems we noted were:
	- Linear Interaction problems (difficulty of modeling long sequences)
	- Parallelizability problems



Recall:
[[Attention]] treats each words representation as a QUERY to access and incorporate information from a set of VALUES
- We saw attention from the decoder to the encoder; today, we'll think about attention within a single sentence!

All words attend to all words in the previous layer!
- I might attend to you, independent of how far away from me you are!

==You can think of Attention as performing a fuzzy lookup in a key-value store!==

  ![[Pasted image 20240409184602.png]]

![[Pasted image 20240409184629.png]]
Notice that our word embeddings and attention mechanism don't incorporate any notion of the order of the token in the sequence! We can create/learn/define positional encodings that we *add* to our token representations!

![[Pasted image 20240409184711.png]]
Not very intuitive yet, is it?

![[Pasted image 20240409184827.png]]
Alternatively, we could just learn a matrix that lets us represent our positional information!
- Downside of this is that if we have a sequence of length n, and the dimensions d, we can't have sentences longer than length n! oops


Another problem:
- Based on our understanding of attention that we've done so far, there aren't any nonlinearities that help us fit interesting questions!
- Solutions:
	- Add a feed-forward network (with a nonlinear activation function)

$m_i = MLP(output_i)$ 
    $= W_2 * ReLU(W_1output_i + b_1)+b_2$  

![[Pasted image 20240409185159.png]]
You can fit a lot of computation into these 


Another problem: We need to ensure that we don't "look at the future" when predicting a sequence!
- Like in machine translation
	- ((Is this obviously true for MT?))
- Or language modeling

![[Pasted image 20240409185846.png]]
We mask out our attention matrix, setting the values  to -inf as appropriate 

![[Pasted image 20240409190330.png]]

That was our last big building block for self-attention!

![[Pasted image 20240409190425.png]]
Above:
- Maybe you repeat that SelfAttention/FeedForward block multiple times 😜

---

# Part 2: The Transformer Model

What we were pitched in the previous section is a minimal attention architecture.
- The Transformer architecture as we'll present it now is not necessarily the endpoint of search for how to represent language, though it's now ubiquitous.

![[Pasted image 20240409215026.png]]
The examples of learning multiplet things using [[Multi-Headed Attention]]
- How do we ensure they learn different things? We don't! In practice they end up learning different things, though.

Let's talk about how to do this efficiency

- Instead of each word being a vector of dimension d, instead we'll look at these as big stacked matrices.
	- Let's stack all of our n d-dimensional word embeddings into a matrix of size n x d

Now our matrices K, Q, V  are all doing to be d x d, so we can multiply them  as so

$XK$ , $XQ$, $XV$ 
These are all (n x d) (d x d), resulting in again an (n x d) matrix like we had as input! So we're doing a transformation without changing the dimensionality.

![[Pasted image 20240409215616.png]]

Now, let's look back at Multi-headed attention, which gives us the ability to look at multiple patterns, for different reasons.
- We define multiple Q, K, V matrices -- one for each attention head!
![[Pasted image 20240409215817.png]]

For each head, we define an independent Q, K, V matrix
- the d/h is for computational efficiency; it result in us projecting down to a lower-dimensional space.

We concatenate the outputs (each being d x h), and mix them together with a final linear transformation


Let's go through it visually!
- Even though we compute h many attention heads, it's actually not really more costly!
	- In single headed attention, we computed XQ; we do the same thing in multiheaded attention
		- Reshape to (n x h x d/h)

Almost everything else is identical, and the matrices are the same size
![[Pasted image 20240409220036.png]]
Picture: See that the XQ has those three little 'columns'? Thats us showing each of the heads!
So the math ends up working very similarly.
And then we use that P matrix to mix them together...

![[Pasted image 20240409220559.png]]
==great reasoning for why we use the scaled dot product (dividing by the square root of the dimensionality of the key vector!)==


![[Pasted image 20240409220820.png]]
The transformer diagrams that you see around the web often show this "==add and norm==" box


[[Residual Connection]]s
![[Pasted image 20240409221046.png]]
They're a good trick to help deep models train better, helping solve the vanishing gradients problem
Instead of having the next layer be the result of Layer(previous layer), it's now NextLayer = PreviousLayer + Layer(PreviousLayer)
As a result, we only have to learn the "Residual"

![[Pasted image 20240409221128.png]]


## [[Layer Normalization]]
- Another thing to help your model tarin faster!
- The intuitions around LayerNorm and the empiricism around why it learns well maybe aren't super connected, but... you should imagine that the variation within each layer... things can be very big or small... that's not very informative, because of variations between the gradients, or I've got weird things going on in the layer that we can't control... Some things explode, some things shrink. We want ot cut down on uninformative variation between layers
![[Pasted image 20240409221435.png]]
We're going to normalize our (word) vector to unit norm, with mean zero and unit variance.
- We sum up all the values, and divide by the dimensionality for the mean
- We estimate standard deviation as the square root of the  average squared distance of the mean

We have some optional $\gamma$ and $\beta$ params that can be learned

![[Pasted image 20240409221904.png]]


![[Pasted image 20240409222133.png]]


![[Pasted image 20240409222149.png]]
- On the decoder, we have masked multi-head self attention, followed by cross-attention, where we use our decoder vectors as our queries, and then use the output of the encoder as our keys and values. So every word in the decoder looks at every possible word in the outputs of the encoder.

