https://www.youtube.com/watch?v=cG3PQX64rKE&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=16

# Topic: Positional Embeddings and Efficient Attention

----

Recall that the models, when computing self-attention, don't have a notion of the order of the sequence without some explicit injection of positional information.

![[Pasted image 20240604184402.png]]
In the original Transformer paper, they used a collection sinusoidal function that would give a positional vector for a certain positive, and then added those positions.

But we've also seen positional embeddings that start randomly initialized and are updated to minimize the loss, like other parameters. 
- This might not a good idea in all cases! If we have a maximum training sequence length of 20, but at test time we want to process a sequence with length 60, then we don't have any learned embeddings for positions 20-60!
	- This is one of the benefits of an additive sinusoidal embedding :), is that they can give embeddings for any position!

One class of positional embeddings are [[Absolute Positional Encoding]]/Embedding.
- Regardless of whether it's fixed or learned, we have a different positional embedding for each absolute position in the sequence.
	- Position 1 has a specific positional embedding
	- Position 2 has a specific positional embedding
	- ...


This is in contrast to [[Relative Positional Encoding]]/Embedding, where the goal is to represent every PAIR of tokens.
- "Students opened their books"
	- Note that Students (1) and books (4) position
- The idea behind relative positional embeddings is.. let's say we had a sentence before our other one.
- "Students walked into the classroom. Students opened their books."
- In that case, Students and books' absolute positional embedding has changed quite a bit, but their relative difference in positions (3) is the same!
	- Maybe we only care about how close two words are, rather than their absolute position in the sequence!

But it's not immediately obvious how to get this feature if we're just *adding* something to our word embedding at the beginning of the process, right? What do we possibly add to our Students vector? Relative Positional Embeddings are a pairwise thing (of which there are many, from Student), so it's not like there's a single vector that we should obviously be adding.

Relative Positional Embeddings were first used/popularized by [[T5]]
- Problem: Makes it difficult to have a [[KV Cache]]!

## Relative Positional Embeddings
- Features
	- Generally cannot be added directly to the input vectors at the bottom of the transformer stack ([[Rotary Positional Embedding|RoPE]] is an exception!). 
	- Instead, most of these methods directly modify the Attention Matrix!

###  ALiBi
- [[Attention with Linear Biases]]: We're not going to add anything at all to the input!
![[Pasted image 20240604185505.png]]
We compute our query key dot products without any positional information!
We form our matrix with all of our dot products (then do our masking, softmaxes, etc)
![[Pasted image 20240604185551.png]]
If we added no positional information about queries and keys, then permuting the order of words will have no impact on dot prodcuts.

In [[Attention with Linear Biases|ALiBi]], ==they decided that they'd just modify these scores (before doing softmax), where the magnitude of the dot product decays when the query and key become further away from eachother==!

![[Pasted image 20240604185706.png]]
Above: M is some hyperparameter called slope

So even though there was no positional information in the dot products, we post-hoc addd some positional bias to the scores directly!
None of this is on the right side of the plus sign is learned; this is just a fixed thing, with m being a hyperparameter.
- With different attention heads, we choose different values for m (eg 1/2, 1/4, 1/8).

A pretty straightforward technique and very easy to implement (all you have to do is modify your attention matrix using these simple linear biases).

ALiBi enables extrapolation beyond the training sequence length, meaning that even if we train the model with sequences of length 500, at test time, it might still work if you feed it sequences of length 2000 tokens -- it's not clear how well, though -- but definitely more than learned or fixed/absolute embeddings.

Note that ==positional info is only affecting  q, k, but not our v(value) vectors!== 
This is also true with the [[Rotary Positional Embedding|RoPE]] embeddings we'll do next. If you think about it, the most important place to inject order information is in the attention computation -- if you know two words are far away and irrelevant, you can assign their attention to be some low value, thus, whatever the value vector is, we don't give it much weight.

Q: "Once you have really long sequences, won't the negative bias between two distant tokens (that are *actually* related) increase to the extent that the attention-adjusted values will be ~0?"
A: yep! :(


## Rotary Positional Embeddings (RoPE)
- [[Rotary Positional Embedding|RoPE]] enables [[Relative Positional Encoding]]/Embedding without modifying the attention matrix like [[Attention with Linear Biases|ALiBi]] does!
	- ==We just have to do some modification to the original query and key vectors==, and we can keep everything else the same -- there no need to modify the attention computations at every layer.
	- However RoPE does NOT use additive positional embeddings like in the original Transformer paper. ==Instead, we *rotate* the original q, k vectors by a matrix/vector product with a rotation matrix==!
	- We rotate each by some angle, and there will be some nice properties such that if we take the dot product of a rotated key and rotated query, the dot product should be a function of the *relative position* only,  not the *absolute* position.
		- It's interesting that the angle we rotate these queries and keys by is a function of the *absolute* position, but the *dot product* is a function of the *relative position*. 

