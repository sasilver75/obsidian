---

---
#article 
Link: https://nlp.seas.harvard.edu/annotated-transformer/


## Background
- [[Self Attention]] is an attention mechanism that relates different positions of a sequence to eachother, in order to compute an enhanced representation of the sequence.
	- It's been used for a variety of tasks, like reading comprehension, abstractive summarization, textual entailment, and learning task-independent sentence representation.
- To the best of the author's knowledges, the Transfer is the first transduction model relying entirely on self-attention to compute representations of its input and output (rather than using RNNs or Convolutions).


## Part 1: Model Architecture

### Model Architecture
- Most competitive neural sequence transduction models have an ==encoder-decoder== architecture. 
	- Here, the ==encoder== maps an input sequence of symbol representations x = (x1, ..., xn) to a sequence of continuous representations z = (z1, ..., zn). 
	- Given Z, the ==decoder== then generates an output sequence = (y1, ..., yn) of symbols, one element at a time.
	- At each step, the model is auto-regressive, consuming the previously generated symbols as additional input when generating the next symbol.

![[Pasted image 20240105215257.png]]
- The [[Transformer]]  follows this overall architecture of using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder (the left and right halves, respectively).
- We also employ a [[Residual Connection]] around each of the two sublayers, followed by [[Layer Normalization]].
	- That is, the output of each sublayer is thus `LayerNorm(x + Sublayer(x))`, where `x` is the function implemented by the sublayer itself.
- We also add [[Dropout]] to the output of each sublayer, before it's added to the sublayer input and normalized.
- Each layer has two sublayers:
	- The first is a multi-head self-attention mechanism
	- The second is a simple, position-wise fully-connected feed-forward network.

### Decoder
- The decoder is *also* composed of a stack of N=6 identical layers.
- In addition to the two sublayers that are present in each encoder layer, the decoder inserts a *third* sublayer, which  performs multi-head attention over the output of the encoder stack.
- Just like in the encoder, we again employ [[Residual Connection]]s around each sub-layer in the decoder, followed by [[Layer Normalization]].
- We also modify the self-attention sublayer in the decoder stack to prevent positions in the sequence from attending to subsequent (later) positions in the sequence.
	- This masking ensures that the predictions for position `i` can depend only on the outputs at positions *less than* `i`.
![[Pasted image 20240105221045.png]]

### Attention
- An attention function can be described as *mapping a query and a set of key-value pairs to an ouput*, where the query, keys, values, and output are all vectors.
	- You can sort of think of it as a fuzzy hashtable lookup
- The attention output is computed as a *weighted sum of the values*, where weight assigned to each value is computed by a compatability function of the query with the corresponding key.
- The particular attention in this paper is called "==Scaled Dot-Product Attention=="
	- The input consists of queries and keys of dimension $d_k$ and values of dimension $d_v$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$ and apply a softmax function to obtain the weight on the values.
	- In practice, we're able to compute the attention function on a set of queries simultaneously, paced together into a matrix $Q$.

$Attention(Q,K,V) = softmax(QK^T/\sqrt{d_k})V$  

The two most commonly used attention functions are ==additive attention== and ==dot-product== attention. 
- Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt(d_k)}$ . 
- Instead , additive attention computes the compatibility function using a feed-forward neural network with a single hidden layer.
While they're similar, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly-optimized matrix multiplication code.

So why did we do the $\frac{1}{\sqrt{d_k}}$ ?
- We suspect that for large values of $d_k$ , the dot products grow large in magnitude, pushing the softmax function into regions where it has very small gradients... to counteract this effect, we scale the dot products by $\frac{1}{\sqrt(d_k)}$ .


![[Pasted image 20240105222642.png]]
- [[Multi-Head Attention]] allows the model to jointly attend to information from different representation subspaces at different positions.
	- In other words, the network can learn "multiple things." Each attention will attend to different aspects of the input.
- In this work, they employed $h=8$ parallel attention layers, or attention heads. They even used reduced dimensions fore each head (such that the multi-head attention head's dimensionality times the number of heads equaled the original dimensionality of the single-headed example.)


### Applications of Attention in our Model
- The Transformer uses multi-head attention in *three* different ways!
	1. In "encoder-decoder attention" layers, the *queries* come from the previous *decoder* layer, and the memory *keys and values* come from the output of the *encoder*. This is basically the decoder asking the question: "Which aspects of the encoder output should I pay attention to?"
	2. The *encoder* contains self-attention layers. In a ==self-attention layer==, all of the queries, keys, and values come same place. In this case, the the input to the encoder's self-attention layer is the output of the previous layer in the encoder! 
		- Each position in the encoder can attend to all positions in the *previous* layer of the encoder.
	3. Similarly, self-attention layers in the *decoder* allow each position in the decoder to attend to all positions in the decoder up to and including that position. 
		- We need to prevent leftward information flow in the decoder to preserve the autoregressive property. We implement this by masking out (setting to -inf) all of the values in the softmax which corresponding to illegal connections. This is also called [[Masked Attention]] or Masked Self-Attention.


### Position-wise Feed-Forward Networks
- In addition to attention sublayers, each of the layers in both our encoder and decoder contain a fully-connected [[Feed-Forward Network]], is applied to each position separately and identically.
- This consists of two linear transforms with a [[Rectified Linear Unit]] Activation Function in between.
	- $FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$ 
- While the linear transformations are the same across positions, they use different parameters from layer to layer.

### Positional Encoding
- Since our model contains no recurrence and no convolutions, in order to for the model to make use of the *order* of the sequence, we must inject some information about the position of the tokens in the sequence.
- To this end, we add [[Positional Encoding]]s to the input embeddings at the bottom of the encoder and decoder stacks.
	- The positional encodings have the same dimension $d_{model}$ as the embeddings, so the two can be *summed!* 
	- There are many choices of how to do positional encodings; both learned and fixed.
		- In this paper, we use sine and cosine functions of different frequencies.
![[Pasted image 20240105225846.png]]
- They also experimented with using learned positional embeddings instead, but found that the two versions produced nearly identical results.
	- They chose the sinusoidal version because it allows the model to extrapolate to sequence lengths longer than the ones encountered during training.


## Part 2: Model Training

### Training Data and Batching
- We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using [[Byte-Pair Encoding]]. 
- Each training batch contained a set of sentence pairs containing approximately 25,000 source tokens and 25,000 target tokens.

### Hardware and Schedule
- We trained our models on one machine with 8 NVIDIA P100 GPUs.
- We trained the base models for a total of 100,000 steps (each taking ~.4 steps) over 12 hours. The big models we trained were 300,000 steps (each taking 1 second) over 3.5 days.

### Optimizer
- We used the [[Adam]] ...
- We varied the learning rate over the course of training.
- This corresponds to increasing the learning rate linearly for the first n steps, and then decreasing it thereafter proportionally to the inverse square root of the step number.

![[Pasted image 20240105230558.png]]
Above: The learning rate schedule

### Regularization
- [[Label Smoothing]]
	- During training, we employed label smoothing of value $E_{ls} = 0.1$ . This is as form of regularization in classification where we turn a label vector of (eg) 
$$ \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix} $$
	into
	$$ \begin{bmatrix} .05 \\ .9 \\ .05 \end{bmatrix} $$
	- This reduces overfitting via [[Regularization]], but hurts [[Perplexity]], as the model learns to be more unsure. Label smoothing actually starts to penalize the model if it gets very confident about a given choice.

### Additional Components: BPE, Search, Averaging
- There are four aspects that we didn't cover explicitly above:
	1. Byte-Pair Encoding(BPE)/Word-piece: We use a library to first preprocess the data into subword units.
	2. Shared Embeddings: When using BPE with shared vocabulary we can share the same weight vectors between the source/target/generator.
	3. [[Beam Search]]: Too complicated to cover here!
	4. Model Averaging: The paper averages the last k checkpoints to create an ensembling effect. We can do this after the fact if we have a bunch of models.

## Results
- They're good!


















