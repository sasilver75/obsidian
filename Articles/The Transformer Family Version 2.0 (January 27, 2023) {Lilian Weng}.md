#article 
Link: https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/

------

![[Pasted image 20240429220939.png]]

# Transformer Basics
- The Transformer (vanilla transformer) model has an [[Encoder-Decoder Architecture]]. as commonly used in many neural [[Machine Translation|MT]] models.
- Later, simplified transformers were shown to achieve great performance in language modeling tasks:
	- The encoder-only [[Bidirectional Encoder Representations from Transformers|BERT]]
	- The decoder-only [[GPT]]

## Attention and Self-Attention
- Attention is a mechanism in neural network that helps the model learn to make predictions by selectively attending to a given set of data.
- The amount of attention is quantified by learning weights, and thus the output is usually formed as a weighted average.
- [[Self-Attention]] is a type of attention where the model makes predictions for one part of a data sample using other parts of the observation about the same sample. 
- There are various forms of Attention -- the Vanilla Transformer relies on the *scaled dot-product attention*.
	- ==Given a query matrix, key matrix, and value matrix, the output is a weighted sum of the value vectors, where the the weighting assigned to each value slot is determined by the dot product  is determined by the dot product of the query with the corresponding keys:==

![[Pasted image 20240429222432.png]]

## Multi-Head Self-Attention
- [[Multi-Headed Attention]] is a key component in Transformer.
- Rather than only computing the attention once, the multi-head mechanism splits the inputs into smaller chunks and then compute the scaled dot-product attention over each subspace in parallel.
- ==The independent attention outputs are simply concatenated and linearly-transformed into expected dimensions.==

![[Pasted image 20240429223020.png]]

Where .;. is a concatenation operation

![[Pasted image 20240429223251.png|400]]

## Encoder-Decoder Architecture
- The [[Encoder]] generates an attention-based representation, with the capability to locate a specific piece of information from a large context.
	- It consists of a stack of 6 identity modules, each containing two submodules: a multi-head self-attention layer and a pointwise fully-connected feed-forward network.
	- By pointwise, we mean that it applies the same linear transformation to each element in the sequence; can be viewed as a convolutional layer with filter size 1.
	- Each submodule has a [[Residual Connection]] and [[Layer Normalization]]
	- All of the submodules output data of the same dimension $d$.
- The [[Decoder]]'s function is to retrieve information from the encoded representation.
	- The architecture is quite similar to the encoder, except that the decoder contains *==two==* multi-head attention instead of one in each identical repeating module.
	- The first multi-head attention submodule is *==masked==* to prevent positions from attending to the future.

![[Pasted image 20240429223933.png]]

## Positional Encoding
- Because ==self-attention operations are permutation invariant==, it's important to use proper [[Positional Encoding]] to provide *ordering information* to the model.
- The positional encodings have the same dimension as the input embedding, so it can be ==added to the input embedding directly==. 

The vanilla Transformer considered two types of encoding:

### Sinusoidal Positional Encoding
- An absolute encoding of the position of a token in the sequence
![[Pasted image 20240429224815.png]]
![[Pasted image 20240429224833.png]]

### Learned Positional Encoding
- Learned positional encodings assigns each element a learned column vector, which encodes its absolute position, and furthermore this encoding can be learned differently per layer.
	- ((Assuming from this that it's one of the types of positional encodings that you don't add singularly at the beginning))

### Relative Position Encoding 
- Shaw et al (2018) ==incorporated relative positional information into $W^k and $W^u$ ==
- ==Maximum relative position is clipped to some maximum absolute value of $k$,== and this clipping operation enables the model to "generalize" to unseen sequence lengths.
	- Therefore, 2k+1 unique edge labels are considered
- [[Transformer-XL]] proposed a type of relative positional encoding based on reparameterization of dot-product of keys and queries.
	- To keep the positional information flow coherently across segments, Transformer-XL encodes the ==*relative*== position instead, as it could be sufficient enough to know the position offset for making good predictions.

### Rotary Position Embedding
- Rotary position embedding ([[Rotary Positional Embedding|RoPE]]) encodes the absolute position with ==rotation matrix==, and multiples keys and value matrices of every attention layer with it to inject relative positional information.
- When encoding relative positional information into the inner product of the i'th key and the j'th query, we want to formulate the function in a way such that the inner product is only about the relative position `i-j`. ==RoPE makes use of the rotation operation in Euclidean space, and frames the relative positional embedding as simply rotating the feature matrix by an angle proportional to its position index.==

![[Pasted image 20240429230021.png|400]]

# Longer Context
- The length of an input sequence for transformer models at inference time is upper-bounded by the context length used for training.
- Naively increasingly context length leads to high consumption in time and memory, and may not be supported due to hardware constraints.

Let's talk about several improvements to support long contexts at inference:

## Context Memory
- The vanilla Transformer has a fixed a limited attention span; the model can only attend to other elements in the same segments during each update step, and no information can flow across separated fixed-length segments. This results in:
	- The model cannot capture very long term dependencies.
	- It's hard to predict the first few tokens in each segment, given no or thin context.
	- The evaluation is expensive. Whenever the segment is shifted to the right by one, the new segment is re-processed from scratch, though there is a lot of overlap tokens.

[[Transformer-XL]] ("Extra Long") modifies the architecture to reuse the hidden states between segments with an additional memory. The recurrent connection between segments is introduced into the model by continuously using the hidden states from the previous segments.

![[Pasted image 20240429230618.png|350]]

In addition to the hidden state of the last layer for the same segment, it also depends on the hidden state of the same layer for the previous segment.
By incorporating information from previous hidden states, the model extends the attention span much longer in the past, over multiple segments.
The Transformer-XL needs to use relative positional encoding because previous and current segments would be assigned with the same encoding if we encode absolute positions, which is undesired.

***[[Compressive Transformer]] (Rae et al, 2019)*** extends Transformer-XL by compressing past memories to support longer sequences. It explicitly adds *memory* slots of size $m_n$ per layer for storing past activations of this layer, and to preserve long context. When past activations become old enough, they're compressed and saved in an additional compressed memory of size $m_{cm}$ per layer. Both memory and compressed memory are FIFO queues, and there are several choices of compression functions ((Omitted)).

![[Pasted image 20240429231312.png]]

((Some more omitted information))

### Non-differentiable External Memory

#### K-NN-LM (Khandelwal et al, 2020)
KNN-LM enhances a pretrained LM with a separate kNN model by linearly interpolating the next token probabilities predicted by *both* models.
- The kNN model is built on an external K-eyValue store which can store any large pretraining dataset or OOD dataset.
- Nearest neighbor retrieval happens in the LM space using libraries like [[Faiss]] or ScaNN.
- At inference time, the next token probability is a weighted sum of two predictions:
![[Pasted image 20240429231706.png]]

#### SPALM (Yogatama et al, 2021; Adaptive semiparametric language models)
- Incorporates Transformer-XL-style memory for hidden states frmo an external context as short-term memory, and a KNN-LM-style key-value store as long memory.
![[Pasted image 20240429231859.png|300]]

#### Memorizing Transformer (Wu et al, 2022)
- Adds a kNN-augmented attention layer near the top stack of a decoder-only Transformer. The special layer maintains a Transformer-XL-style FIFO cache of past key-value pairs.
- The same QKV values are used for both local attention and kNN mechanisms. The kNN lookup returns the top-k (key-value) pairs for each query in the input sequence and then they are processed through the self-attention stack to compute a weighted average of retrieved values. The two types of attention are combined with a learnable per-head gating parameter.


### Distance-Enhanced Attention Scores

#### Distance-Aware Transformer (DA-Transformer, 2021) and Attention with Linear Biases (ALiBi, 2022)
- [[Attention with Linear Biases|ALiBi]] and DA-Transformer are both motivated by similar ideas
- In order to encourage the model to extrapolate over longer context than what the model is trained on, we can explicitly attach the positional information to every pair of attention score based on the distance between key and query tokens.
	- ((Recall that the vanilla transformer only adds positional information to the input sequence, while later , improved encoding mechanisms alter attention scores of every layer, such as [[Rotary Positional Embedding|RoPE]], and they take on forms very similar to distance-enhanced attention scores))

- DA-Transformer multiplies attention scores at each layer by a learnable bias that's formulated as a function of the distance between key and query.
- Instead of multipliers, [[Attention with Linear Biases|ALiBi]] adds a constant bias term on query-key attention scores, proportional to pairwise distances. This bias introduces a strong recency preference, and penalizes keys that are too far away. 

### Make it Recurrent

#### Universal Transformer (Denghani et al, 2019)
- The Universal Transformer combines self-attention with the recurrent mechanism in RNNs, aiming to benefit from both a long-term global receptive field of the tranformer, and learned inductive biases of RNNs.
- Instead of going through a fixed number of layers, the UT dynamically adjusts the number of steps using "adaptive computation time". If we were to fix the number of steps, the UT would just be equivalent to a multi-layer Transformer with shared parameters across layers.

![[Pasted image 20240429233821.png|450]]
![[Pasted image 20240429233832.png|400]]


## Adaptive Modeling
- Adaptive modeling refers to mechanisms that can *adjust* the amount of computation, according to different inputs. For example, some tokens may only need local information, and thus demand shorter attention spans, or others are relatively easier to predict and don't need to be processed through the entire attention stack.

### Adaptive Attention Span
- A key advantage of the Transformer is the ability to capture long-term dependencies (cf RNNs). If the attention span could atdapt its length flexibly and only attend further back when needed, it would help to reduce both computation and memory cost to support longer maximum context size in the model.

#### Adaptive Attention Span (Sukhbaatar et al, 2019)
- Proposed a self-attention mechanism that seeks an optimal attention span; They hypothesized that different attention heads might assign scores differently within the same context window, and thus the optimal span would be trained separately per head.
- Using "Adaptive Computation Time", the approach can be further enhanced to have flexible attention span length, adapting to the current input dynamically.

### Depth-Adaptive Transformer
- At inference time, it's natural to assume that ==some tokens are going to be easier to predict, and thus don't require as much computation as others. Therefore, we may only process its prediction through a limited number of layers to achieve a good balance between speed and performance.==

#### Depth-Adaptive Transformer (Elabyad, 2020) and Confident Adaptive Language Model (CALM, Shhuster 2022) 








































