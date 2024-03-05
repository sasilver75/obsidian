#article 
Link: https://cameronrwolfe.substack.com/p/decoder-only-transformers-the-workhorse?utm_source=post-email-title&publication_id=1092659&post_id=142044446&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email

------
![[Pasted image 20240304124941.png]]
The current pace of AI research is staggering -- it's impossible to keep up with the field and maintain a good grasp on all topics. Recent advancements in LLMs include:
1. New foundation models ([[OLMo]], [[Gemma]])
2. Better alignment techniques ([[Direct Preference Optimization|DPO]] vs [[Proximal Policy Optimization|PPO]] vs [[REINFORCE]])
3. Exotic topics like model merging

But despite all of this, one thing has remained constant: The [[Decoder-Only Architecture]] for Transformers
- Let's understand it comprehensively and explore how it's evolved!


# The Self-Attention Mechanism
- [[Self-Attention]] transforms the representation of each token in a sequence based on its relationship to other tokens in the sequence.
- ![[Pasted image 20240304131733.png]]

### Understanding Scaled Dot Product Attention

> An attention function maps a query and a set of key-value pairs to an output, where the aforementioned query, keys, values, and outputs are all vectors.
> The output is computed as a ***weighted sum of the values***, where the ***weight*** assigned to each value is computed by a ***compatability function*** of the query with the corresponding key.
> ((Said again: ""Attention combines the representation of input vector's value vectors, weighted by the importance score (computed by the query and key vectors)."))

![[Pasted image 20240304132151.png]]

Projecting the input:
- The input to a self-attention layer is simply a batch of toke  sequences, where each token in the sequence is represented with a vector.
- Assuming we have a batch size of `B` and each sequence is of length `T`, then our self-attention layer receives a tensor of shape `[B, T, d]` as input, where `d` is the dimensionality of the token vectors.
	- ((Each row is a sequence... the width/columns are the number of tokens... the depth is the cardinality of each token vector in the sequence))
- Let's outline the self-attention operation using only *one sequence of tokens as input*, which we can represent as a list of vectors, or a matrix:
![[Pasted image 20240304132646.png]]

The ==first step== of self-attention is to **perform three separate** (linear) **projections of the token vectors in our input sequence, forming key, query, and value vector sequences!**
![[Pasted image 20240304132733.png]]
Above:
- To create the Query, Key, and Value projections of our input sequence, we multiply our input sequence (represented by a matrix) by our Query, Key, and Value matrices -- we end up with three separate sequences of token vectors.

#### Computing Attention Scores
- After we've projected our input sequence into Query/Key/Value sequences, we can generate our *attention* scores using our Query and Key vectors.
	- We compute an attention score `a[i,j]` for every pair of tokens `[i,j]` within the sequence. 
	- Attention scores lie in the range `[0, 1]` and quantitatively characterize how much token `j` should be considered when computing the new representation for token `i` 
		- ((It seems that this means that the scores aren't bi-directional, ie that `a[i,j] != a[j,i]` ))
- ==Practically, we compute these `a[i,j]` attention scores by taking the *Dot Product* of the query vector for token `i` with the key vector for token `j`.==

![[Pasted image 20240304135447.png]]
- We can ==efficiently== compute all of the pairwise attention scores by ==*stacking* the query and key vectors into two matrices, and *multiplying* the query matrix with the transposed key matrix.==
	- The result of this operation is a matrix of size `[T,T]` (where `T` is the sequence length). We call this the ==Attention Matrix==, and it contains all pairwise attention scores in the sequence.
	- From here, ==we *divide each value* in the attention matrix by the square root of `d`== (empirically improves training stability) and applies 
		- (Note: `d` typically represents the square root of the dimensionality of the key vectors in the transformer model)
	- After ==softmax== has then been taken, the attention scores for each token lie within the range `[0,1]` and form a valid probability distribution:

![[Pasted image 20240304140403.png]]

#### Value Vectors
- Once we have the Attention scores, deriving the *output* of self attention is easy!
- ==The output for each token is simply a weighted combination of the value vectors, where the weights are given by the attention score!==
- ==To compute this in batch==, we can simply stack all the Value vectors into a matrix, and take the product of the Attention matrix with the Value matrix.
- Notably, self-attention preserves the size of its input -- A *transformed*, `d`-dimensional output vector is produced for every `d`-dimensional token vector within the input! Each token's output representation is just a weighted average of value vectors with weights given by attention scores, which are computer as the dot product between our token's query vector and every tokens' value vector.


## Causal Self-Attention for LLMs
- The self-attention operation described above forms the basis of the transformer architecture.
- However, the transformer's decoder uses a slightly more complex version of self-attention called *Masked* [[Multi-Head Attention]]!
#### (1/2) Masked Self-Attention
- Decoder-only transformers use a variant of self-attention called [[Masked Attention]], or [[Masked Attention|Causal Attention]]... While vanilla (or [[Bidirectional Attention]])allows all tokens in the sequence to be considered when computing attention scores, *masked* self-attention modifies the underlying attention pattern by "masking out" tokens that follow a given token within the sequence.
- *==Masked self-attention prohibits us from looking forward in the sequence during self-attention==.*

![[Pasted image 20240304142536.png]]
- Prior to performing the softmax operation across each row of this matrix, we can set all values *above* the diagonal of the attention matrix to *negative infinity*.
- By doing this, we ensure that, for each token, all tokens that *follow* this token in the sequence are given an attention score of zero after the softmax operation has been applied.

#### (2/2) Multi-Headed Attention
- The attention operation we have described so far uses softmax to normalize attention scores that are computed across the sequence.
	- We know that softmax generally preferences one/few items, and pushes the others to zero.
- ==Using softmax limits the ability of self-attention to focus on multiple positions within the sequence== -- the probability distribution can easily be dominated by one (or a few) words.
	- To solve this, we typically compute attention across multiple attention "heads" in parallel!
![[Pasted image 20240304143743.png|200]]

Within each head, the masked attention operation is identical -- except:
1. We use ==separate Key, Query, and Value vectors==.
2. We typically ==change the dimensionality of these vectors from `d` to `d // H`,== where `H` is the number of attention heads. ==This is just to keep computational costs reasonable==. Using this approach, each attention head can learn a unique representational subspace and focus on different parts of the underlying sequence

Finally, theres another detail to consider with multi-headed self-attention: *How do we combine the output of each head?*
There are a variety of different options, but the vanilla implementations of multi-headed self-attention typically do one of:
1. Concatenate the output of each head
2. Linearly project the concatenated output


Because each attention head outputs token vectors of dimension `d // H`, the concatenated output of all attention heads has dimension `d` , the same as the attention layer's input dimension!


# The Decoder-Only Transformer Block
- The decoder-only transformer architecture is comprised of several "blocks" with identical structure that are stacked in sequence. *Within* these blocks, there are two primary components:
	1. Masked ==multi-headed self-attention==
	2. A ==feed-forward== transformation
- Additionally, we surround these components with a *==residual connection==* and a *==normalization layer==*.

### Layer Normalization
- Although high-performance GPUs and advancements in model architectures may make us think otherwise, *training deep NNs hasn't always been easy!* 
- Early attempts at training NNs with many layers were largely unsuccessful due to issues with vanishing, exploding, and unstable gradients. Several advancements have been proposed to address these issues:
	1. ==Better methods of initializing weights== (e.g. [[Xavier Initialization]] or [[He Initialization|Kaiming Initialization]]/[[He Initialization]])
	2. ==Replacing sigmoid activation functions== with [[ReLU]] (This keeps gradients in the activation function from becoming very small)
	3. ==Normalizing== intermediate neural network activations

- Within this section, we will focus on the final advancement mentioned above -- *normalization*!
	- The idea behind normalization ins simple: The intermediate activation values of a deep NN can become ==unstable== (i.e. very large or very small) because we repeatedly multiply them by a matrix of model parameters.
	- To solve this, we *normalize* the activation values between each matrix multiplication, allowing activation values to remain stable over time.

#### Popular Normalization Variants
- Depending on the domain of architecture being used, there are several normalization techniques that we can adopt. 
- The two most common forms of normalization are:
	1. [[Batch Normalization]]
	2. [[Layer Normalization]]

These techniques are quite similar! For each, we just transform activation values using the equation below:
![[Pasted image 20240304150130.png|200]]
The difference between them lies in how we choose to compute the mean and standard deviation!
##### (1/2) Batch Normalization
- As the name indicates, we compute a *per-dimension mean and standard deviation* ==over the *entire mini-batch==!*
- This approach works well, but it's ==limited by the fact that we must process a sufficiently large mini-batch of inputs to get a reliable estimate of the mean and variance==.
- ==This becomes an issue during inference, where processing only a small number of input examples at once is common==.
	- As a result, we must compute a running estimate of the mean and standard deviation during training can be used for inference.
	- Nonetheless, BatchNorm is widely used and is the standard choice of normalization techniques within computer vision applications.

![[Pasted image 20240304150700.png]]

##### (2/2) Layer Normalization 
- Eliminates batch normalization's dependence upon the batch dimension by ==computing the mean and standard deviation over the final dimension of the input==.
- In the case of decoder-only transformers, this means that we compute normalization statistics over the embedding dimension.

##### Affine Transformation
- Normally, layers in deep NNs are also typically combined with an affine transformation.
- It just means that we modify layer normalization as shown in the equation below.
- After normalizing the activation value, we multiply it by a ==constant $\gamma$== , as well as add a ==constant $\beta$ .== Both of these constants are learnable and treated the same as normal model parameters.
- We also see below that layer normalization uses a slightly modified form of standard deviation in the denominator that incorporates a small, ==fixed constant $\epsilon$ to== avoid issues with dividing by zero.

![[Pasted image 20240304151901.png]]

#### Feed-Forward Transformation
- Each decoder-only transformer block contains a pointwise feed-forward transformation.
- The transformation ==passes every token vector within ints input through a small, feed-forward neural network==.
- This neural network consists of ==two linear layers== (with optional bias) that are ==separated by a non-linear activation== function.
![[Pasted image 20240304152040.png]]

- ==The neural network's hidden dimension is usually *larger*== (4x larger, in the case of GPT, GPT-2, and many other LLLMS, than the dimension of the token vector that it takes as input!) than the dimension of the token vector taken as input!

#### Activation Function
- Which activation function to use in an LLM's feed-forward layer is important -- authors have found that the [[SwiGLU]] activation yields very good performance when given a fixed amount of compute. It's used by popular LLMs like [[LLaMA 2]] and [[OLMo]] -- but others (Falcon, [[Gemma]]) us [[GeLU]]
- ![[Pasted image 20240304153025.png]]

#### Residual Connections
- We typically add [[Residual Connection]]s between each of the self-attention and feed-forward sub-layers of the transformer block. 
- The concept of a residual connection was originally proposed in the [[Residual Network|ResNet]] architecture, which is a widely-used an famous convolutional neural network architecture for computer vision tasks like image classifications and object detection.
- Instead of just passing neural network activations through a layer in the network, we:
	1. ==Store the input to the layer==
	2. ==Compute the layer's output==
	3. ==Add the layer's input to the layer's output==
![[Pasted image 20240304153338.png|400]]

Residual connections are a generic idea that can be applied to any neural network layer that doesn't change the dimension of the input!
- ==By adding residual connections, we can mitigate problems with vanishing and exploding gradients, as well as improve the overall ease and stability of the training process==.
Residual connections provide a "==shortcut==" that allow gradients to flow freely through the network during backpropagation.

#### Putting it all together
- ![[Pasted image 20240304153849.png|200]]

- To construct a full decoder-only transformer block, we have to use all of the components that we have talked about so far:
	- Masked, multi-headed self-attention
	- Layer normalization
	- Pointwise feed-forward transformations
	- Residual Connections

# The Decoder-Only Transformer
- Now let's look at a full decoder-only transformer architecture, which is primarily composed of building blocks we've seen so far... (but we'll first need to cover a few extra details, like how we construct the model's input and how we use use the model's output to predict/generate text).

## Constructing the Model's Input
- As outlined previously, the input to a transformer block is expected to be a (batched) sequence of token vectors, usually with the shape `[B, T, d]` 
	- (B: Batch Size. T: Sequence length: d: dimensionality of each token vector in the sequence)
- How do we convert this textual prompt into the sequence of token vectors that is expected?

### Tokenization
- The transformer receives raw text as input.
- The first step is to ==tokenize== the textual input, breaking/converting it into a sequence of discrete words/sub-words. These words and sub-words are commonly called ==tokens==.
![[Pasted image 20240304154628.png]]
- The tokenization process is handled by the model's tokenizer, which uses an algorithm like [[Byte-Pair Encoding]] (BPE), [[SentencePiece]], or [[WordPiece]].
- The tokenizer has a *fixed-size vocabulary* (usually 50-300k unique tokens) that define the set of known tokens that can be formed from a raw sequence of text.
- The ==tokenizer== ==has its own training pipeline== that derives its underlying vocabulary and typically implementing two major functions:
	1. *==Encode==*: Convert a string into a sequence of tokens
	2. *==Decode==*: Convert a sequence of tokens into a string
- ==Tokenization is an oftentimes overlooked aspect of LLM training and usage.==
- Failing to investigate and understand the tokenization process for an LLM is a huge mistake! He recommends [[Andrej Karpathy]]'s recent Youtube Video where he implements BPE from scratch.

### Token Embeddings
- Once we've tokenized our text and formed a sequence of tokens, we need to convert each of these tokens into a corresponding *embedding vector.*
- To do this, we create an *embedding layer*, which is a part of the decoder-only transformer model.
- ==This embedding layer is just a matrix== with `d` columns and `V` rows, where `V` is the size of the tokenizer's vocabulary.
- ==Each token in the vocabulary is associated with an integer index== that corresponds to a row within this embedding matrix. We can ==convert tokens into a `d`-dimensional embeddings by simply looking up the row in the embedding matrix using the token's integer index==.
![[Pasted image 20240304155529.png|400]]
==This embedding layer is trained *during the LLM's training process* similar to any other model parameter!==
- Token embedding matrices *are not fixed, but are learned from data!*
### Position embeddings
- Now, we've converted our raw text into a sequence of token vectors.
- If we did this for our entire *batch* of textual *sequences*, we now have an input of size `[B, T, d]` as expected by our transformer blocks. 
- But there's one final step that we need to do! ==Positional embeddings!==
> 
> *"Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we need to inject some information about the relative or absolute position of the tokens in the sequence!"*
> 
- In studying the self-attention mechanism, we might notice that the position of each token in the sequence is not considered when computing the output!
- The order of words within a sequence of text is obviously important, though -- so we need some way of injecting positional information into the self-attention process!
- ==This was done by adding positional embeddings of dimension `d` to each token within the models input!==
	- Because each position in the sequence has a unique position embedding, the position of each token can be distinguished. 

Similarly to token embeddings, ==we can store position embeddings in an embedding layer and learn them from data during the LLM's training process==. We ==can alternatively generate fixed token embeddings via some rule or equation==

![[Pasted image 20240304160444.png|350]]
Above: an example of a "fixed" token embedding using sine and cosine functions.

These approaches are referred to as =="absolute" positional embedding strategies==, as the embedding being used is determined by the token's absolute position in the sequence. As we'll see later, ==absolute positional embedding strategies fail to generalize to sequences that are longer than those seen during training, which has led to the proposal of more generalizable strategies.==

### The full decoder-only transformer model
- Once we've got the model's input (a batch of sequences of tokens), we can simply pass this input through a sequence of decoder-only transformer blocks (see below):
![[Pasted image 20240304160948.png|500]]
- The total number of transformer blocks depends on the size of the model.
- *Transformer blocks preserve the size of their input*, so the output of the model's body (including all transformer blocks) is a sequence of token vectors that is the same as the input.

Increasing the number of transformer blocks/layer within the underlying LLM is one of the primary ways of increasing the size of the model.

Alternatively, we increase the value of `d`, which increases the size of weight matrices for all attention and feed-forward layers in the model. ((Recall: `d` is the size/dimensionality of each token vector in the sequence, which increases the size of weight matrices for all attention and feed-forward layers in the model))
- As shown below, we typically scale up the size of a decoder-only transformer by simultaneously increasing both:
	1. The number of transformer block layers
	2. The hidden dimension 

Oftentimes, we also also increase the number of heads within each attention layer, but this doesn't impact the # of parameters in the model, assuming that each attention head has a dimension of `d // H`.

#### Classification Head (Constructing the Model's Output)
- Finally, there's one more detail that we have to consider -- once we've passed our input sequence through the model's body, we receive as output a same-size sequence of token vectors.
- To actually *generate text* and predict the next token, ==we convert each token vector into a probability distribution over potential next tokens==!
	- To do this, we add an extra linear layer with input dimension `d` and output dimension `V` (size of vocabulary), which serves as a classification head, to the end of the model (see below):

![[Pasted image 20240304161655.png|450]]

Using this linear layer, we can convert each token vector in our output into a probability distribution over the token vocabulary. From this distribution we can perform next-token prediction!


# Modern Variants of the Architecture
- Now that we can understand the decoder-only transformer architecture, we can look at some of the variants of this architecture being used by modern LLMs.
































