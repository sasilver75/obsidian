#article 
Link: https://www.artfintel.com/p/transformer-inference-tricks

-----

Let's talk about a number of optimizations that can be done to make inference for transformers either faster or more efficient.

# KV Cache
- By far the most common (and important) optimization for a ==decoder== is a [[KV Cache]].
- ==In a decoder model, the keys and values will be identical for the prompt for every iteration of decoding; the query is the only thing that changes.==
	- As a result, we can cache the KV matrices for a sequence and reuse it for every token in the sequences, avoiding a lot of compute.

Inside the attention mechanism, we're able to go from multiply two tensors of shape (batch, context_length, feature_dimension) to multiplying a query tensor of share (batch, 1, feature_dim) with your KV tensors of shape (batch, context_length, feature_dim).

Consequently, sampling is no longer quadratic in complexity, allowing us to get decent decoding (sampling) performance with longer context lengths.

In practice, this also accrues added complexity in your implementation, since you're now storing state, and not just running pure functions -- memory requirements are already stressed by models themselves (especially on consumer cards).

# Speculative Decoding
- [[Speculative Decoding]] is a (family of) techniques that are used when you have excess compute capacity, typically in the local inference setting.
- It exploits the property of modern accelerators (GPUs) whereby it takes the same amount of time to run inference on a *batch* of data as it does to run inference on a single datapoint!
	- As a result, many techniques have cropped up to exploit this, such as [[Beam Search]], [[Monte-Carlo Tree Search]], or Speculative decoding

In Speculative decoding, one has two models:
1. A small, fast one
2. A large, slow one

Because the inference speed of a modern decoder is directly proportional to the number of parameters, we can run multiple inferences with the smaller one in the time it takes a large model to run a single inference!

Modern decoders use autoregressive sampling, where we, to sample a sequence of N tokens, run N inferences of our model, each time consuming the results of the previous inference. 

In speculative decoding, we run two models in parallel:
1. The fast one runs a batch of inference and guesses which tokens the big one will predict.
2. It compounds these guesses.
3. In the meantime, the big model is running in the background, checking that the smaller model recorded the same results.
4. The smaller model is able to make many guesses in the same time that the big model is able to make *one*. But given that we have some spare compute capacity, the big model is able to evaluate all of the guesses in parallel! As such, the only place where we pay the sequential cost of generating a sequence is for the smaller model.

The key idea behind having a smaller model make predictions in sequence, in the shadow of the larger model, is that it can discover diverse and potentially better future solutions that the larger model might have missed.

((The larger model in some way takes into account the predictions from the smaller model))

![[Pasted image 20240413183806.png]]

The major disadvantage of speculative to speculative decoding is that it requires a "draft" model that is able to predict the output of a larger model, and you have to have both models in memory on the same machine (or on the same node in a multi-GPU setting).
- This adds complexity
- This adds additional work, as you have to be training two models.
- Performance benefits are limited by how accurately the smaller model is able to predict the behavior of the larger model.
	- ==But wait, if the smaller model were actually able to predict the behavior of the larger model, we'd just use it!==
	- As a result, there's a fundamental gap in how well speculative decoding can perform.

HuggingFace has claimed that it typically ==doubles the decoding rate==, and the paper itself claims a 2-3x improvement.

![[Pasted image 20240413211952.png]]

----
A technique came out recently that tried to improve on this by having the model generate n-grams, and recursively match them, without requiring a draft model.

A technique called ==Jacobi decoding== is another potential improvement over greedy decoding; how it works is that , at every point where you generate a token, you generate *n* tokens, making a guess as to the entire sequence. Then you verify this against your previous guess -- if the two match, you accept your guess -- this can enable latency improvements with no downside, since, in the worst case, it devolves into greedy decoding.

==Lookahead== decoding improves on this further by keeping the n-grams that have been generated through the decoding process, and trying to use them as guesses. Since there's a high correlations between the text that's *been generated* and the text that *will be generated*, this also has a possibility to improve latency dramatically, with minimal cost.

-------

# Effective Sparsity

In a decoder transformer, the beating heart of the model is the attention mechanism, summarized in the attention equation:
![[Pasted image 20240413190130.png]]

The softmax produces many values that are *not* very large:

![[Pasted image 20240413190205.png]]

Consequently, we are multiplying the values tensor (V in the attention equation) by a tensor that is mostly ~zerores. As a result, the output of the attention mechanism has a lot of zeroes -- up to 97%! There's a lot of sparsity. (Similarly after ReLUs in MLPs, we have a lot of sparsity)

Now, unfortunately, it's kinda tough to actually make use of this. If we have sparsity in the weights, there's a lot of work that can be done there through structured sparsity (e.g. torch.sparse), but it's not actually clear how much current systems can make use of this sparsity.

One optimization:
- If an activation is zero, you can just skip loading the weights that correspond to that activation, and skip the corresponding computation. This isn't really supported in mainstream tensor programs.

The reason for this is that the activations are a functions of each token, and thus, so too is the effective sparsity, causing it to be randomly distributed over the tokens. 

....


# Quantization

It's one of the better-known tricks.
- Most of the literature, such as the GPTQ paper [Link](https://arxiv.org/abs/2210.17323), was done with models that aren't close to SoTA, since the big labs aren't publishing.
	- "Accurate Post-Training Quantization for Generative Pre-Trained Transformers"
	- A new one-shot weight quantization method based on approximate second-order information that is both highly-accurate and highly-efficient... GPTQ can quantize GPT models with 175 billion params in ~ 4 GPU hours, reducing bitwidth down to 3 or 4 bits per weight with negligible accuracy degradation.

A lot of hobbyists are blinded by the appeal of being able to run larger models on consumer hardware, and so they get really excited about quantization.

Recent work like the k-bit inference scaling laws paper ran an incredible number of experiments across a family of LLM architectures, reporting how allocating your bits differently affects performance.
- They studied the tradeoff between having a model with N parameters at a given level of precision vs having a model with 2N parameters and half the precision -- the results were impressive, being almost indistinguishable from no penalty for quantizing (at least for 4 or more bits).

![[Pasted image 20240413191219.png]]
They found, basically, that you can go down to 4 bits without any penalty -- ==there's almost no tradeoff from quantizing!==

You can run a 4x smaller model without a significant drop in performance.
As inference performance on modern accelerators is equal to the number of bits you process, this is great (you can get Nx more operations per second, when using Nx less precision).

My conclusion then, is to use the recommendations from the k-bit inference paper (https://arxiv.org/abs/2212.09720).

However, FP8 seems to be the lowest precision level floating point format natively supported by modern accelerators, and even then, support is limited! 
- Author recommends the lowest quantization being FP8 as the lowest precision level floating point format for production workloads.
	- (It seems that FP8 is strictly better than INT8)



