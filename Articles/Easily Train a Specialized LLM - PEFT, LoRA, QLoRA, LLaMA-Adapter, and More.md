#article 
Link: https://cameronrwolfe.substack.com/p/easily-train-a-specialized-llm-peft

-------
Due to the surge of interest in LLMs, AI practitioners commonly as:

> *"How can we train a specialized LLM over our own data?*

But the answer isn't so simple! Recent advances in generative AI are powered by *massive models* that don't easily fit into the GPU memory of most people, who generally have at most ~24GB of GPU memory (unless they're rocking an A100 or multiple GPUs).

We need a training technique and hardware that can handle fine-tuning meaningfully capable models!

How do we fix this? With [[Parameter-Efficient Fine-Tuning]] (PEFT)!
- Instead of training the full model end-to-end, PEFT ==*leaves the pretrained model weights fixed, and only adapts a small number of task-specific parameter during finetuning!*==
- This drastically reduces memory overhead and lets us finetune LLMs with more accessible hardware.


# Background Information

#### Structure of a Language Model
- These models are based on the [[Transformer]] architecture... originally proposed for Seq2Seq machine tasks (summarization, translation, conditional generation, etc.) and contains both an [[Encoder]] and a [[Decoder]].
	- Encoder: Each block has *bidirectional* self-attention and a feedforward layer!
	- Decoder: Each block has *causal* self-attention, *cross-attention*, and a feedforward layer!

![[Pasted image 20240225192445.png]]
- Above:
	- See in the encoder that we have bidirectional self-attention
	- See in the decoder that we have first causal self-attention  and then cross-attention with the encoder.
		- ((Note that the cross-attention is removed in the case of a decoder-only architecture))

## Transformer Variants
- The originally-proposed transformer architecture is usually referred to as a [[Encoder-Decoder Architecture]] transformer (for example, [[T5]]), but two other transformer architectures exist!
	1. [[Encoder-Only Architecture]] Transformers
	2. [[Decoder-Only Architecture]] Transformers
		- Note that the Decoder-Only Architecture removes not only the encoder, but also the cross-attention layers in the decoder, as there isn't an encoder to pay attention to!
![[Pasted image 20240225192900.png]]
- Above:
	- See that T5 is an example of the encoder-decoder architecture, which is great for Seq2Seq tasks like translation
	- See that Encoder-only models are heavily used for information retrieval application (eg in vector search / [[Dense Retrieval]]), with examples in the [[BERT|BERT]] family of models. Models like [[Sentence-BERT|sBERT]] even have semantic meaning to their embeddings.
	- See that the Decoder-only models are used by [[GPT]]-style models, and are great for efficient training/inference via next-token-prediction!

Although GPT-style generative LLMs (i.e. large decoder-only transformers) are very popular today, many types of useful language models exist!
- Encoder-only models can be used for discriminative language understanding tasks (eg classification and retrieval)
- Encoder-Decoder models are great for performing conditional generation (eg summarization, translation, text-to-SQL, etc.)

Transformer Layer Components
- Within each transformer layer, two primary transformations are used:
	1. Self-Attention
	2. Feed-forward transformation
- Many different *styles* of self-attention (either bidirectional or masked/causal) are used:
	1. Bidirectional Self-Attention (used in the encoder)
	2. Masked (Decoder) Self-Attention (used in the decoder)
	3. Feed-Forward Layers

#### How are LLMs trained?
- Modern LLMs are trained in several steps. 
- Encoder-only and encoder-decoder models are trained via a transfer-learning approach that includes self-supervised pretraining and finetuning on a downstream task
- Generative GPT-style Decoder-only LLMs follow the multi-step training procedure shown below (we'll focus mostly on this)
![[Pasted image 20240225194050.png]]
Pretraining
- All language models are pretrained using some form of a self-supervised learning objective.
- For example, generative language models are usually pretrained using a *next-token prediction* objective, while encoder-only and encoder-decoder models commonly use a *Cloze* task.

Self-supervised learning techniques are great because they don't rely on manual human annotation of the data, as supervised learning does. Instead, *the labels used for supervision in self-supervision are already present within the data itself!* We use things like Cloze tasks to mask and predict tokens in a sequence.

We collect massive datasets of unlabeled text (eg by scraping the internet) to use for self-supervised pretraining. Due to the scale of data available, the pretraining process is quite computationally expensive. As a result, the resulting pretrained models are often called [[Foundation Model]]s and are used as starting points for training further-specialized models for downstream tasks.


#### Alignment
- The alignment process is only applicable to generative LLMs, and refers to 
	- ((I could frankly see some alignment being useful for embedding models too...))
- After we pretrain the model, we perform additional fine-tuning in order to align it to perform according to the expectations that we/our users have. We do this fine-tuning via two main techniques, usually in sequence:
	1. [[Supervised Fine-Tuning]] (SFT)
	2. [[Reinforcement Learning from Human Feedback]] (RLHF)
![[Pasted image 20240225195829.png]]
Above:
- SFT trains the language model over a set of high-quality reference outputs using a next-token-prediction objective, and the LLM learns to mimic the style and format of the responses within this dataset
	- ((Note that they said style and format; it's said that the large majority of the *knowledge* of the model is gained during pretraining, whereas alignment via SFT is more about style of responses))
- RLHF collects feedback (from humans) and uses it to train a reward model, which is then used as a training signal.

#### Finetuning
- After both pretraining and alignment (an optional step), we have a generic foundation model that can be used as a starting point for many tasks!
- But to solve our *specific task*, we should adapt the language model to this task, usually by finetuning (i.e. further training on data from a task).
	- The combination of pretraining followed by finetuning is commonly referred to as [[Transfer Learning]], and has many benefits:
		1. Pretraining is expensive, but only occurs once, and often by some large company. It can be used as a starting point for many tasks.
		2. In most cases, we can download a pretrained model online for free, and just focus on finetuning!
		3. Finetuning a (large) pretrained model usually performs better than training a (likely smaller) model from scratch on your own, despite requiring less finetuning data and trainingtime.

So finetuning is performant and computationally cheap relative to training your own model, likely.

Although finetuning is computationally cheap relative to pretraining or training from scratch, it can still be expensive! let's focus on the finetuning process and figure out we can make it more efficient from a compute and memory perspective, allowing practitioners to train specialized LLMs with fewer/smaller GPUs.


## Model Quantization
- The idea of [[Quantization]] in deep learning refers to quantizing (i.e. converting to a format that users fewer bits, for a less-granular representation) of the model's *activations and weights* during the forward and/or backward pass.
	- Because we use less precision to represent our weights, each weight takes up less space. Because a model is primarily a large file of such floats (representing weights), this means that we can fit a larger model into our GPU!
- ==We usually do not sacrifice performance when performing such quantization== -- we basically get these efficiency benefits for free!
	- ((I'm not sure that this is true...))

the literature for quantization for deep learning is vast, but proposed techniques can be (roughly) categorized into *three primary areas!*
1. ==Quantized Training==
	- Perform quantization *during training* to make the training process more efficient.
	- ((Assumedly the model stays "quantized" for inference, making that more efficient too))
	- ((I wonder if a model that is *trained quantized* is smarter than one that *gets quantized* afterwards! It would make sense to me... maybe?))
2.  ==Post-Training Quantization==
	- Quantize a model's weights *after training* to make inference more efficient.
	- ((Does this minorly "lobotomize" the model?))
3. ==[[Quantization-Aware Training]]==
	- Train in a manner that makes the model *more amenable* to post-training quantization, thus avoiding performance degradations due to post-training quantization!


Side note: One of the author's favorite applications of quantization is [[Automatic Mixed-Precision Training]] (AMP)
- With AMP, we can train arbitrary neural networks using lower precision in *specific portions* of the network architecture.
- *==This approach can cut training time in half while achieving comparable performance== without any code changes!*

# Adaptation of Foundation Models
- Finetuning is more efficient than training from scratch -- the model converge more quickly, performs better, and requires less data to perform well.
![[Pasted image 20240225204040.png]]
- Above:
	- BERT (encoder-only) and T5 (encoder-decoder) are pretrained using a Cloze objective and finetuned to solve a variety of downstream tasks.
	- Generative LLMs follow a similar approach, but pretraining is achieved using a next-token-prediction objective.
- In either case, the model can be finetuned using a variety of techniques (SFT, RLHF, Task-Specific Finetuning) 

## In-Context Learning
- Pretrained LLMs have rudimentary ability to solve problems via prompting, but ==the alignment process improves language models' instruction-following capabilities ((and steerability))==
- In-context learning refers to a single model learning to solve a variety of problems by writing task-specific prompts (textual inputs to the language model), rather than finetuning the model on each task.

Because no finetuning is required, and a good foundation model can just be used  tasks, in-context learning is by far the easiest way to adapt an LLM to solve a downstream task. [[In-Context Learning]] should always be the first stop for adapting a model to some new task -- but its performance will lack behind that behind finetuning,

## Full Finetuning
- If in-context learning doesn't perform well enough for our needs, our next option is to finetune the model on our dataset.
- The approach works well, but has several downsides:
	- The finetuned model has as many parameters as the original model, which means that finetuning large models is extremely burdensome/expensive.
	- We must store all the model's parameters each time we either retrain the model or train it on a new/different tasks.
	- Training the full model is both compute-intensive and memory-intensive
	- E2E training might require more hyperparameter tuning or data to avoid overfitting and achieve the best possible results.


==Full-finetuning is burdensome if== we:
1. Want to frequently retrain the model ((Maybe because of distribution shift))
2. Are finetuning the same model on many different tasks, in which case we end up with N "copies" of an already-large model

Storing and deploying many independent instances of a large model can be challenging!
![[Pasted image 20240225211403.png]]

Finetuning large models end-to-end is not cheap and/or easy, by any means!

# Adapter Layers and Prefix Tuning
- Given these limitations, we ask -- how can we finetune an LLM in a more compute/data efficient manner that still maintains maximal performance of the model?
- A variety of research boils down to ONE CORE IDEA:
	- ==Only adapting a small portion of the model's parameters (or some new, added parameters) during finetuning!==
- Each finetuned model only has a small number of task-specific parameters that should be stored/loaded...

Prior to the proposal of [[Low-Rank Adaptation|LoRA]], two main parameter-efficient finetuning techniques were used:
1. Adapter Layers
2. [[Prefix Tuning]]




### ==Adapter Layers==
![[Pasted image 20240225211738.png]]
Above:
- Authors inserted ==two extra adapter blocks into each transformer block==!
- Each adapter block is a ==*bottleneck-style* feedforward model== that:
	1. *Decreases* the dimensionality of the input via a (trainable) linear layer
	2. Applies a non-linearity
	3. *Increases* the dimensionality of the input via another (trainable) linear layer

Simply put, ==the adapter blocks are extra trainable modules inserted into the existing transformer block== (after both attention and feedforward layers)
- This small number of new parameters can be finetuned while *keeping the weights of the pretrained model fixed!*
- As such each of the N finetuned versions of the model that you produce only have a small number of task-specific parameters that are uniquely associated with it.

Several variants of adapter layers have been proposed that are *more efficient*, and go beyond even language models!

> Prefix tuning keeps language model parameters frozen, but optimizes a ==small, continuous task vector==.
> Prefix tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were virtual tokens!

[[Prefix Tuning]]
- Another PEFT alternative is prefix tuning, which keeps the LM's parameters frozen and *==finetunes a few (trainable) token vectors that are added to the beginning of the model's input sequence==*!
	- ((So it's basically some learned gobbledygook that you stick at the beginning of your prompt that helps you do some specific task? eg "F3@faLm0RrSAn33n" helps you do task X, or something? Or does it have to be tokens in the vocabulary? Seems so! See picture below and explanation))
- These token vectors can be interchanged for different tasks, see below:
![[Pasted image 20240225214956.png]]
Above:
- In other words, prefix tuning adds a few extra token vectors to the model's input. ==However, these added vectors don't correspond to a specific word or token -- we train the entries of these vectors just like normal model parameters==!
- We see that authors freeze all model parameters and train only a small subset of prefix token vectors added to the model's input layer for each tasks. These trainable prefix tokens can be shared across a task! 

#### What's the problem?
- Both [[Prefix Tuning]] and Adapter Layers reduce the compute requirements, memory overhead, and number of trainable parameters associated with finetuning an LLM. 
-  But these approaches don't come without limitations!
	- ==Adapter layers *add layers== to the underlying model* that must be processed sequentially, ==resulting in extra inference latency and slower training==!
	- ==Prefix Tuning== is oftentimes ==difficult to optimize== (the finetuning is less stable), reduces the available context window, and doesn't monotonically improve performance with respect to the number of trainable parameters.

*Low-Rank Adaptation (LoRA) will aim to eliminate such issues while maintaining performance that is comparable to full fine-tuning!*

# Finetuning LLMs more efficiently!
- Given our discussion of finetuning techniques so far, it should be clear that we need a finetuning approach that is *not only parameter efficient*, but ALSO:
	1. Compute Efficient: It shouldn't meaningfully slow down training or inference!
	2. Memory Efficient: It shouldn't add a large number of weights that we need to keep track of in memory!
	3. Easy to Deploy: We shouldn't have to deploy several copies of an LLM for each task we want to solve!

[[Low-Rank Adaptation]] (LoRA) checks all of these boxes!
- It effectively lowers the barrier of entry for finetuning specialized LLMs, achieve performance that is comparable to end-to-end finetuning, can easily switch between specific versions of a model, and have no increase in inference latency.

## LoRA: Low-Rank Adaptation of Large Language Models
- When we finetune an LM, we modify the underlying parameters of the model. 
- To make this idea more concrete:
![[Pasted image 20240225220201.png]]
The core idea behind LoRA is to model this update with a *low-rank decomposition*, implemented as a pair of linear projections!

----
**Aside: What's a low-rank decomposition?**
- In linear algebra, a "low-rank" matrix is one that has repetitive rows or columns.
- Any non-null matrix can be written as a product of two matrices; e.g. W = AB, where `W` is of size m\*n , `A` is of size m\*r, and `B` is of size r\*n 
- This is called a rank factorization -- in LoRA, we call this matrix product a low-rank decomposition of W, as the rank of AB is at most `r`.
	- And "rank" measures the dimension of the vector space generated or spanned by its columns or rows
----

Again, the core idea behind LoRA is to model this update to the model's parameters with a low-rank decomposition.

==LoRA leaves the pretrained layers of the LLM fixed and injects a trainable rank decomposition matrix into each layer of the model== -- see below:
![[Pasted image 20240225220733.png]]

Rank decomposition matrix
- ==Put simply, the rank decomposition matrix is just *two linear projections* that reduce and then restore the dimensionality of the input!==
	- The output of these two linear projections is *added* to the output derived from the model's pretrained weights.
	- The updated layer formed by the addition of these two parallel transformations is formulated as shown below:

![[Pasted image 20240225221122.png]]
Above:
- Instead of the finetuned weights being {the pretrained weights plus the weight update from full-model finetuning}, we say that the finetuned weights are {the pretrained weights plus some low-rank approximation of the ∆W matrix}
- The matrix product of `AB` has the same *dimension* as the full ∆W finetuning update matrix!

![[Pasted image 20240225221811.png]]
((Da Sam Classic))

So the *matrix product* `AB` has the same dimension as the full finetuning update ∆W matrix!
- ==Decomposing the update into a *product of two smaller matrices* ensures that the update is low-rank, and significantly reduces the number of parameters that we have to train!==
- LoRA *ONLY* optimizes the rank decomposition matrix, yielding a result that *APPROXIMATES* the update derived from full finetuning!

We initialize `A` with random, small values, while `B` is initialized at zero. This ensures that we begin the finetuning process with the model's original, pretrained weights!

> *We roughly recover the expressiveness of full fine-tune by setting the LoRA rank `r` to the rank of the pre-trained weight matrices*

((Above: Wait, but then we aren't saving on "cells", in my picture :( What gives?))

Increasing `r` improves LoRA's approximation of the full finetune update, but ==incredibly small values of r suffice in practice, allowing us to significantly reduce compute and memory costs with minimal impact on performance!==
- For example, ==we can use LoRA to finetune GPT-3 using only 0.01$ of total parameters, and still achieve performance comparable to that of finetuning!==

### Scaling Factor
- Once the low-rank update to the weight matrix is derived, we ... *scale* it by a factor of $\alpha$ prior to adding it tohe model's pretrained weights.
![[Pasted image 20240225225626.png]]
- The default value of the scaling factor is 1, meaning that the pretrained weights and the low-rank update are weighted *equally* when computing the model's forward pass -- but the value of $\alpha$ can be changed to balance the importance of the pretrained model and new task-specific adaptation.
	- Recent empirical analysis indicates that larger values of $\alpha$ are necessary for LoRA with a higher rank (ie higher rank = larger alpha)


#### Comparison of LoRA to Adapter Layers
- Two notable differenceS:
	- No non-linearity between the two linear projections
	- The rank decomposition matrix is injected into an existing layer of the model, instead of being sequentially added as an extra layer.
==As a result, LoRA has no added inference latency compared to the original pretrained model, which is an advantage over the Adapter Layer strategy!==
- When deploying a finetuned LoRA model into production, we can directly compute and store the updated weight matrix derived from LoRA -- the structure of the model is identical to the pretrained model, the weights are just different!
- As a result, ==we can "switch out" LoRA modules by:== ((indeed tools like [[LoRAX]] are built around this idea))
	1. Subtracting the LoRA update for one task from the model's weights
	2. Adding the LoRA update for another task to the model's weights
- In comparison, switching between *models* that are finetuned end-to-end on different tasks requires loading *all* model parameters in and out of memory, creating a significant I/O bottleneck!
- LoRA's efficient parameterization of the weight update derived from finetuning makes switching between tasks efficient and easy.

![[Pasted image 20240304123308.png]]
- Above:
	- Why does LoRA work? Why can we train a small number of parameters and get benefit in the model?
	- Because large models tend to have a low intrinsic dimension! It means that the weights of very large models *tend to be low rank* -- not all these parameters are necessary!
		- ((See [[Jonathan Frankle]]'s lottery ticket hypothesis, etc.))
	- As a result, we should expect that the weight update derived from finetuning could also have a low intrinsic dimension!


#### LoRA for LLMs
- The general idea proposed by LoRA can be applied to any type of dense layer for a NN (more than just transformers)!
- ==LoRA is used to update the query and value matrices of the attention layer in particular, which is found in experiments to yield the best results. The feedforward module and pretrained weights are kept fixed.== 

#### ==Benefits of LoRA==
1. A ==single pretrained model can be shared by several (much smaller) LoRA modules== that adapt it to solve different tasks, simplifying the deployment and hosting process.
2. LoRA modules can be "baked in" to the weights of a pretrained model to avoid extra inference latency, and we can quickly switch between different LoRA modules to solve different tasks.
3. When finetuning an LLM with LoRA, we only have to maintain the optimizer state for a very small number of parameters, which significantly reduces memory overhead and ==allows finetuning to be performed with more modest hardware==.
4. Finetuning with LoRA is significantly faster than e2e finetuning (~25% faster in the case of GPT3)

#### LoRA in practice
- LoRAs have become popular because they're useful tools for AI practitioners; finetuning LLMs for our desired application is much easier than before! We don't need tons of massive GPUs, and the finetuning process is efficient, which makes it possible for almost anyone to train a specialized LLM on their own data.
- After downloading a pretrained model (eg LLaMA-2), we should get a dataset used for finetuning, like these popular [[Instruction-Tuning]] datasets:
	1. Alpaca
	2. Dolly
	3. LongForm
	4. LIMA
	5. RedPajama

We also notice that LoRA is orthogonal to most existing parameter efficient finetuning techniques, meaning we can use both at the same time! Why? Because LoRA doesn't directly modify the pretrained model's weight matrices, and instead learns low-rank update to these matrices that can be fused with pretrained weights to avoid inference latency.


#### LoRA Variants
- [[Quantized Low-Rank Adaptation|QLoRA]] (adds model quantization to reduce memory usage during finetuning while maintaining roughly an equal level of performance, saving memory at the cost of slightly-reduced training speed.)
- QA-LoRA (further reduces computational burden... separately quantizes different groups of weights in the model -- it finetunes in a quantization-aware manner.)
- LongLoRA (attempts to cheaply adapt LLMs to longer context lengths using a LoRA-based finetuning scheme)
- S-LoRA (aims to solve the problem of deploying multiple LoRA modules that are used to adapt the same pretrained model to a variety of different tasks... Lets you serve 1,000s of LoRA modules on a single GPU and increase throughput of prior systems by 4x.)
- LLaMA-Adapter (follows an approach similar to prefix tuning that adds a set of learnable task-adaptation prompts to the beginning of the transformer's sequence at each layer -- preserves the models' pretrained knowledge and allows adaptation to new tasks and instruction following to be learned with minimal data.)
- LQ-LoRA
- MultiLoRA
- LoRA-FA
- Tied-LoRA
- GLoRA

# Takeaways
- LoRA is the most widely-used practical tool for creating specialized LLMs, and it democratizes the finetuning process by significantly reducing hardware requirements.
- Affordable finetuning. Hundreds of dollars to finetune LLaMA-2.
- An ecosystem - a variety of extensions, alternatives, and practical tools, eg QLoRA, which combines LoRA with model quantization.
