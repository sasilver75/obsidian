#article 
Link: https://huggingface.co/blog/moe

---

With the release of [[Mixtral]] 8x7B, a class of transformer has become the hottest topic in the open AI community: [[Mixture of Experts]] (MoE).
Let's take a look at:
- Their building blocks
- How they're trained
- Tradeoffs to consider while serving them for inference


# TLDR:
- MoEs are pretrained much faster, compared to dense models ((??))
- MoEs have *faster inference* compared to dense models *with the same number of parameters*
- MoEs require *high VRAM*, since all experts still have to be loaded in-memory
- MoEs face many challenges during fine-tuning, though recent work with MoE instruction-tuning is promising ((This is where we should remember that this is from the end of 2023))

----

# What is a Mixture of Experts (MoE)?

- We know that scale of a model is one of the most important predictors of model quality; ==given a fixed computing budget, training a larger model for fewer steps is better than training a smaller model for more steps.==

- MoE models can enable models to be pretrained with far less compute, meaning you can dramatically scale up the model or dataset size with the same compute budget as a dense model. In particular, an MoE model should achieve the same quality as its dense counterpart much faster during pretraining.

So what is an MoE model?
1. ==Sparse MoE layers== are used instead of dense FFNN layers
	- MoE layers have a certain number of "experts" (eg 8) where each expert is a neural network.
	- Generally these FFNNs within an MoE layer have identical architectures, but there's not reason that they need to be -- they could be other architectures, or even an MoE itself, leading to a hierarchical MoE!
2. A ==gate network or router==
	- This determines which tokens are sent to which experts
	- We can send a token to more than one expert; how to route tokens is one of the big decisions when working with MoEs.
	- The Router itself is composed of learned parameters and is pretrained at the same time as the rest of the network.
![[Pasted image 20240415130323.png]]


==RECAP: In MoEs, we replace each vanilla FFNN layer with an MoE layer, which contains a router that routes tokens to one or more "expert" FFNNs within the layer.==

MoEs also come with ==challenges==:
1. Training: MoEs enable more compute-efficient pretraining, but they've ==historically struggled to generalize during fine-tuning, leading to overfitting==
2. Inference: Although an MoE might have many parameters, only some of them are used during inference -- this leads to *faster inference* compared to a *dense* model with the same number of parameters, however ==all parameters need to be loaded in RAM, so memory requirements are high==.


# A Brief History of MoEs
- Roots come from the ==1991 paper *Adaptive Mixture of Local Experts* (Hinton, Jordan)== -- this idea was similar to ensemble methods -- we'd have a supervised procedure for a system composed of separate networks, each handling a different subset of training cases and specializing in a different region of input space. A gating network determined weights for each expert. During training, both the expert and the gates are trained.
- Between 2010 and 2016, two different research areas contributed to later MMoE advancement:
	1. ==Experts as components==
		- In a traditional MoE setup, the *whole system* comprises a gating network and multiple experts -- Later work has reimagined MoEs as *components* of deeper networks.
	2. ==Conditional computation==
		- Traditional networks process all input data through each layer; We now dynamically activate or deactivate components based on the input token.

Shazeer et al in 2017: ==Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer==
- Scaled this idea to a 137B LSTM by introducing sparsity, allowing us to keep very fast inference even at high scale. This worked, but there were still some challenges in the arch (eg training instability).

# What is Sparsity?
- Sparsity uses the idea of conditional computation -- while in *dense* models, *all* parameters are used for the inputs, **sparsity** allows us to only run *some* parts of the whole system.
	- This has led to thousands of experts being used in MoE layers

This introduces some challenges:
1. Although large batch sizes are usually better for performance, ==batch sizes in MoEs are effectively reduced, since each token flows through some subset of active experts.==
	- If a batched input consisted of 10 tokens, we might imagine that 5 tokens went to Expert A, 3 to Expert B, and 2 to Expert C. We still make a gradient update after (eg) each batch, so Expert C is getting quite a noisy update!

$y = \sum_{i=1}^n G(x)_i E_i(x)$ 

In one setup (above), all experts are ran for all inputs, and we take a weighted average based on the router "confiedence" for each expert.

But what does a typical gating function look like? ==In most traditional setups, our gating function looks like a simple linear layer with a softmax function.==

$G_{\sigma} = Softmax(x \cdot W_g)$   

Shazeer's work explored other gating mechanisms too, such as ==Noisy Top-k Gating==, which introduces some (tunable) noise and keeps the top-k values:
1. Add some noise
$H(x)_i = (x \cdot W_g)_i + StandardNormal() \cdot Softplus((x \cdot W_{noise})_i)$((Above: I don't know what SoftPlus is? Ah, it appears that it's a [smooth approximation](https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html) to the ReLU function. Why do we use noise though?)) 

2. We then only pick the top k

![[Pasted image 20240415133252.png]]

3. We then apply the softmax, making the whole equation:

$G(x) = Softmax(KeepTopK(H(x), k))$ 



By using a low enough $k$ , i.e. one or two, we can train and run inference faster than if many experts were activated! 
- But why not just select the top single expert?
- The ***initial conjecture*** was that routing to MORE than one expert was NEEDED in order to have the gate learn how to route to different experts, so at least two experts had to be picked.

***Why do we add noise?***
- That's for load balancing, below!


# Load balancing tokens for MoEs
- As discussed before, if all our tokens are sent to just a few popular experts, that makes training inefficient!
- ==In normal MoE training, the gating network converges to mostly activate the same few experts 😢==
	- This self-reinforces, as favored experts are trained quicker, and hence selected more!
	- To mitigate this, ==we add an auxiliary loss to encourage giving all experts equal importance!== This loss ensures that all experts receive roughly similar numbers of training examples.
	- ((Assumedly the noise is part of this))


# MoEs and Transformers
- Transformers are a very clear case that scaling up the number of parameters improves the performance, so it's not surprising that Google explored this with their [[GShard]] paper, exploring scaling up transformers beyond 600B parameters.
	- GShard replaces every other FFN layer with an MoE layer using top-gating in both the encoder and decocder.
	- This is quite beneficial for large-scale computing!
		- ==When we scale up to multiple devices, the MoE layer is shared across devices, while all other layers are replicated== -- We'll talk more about this later.


To maintain a balanced load and efficiency at scale, the GShard authors introduced a couple of changes in addition to an auxiliary loss similar to the one discussed in the previous section:
1. ==Random Routing:==
	- In a top-2 setup, we always pick the top expert, but the *second* expert is picked with probability proportional to its weight!
2. ==[[Expert Capacity]]==:
	- We can set a threshold of how many tokens can be processed by one expert!
	- If both experts are at capacity, the token is considered ==*overflowed*==, and sent to the next layer via residual connections (or is dropped entirely, in other projects).
	- ==This is one of the most important concepts for MoEs==
	- What is expert capacity needed?
		- ==Since all tensor shapes are statistically determined at compilation time, but we cannot know how many tokens will go to each expert ahead of time, we need to fix the [[Capacity Factor]].==


# Switch Transformers
- Although MoEs showed a lot of promise, they struggled with difficult training and fine-tuning instabilities!
	- [[Switch Transformer]] is a very exciting work that deep dives into these topics!
- The authors ==released a 1.6T parameter MoE on hugging face with 2048 experts==, which you can even run with the `transformers` library (if you have the space!)
- Just as in GShard, the authors replaced the FFN layers with MoE layers. The Switch Transformer layer receives two inputs (two different tokens) and has four experts. ((?))
- Switch Transformers uses a simplified *==single expert strategy!==*
	- The router computation is reduced
	- The batch size of each expert can be at least halved
		- ((Wouldn't batch size increase, if we're only selecting one expert?))
	- Communication costs are reduced
	- Quality is preserved

Switch Transformers also explore the concept of ==Expert Capacity==:
![[Pasted image 20240415135547.png]]

This capacity above evenly divides the number of tokens in the batch across the number of experts -- if we use a capacity factor *greater* than one, we provide a buffer for when tokens are not perfectly balanced!
- Increasing the capacity will lead to more expensive inter-device communication, so it's a trade-off to keep in mind. In particular, Switch Transformers perform well at low capacity factors (1-1.25).

Switch Transformers also revisit and simplify the load-balancing loss mentioned above -- for each Switch layer, the auxiliary loss is added to the total model loss during training... This loss encourages uniform routing and can be weighted using a hyperparameter.

The authors also experiment with selective precision, such as ==training the experts with `bfloat16` while using full precision for the rest of the computations==.
- Lower precision reduced communication costs between processors, computation costs, and memory for storing tensors.
- The initial experiments in which both the experts and the gate networks were trained in `bfloat16` yielded more unstable training.
This was in particular, due to the router computation -- as the router has an exponentiation function, having higher precision is important. To mitigate these instabilities, full precision was used for the router as well.


Switch Transformers uses an [[Encoder-Decoder Architecture]], in which they did a MoE counterpart of T5.

The later [[GLaM]] paper explored pushing up the scale of these models by ==training a model matching GPT-3 quality using 1/3 of the energy.==
- The authors focused on decoder-only models and few-shot and one-shot evaluation rather than fine-tuning.
- They used Top-2 routing and much larger capacity factors. In addition, they explored the capacity factor as a metric that one can change during training and evaluation, depending on how much computing one wants to use.


# Stabilizing training with router Z-Loss
- The balancing/auxiliary loss that previously discussed to make routing among experts roughly uniform can lead to instability issues.
- ==We use many methods to stabilize sparse models at the expense of quality==
	- [[Dropout]] improves stability but leads to loss of model quality.
	- On the other hand, adding more multiplicative components improves quality but decreases stability.
- Router ==Z-Loss== , introduced in ==ST-MoE==, significantly improves training stability WITHOUT QUALITY DEGRADATION by penalizing *large logits* that enter the gating network. This loss encourages absolute magnitude of values to be smaller, so roundoff errors are reduced, which can be quite impactful for exponential functions such as the gating.

# What does an expert learn
- The ST-MoE authors observed that encoder experts specialize in a group of tokens or shallow concepts. We see that we end up with a punctuation expert, a proper noun expert, etc.
- On the other hand, the decoder experts have less specialization -- we could IMAGINE that we might end up with an expert specializing in (eg) a language, but the opposite happens -- due to token routing and load balancing, there is no single expert specialized in any given language.


How does the number of experts impact pretraining?
- ==More experts leads to improved sample efficiency and faster speedup, but there are diminishing gains (especially after 256 or 512), and more VRAM will be needed for inference.==


# Fine-Tuning MoEs
- The overfitting dynamics are very different between dense and sparse models.
- ==Sparse models are more prone to overfitting, so we can explore higher regularization (eg dropout) within the experts themselves==
	- We could have one dropout rate for the dense layers, and another, higher dropout rate for the sparse layers.

- One question is whether to use the auxiliary loss for finetuning
	- The ST MoE authors experimented with turning off the auxiliary (balancing) loss during fine-tuning, and the quality wasn't significantly impact, even when up to 11% of the tokens were dropped.
		- ==In fact, token dropping might be a form of regularization that helps prevent overfitting!==

==Switch Transformers observed that at at fixed pretrain perplexity, the sparse model does *worse* than the dense counterpart in downstream tasks, especially on reasoning-heavy tasks like SuperGLUE. On the other hand, for knowledge-heavy tasks like TriviaQA, the sparse model performs disproportionately well.==

Authors experimented with freezing weights during finetuning too:
1. Freezing everything but the MoE layers and training caused a huge performance drop ❌
2. ==Freezing the MoE layers only seemed to work almost as well as updating all the parameters! This helped speed up and reduce memory usage for fine-tuning. ✅==
	- This was counter intuitive, since 80% or so of the parameters are in the MoE layers (in the ST-MoE project).

One last thing to note: Sparse MoEs tend to benefit more from smaller batch sizes and higher learning rates.


A recent paper, MoEs Meets Instruction Tuning (July 2023) performs experiments doing:
1. Single task fine-tunung
2. Multi-task instruction-tuning
3. Multi-task instruction-tuning followed by single-task fine-tuning
The authors got results indicating that MoEs might benefit much more from instruction-tuning than dense models, and that they benefit more from a higher number of tasks.


# When to use sparse MoEs vs Dense Models?
- Experts are useful for *high-throughput scenarios with many machines!*
	- Given a fixed compute budget for pretraining, a sparse model will be more optimal.
	- For a low-throughput scenarios with little VRAM (eg embedded), a dense model will be better.

Note: You can't directly compare the number of parameters between sparse and dense models, as both represent significantly different things.

# Making MoEs go brrrr 🏎️💨


### Parallelism
A brief review:
1. [[Data Parallelism]]: The same *weights replicated* across *all cores*, with the *data partitioned* across cores. ((Weights replicated; Data partitioned))
2. [[Model Parallelism]]: The *model itself is partitioned across cores*, while the data is *replicated* across cores. ((Weights partitioned; Data replicated))
3. Model *AND* Data Parallelism: We partition the model and the data across cores. Different cores process different batches of data.
4. [[Expert Parallelism]]: *Experts are placed on different workers*. If combined with data parallelism, each core has a different expert and the data is partitioned across all cores

With expert parallelism, experts are placed on different workers, and each worker takes a different batch of training samples.

For non-MoE layers, expert parallelism behaves the *same* as data parallelism.
For MoE layers, tokens in the sequence are sent to works where the desired experts reside.

![[Pasted image 20240415143735.png]]
((Ah, this diagram is actually correct, I think, but it's confusing. For example in the top-left image, this doesn't mean that all of the volume of the large blue square (representing all parameters) is being divided amongst the cores such that each core has a fraction; Instead, each of these small squares is supposed to be "the whole set of model weights". As evidence, in Data Parallelism in the bottom left, the "whole square" represents all of the data, and it's shown as shared/split over all of the cores. In the picture to the right of that, we're not saying that the data is being split into all these cores -- rather, each core gets a full copy of the data (which is represented by this blue square, through shrunked.)))


### Capacity Factor and Communication Costs
- Increasing the [[Capacity Factor]] increases quality, but increases communication costs and memory of activations.
- If all-to-all communications are *slow*, then using a smaller capacity factor is better.
- ==A good starting point is using top-2 routing with 1.25 capacity factor, and having one expert per core.==

### Serving Techniques
- A big downside of MoEs is the large number of parameters -- for local use cases, one might want to use a smaller model:
	- The Switch Transformers authors did early distillation experiments
	- By distilling an MoE back to its dense counterpart, they could keep 30-40% of the sparsity gains! ==So MoE+Distillation provides the benefits of faster pretraining and using a smaller model in production
		- ((How do you measure the "sparsity gains" in this case?))
- Recent approaches modify the routing to route full sentences or tasks to an expert, permitting extracting sub-networks for serving.
- Aggregation of Experts (MoE): This technique merges the weights of the experts, hence reducing the number of parameters at inference time.

### More on Efficient training
- FasterMoE (March 2022) analyzes the theoretical limit of different parallelism strategies, and other strategies... led to a ==17x speedup.==
- MegaBlocks (Nov 2022) explores efficient sparse pretraining by providing new GPU kernels that can handle the dynamism present in MoEs. 


# Exciting Directions of Work
- Distilling a sparse MoE back to a dense model with fewer parameters but similar performance.
- Model merging techniques of experts, and understanding the imapct
- Another area will be quanitzation of MoEs -- QMoE (Oct 2023) is a step in this direction, quantizing the MoEs to less than 1 bit per parameter (???)