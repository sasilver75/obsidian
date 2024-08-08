---
aliases:
  - AFM
  - Apple Foundation Model
---

July 29, 2024
[[Apple]]
[Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)
#zotero 
Takeaway: Apple Foundation Models is a suite of models for various tasks; this paper covers an AFM On-Device (3B) and AFM Server (?B) model. There's a particular focus on using swappable [[Low-Rank Adaptation|LoRA]] [[Adapter]]s at various positions in the network, to customize performance of models to tasks like summarization, rewriting, notifications, while keeping the model small. Efforts are made for effective quantization of models, followed by accuracy-recovering [[Low-Rank Adaptation|LoRA]] adapters that serve as the base for product-specific adapter finetuning described above. Notably in Post-training, authors introduce an ==Iterative Teaching Committee (iTeC)== technique, as well as a ==Mirror Descent with Leave-One-Out Estimation (MDLOO)== technique that authors say outperforms [[Proximal Policy Optimization|PPO]] by combining a leave-one-out estimator to estimate advantage of prompt-response pairs, and a Mirror Descent Policy Optimization (MDPO) to optimize the policy. Their RL incorporates both the *strength* of a preference (strongly, slightly, etc.), as well as "single-side grading" of each response.

> "Most people usually think that LoRAs are 100s-1000s of datapoints, but they used ~10B here." -Vibhu Sapra, Latent Space Paper Club

Notes:
- Note that the represent the values of the adapters using 16 bits, while the main model is ~3.7 bit quantized; this is why they can't just merge back in the common accuracy-recovering part of the adapter.

References:
- Blog: [Swyx's AINews Apple Intelligence](https://buttondown.email/ainews/archive/ainews-apple-intelligence/)
- Official Blog: [Introducing Apple's On-Device and Server Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- Discord thread: [Latent Space Apple Intelligence](https://discord.com/channels/822583790773862470/1249801456870101013)

---

## Introduction
- Apple Intelligence consists of *==multiple highly-capable models==* that are fast, efficient, and specialized for users' everyday tasks, and ==can *adapt on the fly* for their current activity==.
	- Foundation models built into Apple Intelligence have been fine-tuned for user experiences such as writing and refining text, summarizing notifications, creating playful images, and taking in-app actions.
- This paper covers how two of these models (AFM = Apple Foundation Model):
	- ==AFM-on-device== (3B, dense decoder-only)
	- ==AFM-server== (Larger, server-based LM, dense decoder-only)
- These two foundation models are part of a *larger family of generative models* from Apple, including a coding model to build intelligence into Xcode, as well as a diffusion model to help users express themselves in the Messages app.
- ==Responsible AI Principles== to guide how Apple develops AI tools
	1. Empower users with intelligent tools
	2. Represent our users (globally, avoiding biases)
	3. Design with care (misuse/harm, continuously improving)
	4. Protect privacy (Protect user privacy with on-device processing + Private Cloud Compute)

## Architecture
- AFM base models are ==dense==, decoder-only models with the following design choices:
	- Shared input/output embedding matrix ([[Weight Tying]]), reducing memory usage
	- Pre-Normalization with [[RMSNorm]] for training stability
	- Query/Key normalization to improve training stability
	- [[Grouped Query Attention]] (GQA) with 8 KV heads to reduce KV-cache memory footprint
	- [[SwiGLU]] activations for higher efficiency
	- [[Rotary Positional Embedding|RoPE]] with base frequency 500k for long-context support

## Pre-training
- We focus on efficiency and data quality at every step in order to pretrain for a high-quality end-to-end user experience with efficient and low-latency models.
- AFM pretraining dataset consists of a diverse and high-quality mixtures, using publicly available datasets and publicly-available information crawled by our web-crawler ==Applebot,== respecting `robot.txt` directives.
	- No private Apple user data is used in the data mixture
	- Extensive efforts were made to exclude profanity, unsafe material, and personally-identifiable information from publicly available data
	- Rigorous decontamination performed against many common evaluation benchmarks
	- "==We find that data quality, much more so than quantity, is the key determining factor of downstream model performance.=="
- Document-level profanity and PPII filters are followed by an extraction and quality filtering pipeline:
	1. Body extraction using a combination of Safari's reader mode and the ==Boilerpipe== (2010) algorithm.
	2. Safety and profanity filtering, using both heuristics and model-based classifiers.
	3. Global fuzzy deduplication using [[Locality Sensitive Hashing]]
	4. Extensive quality filtering using both heuristics and model-based classifiers (cites [[DataComp-LM]])
	5. Decontamination against 811 common pre-training benchmarks, filtering entire documents upon 4-13 gram collisions with any of the benchmark datasets, unless the collision count for a given n-gram reaches a "common usage" threshold of 1,000.
- Apple also licenses a limited amount of high-quality data from publishers, which they similarly decontaminate.
- Apple acquires Code data in 14 common programming languages from license-filtered GitHub repos, deduplicates it, and further filters for PII and Quality in a way similar to that described above.
- Apple acquires Math data from an unnamed Math Q&A dataset of 3B tokens from 20 web domains rich in math content, as well as 14B tokens from web pages like math forums, blogs, tutorials, and seminars. ((I think these are initial numbers pre-filtering))
	- To filter, they use a specialized pipeline to identify mathematical templates, a quality filter powered by a LM classifier specifically designed for math, and a domain filter... followed by the usual deduplication, decontamination, and PII removal.
- For a tokenizer, they use [[Byte-Pair Encoding|BPE]], following the implementation from [[SentencePiece]]
	- They also split all numbers into individual digits and use byte-fallback to decompose unknown UTF-8 characters into byte tokens.
	- ==Vocabulary size 100k for AFM-server and 49k for AFM-on-device.==
- We break AFM pre-training into 3 distinct stages:
	1. ==Core==
	2. ==Continued==: We down-weight lower-quality bulk web-crawl data, favoring higher code and math weight instead.
	3. ==Context-lengthening==: Similar to another [[Continued Pretraining]] stage, but conducted at longer sequence length, with synthetic long-context data included in the mixture.
	- All stages use decoupled [[L2 Regularization|Weight Decay]] as well as a simplified version of $\mu$param. All stages maintain sharded model and optimizer states in [[float32]] ("full precision"), casting to [[bfloat16]] for the forward and backward passes, for efficiency.
- AFM Server Pretraining
	- ==AFM-server trained from scratch for 6.3T tokens== with a sequence length of 4096 and batch size of 4095 sequences. Learning rate schedule includes a linear warmup for 5k steps, followed by cosine decay to 0.005 of the peak over the remainder of training. LR 0.01, weight decay of 3.16e-4.
- AFM On-Device Pretraining
	- Authors found that [[Distillation|Knowledge Distillation]] and [[Structured Pruning]] are effective ways to improve model performance and training efficiency, and are complimentary to eachother.
	- ==We initialize AFM on-device from a *pruned* 6.4B model trained from scratch using the AFM-server recipe==.
		- Authors only pruned the hidden dimensions in FFNN layers, and used Soft-Top-K masking instead of HardConcrete masking. We employ the same pre-training data mixture as the core phase to learn the mask, training for 188B tokens.
	- During core pretraining, a distillation loss is used by replacing the target labels with a convex combination of the true labels and the teacher model's top-1 predictions (.9 weight for teacher labels), trained for ==6.3T tokens== ((Meaning the total number of tokens is greater than 6.3T?)x.
	- Observation: Initializing from a pruned model improves data efficiency and the final benchmark results by 2%, while adding distillation boosts MMLU and GSM8K about 5% and 3% respectively. All training hyperparameters same as Server, besides batch size.
- Continued pre-training
	- We perform continued pretraining at a sequence length of 8192, with ==another 1T tokens from a mixture that upweights math and code==, and down-weights bulk web-crawl.
	- Peak LR 3e-4 and decoupled [[L2 Regularization|Weight Decay]] of 1e-5 and 1000 warm-up steps with a final LR decay to 0.001 of peak, differently to core pretraining.
	- We didn't find a distillation loss to be helpful here for AFM on-device, so its recipe is identical to that used for AFM-server.
- Context Lengthening
	- A further 100B tokens of [[Continued Pretraining]] at a sequence length of 32768 tokens. We also increase the RoPE base frequency from 500k to 6315089... with the idea that this will allow for better short-to-long generalization.
- Optimizer
	- We use a variant of [[RMSProp]] with momentum for AFM pre-training with some modifications (divide the raw gradient by the square root of a bias-corrected EMA of the squared gradient).
- Training
	- AFM models trained on v4 and v4 TPU clusters (8192 TPUv4 chips, 2048 TPUv5p chips) with AXLearn framework (Apple), JAX, and use a combination of [[Tensor Parallelism]], [[Fully Sharded Data Parallelism|FSDP]], and [[Sequence Parallelism]], allowing training to scale.


## Post-training
- Even though we use task-specific adapters, we found lifts from post-training; we instill general purpose *instruction following and conversation capabilities* to AFM models, with the goal being to ensure these model capabilities are aligned with Apple core values/principles.
- Two stages:
	- [[Supervised Fine-Tuning|SFT]] (using a rejection sampling fine-tuning algorithm with teacher committee (==iTeC==))
	- [[Reinforcement Learning from Human Feedback|RLHF]] (using a mirror descent policy optimization and leave-one-out advantage estimator (==MDLOO==))
- Data
	- We use a hybrid data strategy of both human-annotated and synthetic data; we've found *data quality to be the key to model success*, and have thus conducted extensive data curation and filtering procedures.
	- Demonstration Data
		- To fuel instruction fine-tuning of AFM, we collect high-quality human annotated demonstration datasets from *various sources.* This data consists of *both system-level and task-level instructions/prompts*, as well as their corresponding instructions.
		- We focus on ==helpfulness, harmlessness, presentation, and response accuracy, and target a diverse task distribution covering Apple Intelligence features.==
	- Human preference feedback
		- We instruct human annotators to compare and rank two model responses for the same prompt to collect side-by-side preference labels.
		- We also use "single-side question" to guide this process; these ask raters to grade model response quality on various aspects including instruction following, safety, factuality, and presentation, and we retain these labels for model training.
	- Synthetic Data
		- When guided by robust reward models, AFMs are capable of generating high-quality responses that, in some domains, are superior even to human annotations.
		- Primarily in the domains of:
			- ==Mathematics== (usually expert knowledge and laborious work; generating math problems and their solutions)
				- *Problem **rephrase** and **reversion***: Rephrase seed math questions and curate reverse questions, asking the model to derive some specific number in a problem statement when provided with the final answer.
				- *Problem evolution*: Inspired by [[Evol-Instruct]]; given a seed problem, evolve it either in-depth (adding complexity) or in-breadth (improving topic coverage). Ultimately, we only select problems that score above a certain depth (difficulty) threshold.
				- We then prompt AFM to synthesize responses N responses with CoT per-question. We use ground-truth if available to filter answers, or otherwise assess correctness response by querying an LLM judge.
			- ==Tool Use==
				- We develop tool-use capabilities (eg function calling, code-interpreter, browsing) through a mixture of human and synthetic data.
				- Bootstrap with synthetic data, which focuses on single-tool use cases, and then collect human annotations that involve multi-tool and multi-step scenarios.
			- ==Coding==
				- The generation of synthetic coding dataset involves a [[Self-Instruct]] method with [[Rejection Sampling]]. 
				- They start with 71 different programming topics as seeds, and use the model to generate an initial pool of coding interview-like question. 
					- For each question, we generate a set of unit tests and a number of potential solutions, and use an execution-based ==rejection sampling== method to select the best solution (the solution with the highest number of successful executions against the unit tests).
				- Resulted in 12k high-quality code samples
- SFT
	- We establish a series of quality guards of data, including ratings from in-house human labelers, ==automatic model-based filtering techniques==, as well as ==deduplication using text embeddings==. We scale up via a variety of synthetic data generation methods and rejection sampling, as described above.
	- We treat the mixture ratio as an optimization problem ((It seems like they literally use something like a gradient descent, training models at each point)).
	- AFM Server uses 5e-6 LR, and 2e-5 for AFM-device, as well as a dropout rate of 0.1.
- RLHF
	- We train a robust reward model and apply it in two algorithms of ==iTeC== and ==MDLOO==.
	- Reward Modeling
		- Trained using human preference data, with each item containing one prompt and two responses, along with human labels indicating both:
			1. The preferred response between the two, as the *==preference level==* (significantly better, better, slightly better, negligibly better)
			2. The ==*single-sided grading*== of each response, measuring degree of (instruction following, conciseness, truthfulness, harmlessness) of the responses.
		- RM training has two main innovations:
			1. ==A soft label loss function that takes the *level of human preference* into account.==
			2. ==We incorporate single-sided gradings as *regularization terms* in reward modeling.==
	- Authors incorporate the [[Bradley-Terry-Luce Model]] (BTL Model) for reward modeling, where the probability that a human annotator prefers one response over another is modeled as the sigmoid function of the difference of the rewards.
		- Authors find their method works better than margin-based loss functions (eg like that used in [[LLaMA 2]])
		- ...and that using the single-sided gradings as regularization terms improves accuracy of the reward model.
	- ==Iterative Teaching Committee (iTeC)==
		- Authors use a novel RLHF framework that combines various preference optimization algorithms, including:
			1. [[Rejection Sampling]]
			2. [[Direct Preference Optimization]] and its variants, such as [[Identity-Mapping Preference Optimization|IPO]]
			3. Online reinforcement learning (RL)  ((Meaning [[Proximal Policy Optimization|PPO]]?))
		- Iterative committee: An important lesson is to refresh online human preference data collection using a *diverse set* of the best-performing models!
			- For each batch of human preference data collection, we set up a collection of latest promising models trained from SFT/RS/DPO/IPO/RL, as well as the best models from previous iterations. We refer to this as our ==model committee==. We collect pairwise human preference on responses *sampled from the latest model committee* to update our reward model. We then continue the next round of RLHF.
		- Committee Distillation
			- We further run [[Rejection Sampling]] from the model committee with the latest reward model as a reranker. For each prompt, we sample multiple responses from each model in the committee, using the latest reward model to select the best response for each prompt. This allows us to combine advantages of models trained by different preference optimization algorithms.
				- We find that online RLHF, DPO, and IPO are good at improving math skills, while rejection sampling fine-tuning learns instruction following and writing skills more effectively.
			- Authors found that data quality matters more than data quantity for larger models, but smaller models can achieve tremendous improvement when we scale up the *number* of prompts for distillation.
	- Online RLHF Algorithm: ==MDLOO==
		- We use the commonly-adopted RLHF objective that maximizes the KL-penalized expected reward function:
		- ![[Pasted image 20240802225307.png]]
		- In their RL training authors use the follow equation, whose expectation is equivalent to equation (1) above:
		- ![[Pasted image 20240802225333.png]]
		- Similar to commonly-used RLHF algorithms like [[Proximal Policy Optimization|PPO]], we use a trust-region based policy iteration algorithm, making two main design choices in our online RL algorithm:
			1. Use a ==Leave-One-Out (LOO) estimator== to estimate the advantage of a prompt-response pair.
			2. We use ==Mirror Descent Policy Optimization (MDPO)== to optimize the policy, differently from the more commonly-used clipping-based PPO method.
		- Authors combined these and named it as ==Mirror Descent with Leave-One-Out Estimation (MDLOO)==.
			- For the decoding stage of the algorithm, we decode multiple response for each prompt, assign the *advantage* of each response to be the difference of the reward of the (prompt, response) pair and the mean reward of the other responses generated by the same prompt.
				- This aims to estimate how much better a response is compared to a typical response.
				- We find this crucial in stabilizing the RL algorithm!
			- We use a KL-regularization-based trust-region method (==MDPO==) to control the policy change in each iteration. We find this algorithm to be more effective than PPO in our setting.


## Powering Apple Intelligence Features
- We elevate the performance of even small models to best-in-class performance through task-specific fine-tuning, enabling a single model to be specialized for dozens of tasks.
- Adapter Architecture
	- We use [[Low-Rank Adaptation|LoRA]] [[Adapter]]s that can be plugged into *various layers* of the base model!
		- We adapt all of AFM's linear projection matrices in self-attention layers, and the fully-connected lays in the pointwise feedforward networks.
		- By finetuning adapters, the original parameters of the base pre-trained model remain unchanged, preserving general knowledge of the model while tailoring the adapters to support specific tasks.
		- Even representing adapter parameters using 16 bits, the ==parameters for a rank 16 adapter typically require only 10s of megabytes, and can be dynamically loaded and cached in memory and swapped, allowing for on-the-fly model customization==.
- Optimizations
	- Both inference latency and power efficiency are important for overall user experience. WE apply various optimization techniques to allow AFM to be efficiently deployed on-device and in Private Cloud Compute, reducing memory/latency/power usage while maintaining quality.
	- We develop and apply SoTA model [[Quantization]] techniques to achieve quantization with an average *less than 4 bits per weight, with nearly zero loss in performance*.
	- The model is *==compressed==* and *==quantized==* to an average of <4 bits-per-weight after the post-training stages. ==This results in a moderate quality loss, so we then attach a set of parameter-efficient LoRA adapters for quality recovery.==
		- We make sure that these LoRA adapter training recipes are consistent with pre-training and post-training processes.
		- It's noteworthy that training accuracy-recovery adapters is sample-efficient... ==we require only approximately 10B tokens (~0.15% of base model training) to fully-recover the capacity for the quantized model.==
		- ==Application adapters can then be created by initializing adapter weights from the *accuracy-recovery* adapters, then fine-tune from these accuracy-recovery adapters (all while keeping the quantized base model frozen).==
	- Regarding adapter size, we found that ==adapter adapter rank of 16 offers the optimal tradeoff between model capacity and inference performance== (but they provide a suite of accuracy-recover adapters in ranks {8, 16, 32}).
- Quantization Schemes
	- Another benefit brought by accuracy-recovery adapters is that they allow for more flexible choices of quantization schemes.
	- Previously, people usually group the weights into small blocks, normalize each block by the corresponding maximal absolute values to filter out outliers, then apply quantization algorithms in a block basis (tradeoff of block size w.r.t. throughput and accuracy loss).
		- We found that ==accuracy-recovery adapters== can greatly improve the pareto frontier of this tradeoff!
	- Our AFM on-device model uses *palletization*: For projection weights, every 16 columns/rows share the same quantization constants (i.e. lookup tables) and are quantized using K-means with 16 unique values (4-bit).
	- Mixed-precision quantization: Residual connections exist in every transformer block and every layer in AFM, so it's unlikely that all layers have equal importance. ==Following this intuition, we further reduce the memory usage by pushing some layers to use 2-bit quantization (default is 4-bit)== (on average, AFM-on-device can be compressed to ~3.5 bpw without significant quality loss).


## Evaluation
Basically this can be summed up in the figures in the Paper Figures section below. Used a mix of public ([[MATH]], [[GSM8K]], [[MMLU]], [[IFEval]], [[Berkeley Function-Calling Leaderboard|BFCL]]) and internal evaluations.



Abstract
> We present foundation language models developed to power Apple Intelligence features, including a ==∼3 billion parameter== model designed to run efficiently on ==devices== and a ==large server-based language model== designed for ==Private Cloud Compute==. These models are designed to perform a wide range of tasks efficiently, accurately, and responsibly. This report describes the model architecture, the data used to train the model, the training process, how the models are optimized for inference, and the evaluation results. We highlight our focus on Responsible AI and how the principles are applied throughout the model development.


# Paper Figures

![[Pasted image 20240731151125.png]]


![[Pasted image 20240731152048.png|500]]
Is this true for both AFM-on-device (3B) and AFM-server ("Larger")?

![[Pasted image 20240802230736.png]]


![[Pasted image 20240803125531.png]]
![[Pasted image 20240803125540.png]]
Note that huge frontier models are getting ~90 MMLU (saturated), and L3.1 70B got 79.3, L3.1 8B got 66.7B.,
- Think: For notification summarization, etc... you don't *need* 100 MMLU, which includes knowledge from humanities, hard sciences, us history, virology, nutrition, etc.

![[Pasted image 20240803130003.png]]
Seems like the AFM models are about on-par with the LLaMA 3 models (not 3.1)

![[Pasted image 20240803130059.png|400]]

![[Pasted image 20240803130119.png|500]]

![[Pasted image 20240803130137.png|500]]

![[Pasted image 20240803130148.png|500]]

![[Pasted image 20240803130233.png|500]]






