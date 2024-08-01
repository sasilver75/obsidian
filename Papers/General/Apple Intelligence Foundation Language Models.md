---
aliases:
  - AFM
  - Apple Foundation Model
---

July 29, 2024
[[Apple]]
[Apple Intelligence Foundation Language Models](https://machinelearning.apple.com/research/apple-intelligence-foundation-language-models)
#zotero 
Takeaway: ...
- Scratch
	- AFM on device
	- AFM server
	- Responsible AI principles

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
- 


## Powering Apple Intelligence Features


## Evaluation


## Responsible AI


## Conclusion


Abstract
> We present foundation language models developed to power Apple Intelligence features, including a ==∼3 billion parameter== model designed to run efficiently on ==devices== and a ==large server-based language model== designed for ==Private Cloud Compute==. These models are designed to perform a wide range of tasks efficiently, accurately, and responsibly. This report describes the model architecture, the data used to train the model, the training process, how the models are optimized for inference, and the evaluation results. We highlight our focus on Responsible AI and how the principles are applied throughout the model development.


# Paper Figures

![[Pasted image 20240731151125.png]]


![[Pasted image 20240731152048.png|500]]
Is this true for both AFM-on-device (3B) and AFM-server ("Larger")?
