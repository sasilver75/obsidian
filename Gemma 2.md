June 27, 2024
[[DeepMind]], Gemma Team
[Gemma 2: Improving Open Language Models at a Practical Size](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)
#zotero 
Takeaway: A pair of language models (in base and instruct versions) sized 9B and 27B, trained on 8T and 13T tokens respectively (and a 2.6B Gemma 2 model being on the horizon, trained on 2T tokens); they're competitive alternatives to models 2-3x bigger, and come with permissive use licenses making them great for (eg) synthetic data generation.

Note: Gemma 2's Terms of Use has unrestricted use for the output, meaning models trained on Gemma-2 output can be used for anything. This makes it an interesting choice for (eg) synthetic data generation. The people at r/localLLaMA really seem to love this model in mid-July 2024. Update: A week later in mid-July 2024, LLaMA 3.1 8b and 70b came out and wiped the floor with Gemma 2.

References:
- Blog: [HuggingFace: Welcome Gemma 2 - Google's New Open LLM](https://huggingface.co/blog/gemma2)

----

## Introduction
- Performance in small scale models are largely derived from increasing the length of training, but this approach only scales logarithmically with dataset size... For [[LLaMA 3]], small models required up to 15T tokens, and only improved the SoTA by less than 1-2%!
	- This paper explores alternatives to improving small model performance without naively increasing training length. For example replacing the next-token-prediction task with a richer objective!
	- This paper focuses efforts on [[Distillation|Knowledge Distillation]] (soft-target distillation), replacing the one-hot vector for each token with the distribution of potential next tokens computed from a stronger language model.
		- Authors train the 9B and 2.6B models on a quantity of tokens > 50x the compute-optimal quantity.
			- (They train the 27B model from scratch for this work)
- Author's models also benefit from:
	- Interleaved global and local ([[Sliding Window Attention]]) attention layers
	- [[Grouped Query Attention]]
	- Capped logits

## Model Architecture
- A few architectural elements are similar to the first version of Gemma models:
	- Context lenght of 8192
	- Use of [[Rotary Positional Embedding|RoPE]]
	- Use of approximated [[GeGLU]] nonlinearity/activation
- The following are differences:
- Local Sliding Window and Global Attention
	- We alternate between a local sliding window attention and global attention in every other layer.
	- The sliding window size of local attention layers is set to 4096 tokens, while hte span of the global attention layers is set to 8192 tokens.
- Logit soft-capping
	- Following [[Gemeni 1.5]], we cap logits in each attention layer and the final layer such that the value of the logits stays between `-soft_cap` and `+soft_cap`.
	- More specifically, we set the logits to be `logits = soft_cap * tanh(logits/_cap)`
	- For he 9B and 26B models, they cap attention logits at 50.0 and final logits at 30.0.
	- Note that because attention logit soft-capping is incompatible with common FlashAttention implementations, we have removed this feature from libraries that use FlashAttention, name HuggingFace transformers and the vLLM implementation.
- Post-norm and pre-norm with [[RMSNorm]]
	- To stabilize training, we use [[RMSNorm]] to normalize the input and output of each transformer sub-layer, the attention layer, and the feedforward layer.
- [[Grouped Query Attention]] (GQA)
	- Both the 27B and 9B models use GQA with num_groups=2, based on ablations showing increased speed and inference time while maintaining downstream performance.

## Pre-Training
- ==We train Gemma 2 27B on 13T of primarily-english data, and the 9B model on 8T tokens, and the 2.6B on 2T tokens.== Tokens come from a variety of sources, including web documents, code, and science articles.
- Tokenizer: Uses the same tokenizer as [[Gemma]] and [[Gemeni]], a [[SentencePiece]] tokenizer with split digits, preserved whitespace, and byte-level encodings, and a 256k vocabulary.
- Filtering: We use the same filtering techniques as Gemma; we filter the pretraining dataset to reduce the risk of unwanted or unsafe utterances, filter out certain personal information, or other sensitive data, decontaminate evaluation sets from our pre-training data mixture, and reduce the risk of recitation by minimizing the proliferation of sensitive outputs.
- Knowledge Distillation
	- Given a large model as teacher, we learn smaller models (9B, 2.6B) by *distilling* from the probability given by the teacher of each token x, given its context.
	- They minimize the negative log likelihood between the probabilities from the teacher and student.
	- ![[Pasted image 20240718172546.png|600]]
	- P_s is the probability of a token according the student, and P_T according to the teacher . x is each token, and x_c is its context. Represents minimizing the [[Kullback-Leibler Divergence|KL-Divergence]] (equivalently, [[Cross-Entropy]]) between the teacher's distribution and the student's distribution.


## Post-Training
- The fine-tune their pretrained models into Instruction-Tuned models, applying SFT on a mixture of synthetic and human-generated response pairs. We then apply RLHF on top of these models.
- Authors extend the post-training data from [[Gemma]] with a mixture of internal and external public data; they use the prompts (==but not answers==) from [[LMSYS-Chat-1M]]. 
- Supervised Fine-Tuning (SFT)
	- They run behavioral cloning (SFT) on synthetic and real prompts, and run distillation from the teacher *on the student's distribution* (This is =="On-Policy Soft-Target Distillation"==).
- Reinforcement Learning from Human Feedback (RLHF)
	- They use a similar RLHF algorithm as Gemma v1.1., but with a different reward model that's an order of magnitude larger than the policy (language model under training).
	- The ==new reward model is oriented more towards conversational capabilities, specifically multi-turn.==
- Model Merging
	- ==They average models from experiments run with different hyperparameters==
- Data filtering
	- Using synthetic data, we run several stages of filtering to remove examples that show certain personal information, unsafe or toxic outputs, mistaken self-identification data, and duplicated examples.
	- They include subsets of data that encourage better in-context attribution, hedging, and refusals to minimize hallucinations.


## Ablations
- The main finding of this work is the impact on knowledge distillation on small models.
- Added figures to figure section.


## Evaluation


## Responsibility/Safety/Security


## Discussion and Conclusion







Abstract
> In this work, we introduce Gemma 2, a new addition to the Gemma family of lightweight, state-of-the-art open models, ranging in scale from 2 billion to 27 billion parameters. The ==9 billion== and ==27 billion parameter== models are available today, with a ==2 billion parameter model== to be released shortly. In this new version, we provide several technical modifications to our architecture, such as ==interleaving local-global attentions== (Beltagy et al., 2020a) and [[Grouped Query Attention]] (Ainslie et al., 2023). We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction. The resulting models deliver the best performance for their size, and even offer competitive alternatives to models that are 2-3× bigger. We release all our models to the community.

# Paper Figures
![[Pasted image 20240718161743.png|300]]
Model parameter and design choices

![[Pasted image 20240718170630.png|400]]
Parameter counts for the various models sizes. Interesting that they separate the embedding and non-embedding parameters.

![[Pasted image 20240718174500.png|300]]
Comparing training from scratch versus distilling, at the 2.6B model size.

![[Pasted image 20240718174508.png|300]]
Comparing training from scratch versus distilling, at various model sizes.

![[Pasted image 20240718175459.png|300]]
Deep > Wide network

![[Pasted image 20240718175518.png|300]]
Impact of introducing [[Grouped Query Attention|GQA]] over vanilla [[Multi-Headed Attention]].

![[Pasted image 20240718175542.png|300]]
Impact of [[Sliding Window Attention]] window size; see that it doesn't matter much for perplexity, and the smaller window sizes will achieve better inference speeds, since smaller window sizes require less O(N^2) computation.

![[Pasted image 20240718175831.png|400]]
Gemma 2 compared to larger [[LLaMA 3]]-70B and [[Qwen 1.5]] 32B models. See that it beats Qwen and is competitive with LLAMA 3-70B, despite being many fewer parameters.




-----
Notes from [HuggingFace Gemma Blog Post](https://huggingface.co/blog/gemma2)

Gemma comes in 9B (8T training tokens) and 27B (13T training tokens) sizes, with each offered in both base and instruct-tuned versions. We don't know the exact details of the training mix.

Comes in a permissive license that allows redistribution, fine-tuning, commercial use, and derivative works.

Details:
- Context length: 8192 tokens
- [[Rotary Positional Embedding|RoPE]] embeddings
- [[Sliding Window Attention]] (interleave sliding window and full-quadratic attention for quality generation)
- Logit soft-capping (prevents logits from growing excessively by scaling to a fixed range, improving training)
- [[Distillation]]: Leverages a larger teacher model to train a smaller model (for the 9B model)
- [[Model Merging]]: Combines two or more LLMs into a single new model

The Instruct versions have been trained on a mix of synthetic and human-generated prompt-response pairs using SFT, distillation from a larger model, [[Reinforcement Learning from Human Feedback|RLHF]], and model merging using WARP to improve overall performance.
- No real details about the fine-tuning datasets or the hyperparameters associated with SFT and RLHF have been shared.

[[Sliding Window Attention]]
- A method to reduce memory and time requirements of attention computation.
- Novelty in Gemma 2 is that sliding attention is applied to *every other layer*, while the layers between still use full quadratic global attention.
	- This seems to be a way to increase quality in long-context situations, while partially benefitting from the advantages of sliding attention.

==Soft-capping and attention implementations==
- ==[[Soft-Capping]]== is a technique that prevents logits from growing excessively large without truncating them.
	- We divide the logits by some maximum value threshold (soft_cap), then passing them through a tanh layer (ensuring they are in the (-1,1) range).
	- This guarantees that the final values will be in the (-soft_cap, +soft_cap) interval without losing much information, but stabilizing the training.
- Logits are then calculated by: `logits = soft_cap * tanh(logits/soft_cap)`
- ==Gemma 2 employs soft capping for the final layer, and for every attention layer.==
	- Attention logits are capped at 50, and the final logits at 30.
	- At the time of release, Soft capping is *incompatible* with [[FlashAttention]]/Scaled Dot Product Attention ("normal" attention), but they can still be used in inference for maximum efficiency. (?)

[[Distillation|Knowledge Distillation]]
- In this paper, knowledge distillation is used to train a smaller student model to mimic the behavior of a larger teacher model by training against the distribution of token probabilities from the teacher, which provides a richer signal for the student to learn from.
- Knowledge distillation was used to pre-train the 9B model, while the 27B was trained from scratch.
	- For the 9B, they used a teacher (unspecified, but presumably [[Gemeni Ultra]]), and trained the student model on it in a supervised fashion.
	- This method has drawbacks, since the model capacity mismatch between student and teacher can lead to a *train-inference mismatch* (([[[Exposure Bias]]?])), where the text generated by the student during inference is out of distribution compared to that seen during training.
	- To handle this, Gemma 2 team used "==On-Policy Distillation==," where the student generates completions from the SFT prompts, and then these completions are used to compute the KL divergence between the teacher and student's logits (for the teacher's response to the same prompts.)
		- By minimizing the KL divergence throughout training, the student learns to better model the behavior of the teacher accurately, while also minimizing train-inference mismatch.
	- ((Sam explanation: In norma off-policy distillation, inputs (prompts/contexts) come from a fixed dataset... the teacher generates responses to these inputs, and the student learns to mimic the teacher's outputs for the fixed inputs... In On-Policy distillation, the student generates responses to SFT prompts, and then the teacher model produces probability distributions for the *student-generated inputs*; the student learns by comparing its own output probabilities to the teacher's for the self-generated inputs... So the student is the one that determines the sequence that we do comparisons on, not the teacher.))
	- One advantage of on-policy distillation is that you only need the logits from the teacher, so you don't need to rely on reward models or LLM-as-a-Judge to improve the model (??? This seems apples to oranges -- this is distillation we're talking about.)

[[Model Merging]]
- A technique that combines two or more LLMs into a single new model
- Relatively new and experimental; [[MergeKit]] is a popular open-source toolkit for merging LLMs, implementing linear, SLERP, TIES, DARE, and other merging techniques.
- Gemma 2 used ==Warp==, a new merging technique that merges models in three distinct stages:
	1. Exponential Moving Average (EMA): Applied during the reinforcement learning finetuning process
	2. Spherical Linear Interpolation (SLERP): This is applied after the RL fine-tuning of multiple policies
	3. Linear Interpolation Towards Initialization (LITI): This stage is applied after the SLERP stage

Gemma 2 Evaluation
![[Pasted image 20240718154931.png|500]]
Pretty competitive performance relative to LLaMA 3 70B, despite being trained on fewer tokens and having smaller parameter count.

![[Pasted image 20240718154942.png|500]]
For the small Gemma 2, it seems to be meaningfully more performance than (eg) [[Mistral 7B]], or even [[LLaMA 3]]-8B. Note that it does have more parameters.





