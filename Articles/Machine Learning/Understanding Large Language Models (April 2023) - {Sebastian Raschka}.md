---
tags:
  - article
---
Link: https://magazine.sebastianraschka.com/p/understanding-large-language-models

This article is a summary of landmark papers in transformers

---------
### (1/19) *Neural Machine Translation by Jointly Learning to Align and Translate (2014)*
Bahdanau, Cho, Bengio
[https://arxiv.org/abs/1409.0473](https://arxiv.org/abs/1409.0473)
- ==Introduced an attention mechanism for recurrent neural networks== (RNN) to improve long-range sequence-modeling capabilities.
- Allows RNNs to translate longer sentences more accurately.
- Motivation for developing the original [[Transformer]] architecture later.

### (2/19) *Attention is All You Need (2017)*
Vaswani, Shazeer, Parmar, Gomez, ...
[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- Introduces the ==original [[Transformer]] architecture== with an [[Encoder-Decoder Architecture]]. Introduces scaled dot product attention, [[Multi-Headed Attention]] blocks, and positional input encoding.

### (3/19) *On Layer Normalization in the Transformer Architecture (2020)*
Xiong, Yang, Zheng, ..
[https://arxiv.org/abs/2002.04745](https://arxiv.org/abs/2002.04745)
- The location of [[Layer Normalization|LayerNorm]] in the Transformer architecture diagram remains a hotly debated subject -- should it be between the residual blocks, or elsewhere? There are "Post-LN Transformers" and an updated implementation defaulting to a "Pre-LN Transformer" variant.
- It's suggested by some papers that the "Pre-LN" works better. There's still ongoing discussions.


### (4/19) *Learning to Control Fast-Weight Memories: An Alternative to Dynamic Recurrent Neural Networks (1991)*
Schmidhuber
[Paper](https://www.semanticscholar.org/paper/Learning-to-Control-Fast-Weight-Memories%3A-An-to-Schmidhuber/bc22e87a26d020215afe91c751e5bdaddd8e4922)
- An interesting paper for those interested in historical tidbits and earlier approaches  fundamentally similar to modern transformers.
- This is a proposed alternative to RNNs called Fast Weight Programmers (FWP); involves a feedforward neural network that slowly leans by gradient descent to program the changes of the fast weights of another neural network.


### (5/19) *Universal Language Model Fine-Tuning for Text Classification (2018)*
Jeremy Howard and Sebastian Ruber
[https://arxiv.org/abs/1801.06146](https://arxiv.org/abs/1801.06146)
- Written one year after the original *Attention is all you need* paper; It didn't involve transformers, but instead focuses on recurrent neural networks.
- It ==proposed pretraining language models and transfer-learning them for downstream tasks==.
- While transfer learning was already established in CV, it wasn't yet prevalent in NLP. The [[ULMFiT]] paper was among the first to demonstrate that pre-training a language model and finetuning it on a specific task could yield state-of-the-art results in many NLP tasks.
- Process:
	- Train language model on large corpus of text
	- Finetune this LM on task-specific data
	- Finetune a classifier on the task-specific data with *gradual unfreezing of layers to avoid catastrophic forgetting*.
		- This is typically not done in practice when working with *Transformer* architectures, where all layers are typically finetuned at once.

### (6/19) *[[Bidirectional Encoder Representations from Transformers|BERT]]: Pre-training of Bidirectional Transformers for Language Understanding (2018)*
Devlin, Change, Lee, Toutanova
[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805)
- After the original Transformer paper, LLM research bifurcated in two directions:
	- [[Encoder-Only Architecture]] for predictive modeling tasks like text classification
		- BERT is an ==encoder-only architecture==
	- [[Decoder-Only Architecture]] for generative modeling tasks like translation, summarization, and other forms of text creation.
- The BERT paper ==introduces the original concept of masked-language modeling, and next-sentence prediction==.
- Highly recommended:
	- Follow up with [[RoBERTa]], which simplified the pre-training objectives by *removing* the next-sentence prediction tasks.

### (7/19) *Improving Language Understanding by Generative Pre-Training (2018)*
Radford and Narashiman
[Paper](https://www.semanticscholar.org/paper/Improving-Language-Understanding-by-Generative-Radford-Narasimhan/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035)
- The ==original GPT paper== ==introduced the popular decoder-only-style architecture and pretraining by next-token prediction== 
- While BERT was a bidirectional transformer due to its masked language model pretraining objective, GPT in contrast was a ==unidirectional, autoregressive model==.


### (08/19) *Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (2019)*
Ghazvininejad, Mohamed, Levy, Stoyanov, Zettlemoyer
[https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461)
- We mentioned earlier that BERT-type encoder-only LLMs are usually preferred for predictive modeling tasks, whereas GPT-type decoder-only LLMs are better at generating texts.
- To get the best of both worlds, the [[BART]] paper above combines both the encoder and decoder parts (not unlike the original transformer)

### (09/19) *Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond (2023)*
Yang, Jin ,Tang, Han ,Feng, Jiang, Yin, Hu
https://arxiv.org/abs/2304.13712
- This isn't a research paper, but is ==probably the best general architecture survey to-date, illustrating how different architectures evolved!==
![[Pasted image 20240124184220.png]]


### Scaling Laws and Improving Effficiency
- If you want to learn more about the various techniques to improve the *efficiency* of transformers, check out:
	- *2020 Efficient Transformers: A Survey* followed by *2023 A Survey on Efficient Training of Transformers*


### (10/19) *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022)*
Dao, Fu, Ermon, Rudra, Ré
[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
[[FlashAttention]]
While most transformer papers don't bother about replacing the original scaled dot product mechanism for implementing self-attention, FlashAttention does.

![[Pasted image 20240415224155.png|300]]
### (11/19) Cramming: Training a LM on a Single GPU in One Day ( 2022)
- Researchers train a masked LM/encoder-style LLM (eg BERT) for 24h on a single GPU
	- In comparison, 2018 BERT trained it on 16 TPUs for four days

### (12/19) [[Low-Rank Adaptation|LoRA]]: Low-Rank Adaptation of Large Language Models (2021)
- If we want to improve the ability of transformers on domain-specific tasks, we usually finetune them -- but this can impose high memory requirements for many consumers/companies!
- LoRA is one of the most influential approaches for finetuning LMs in a parameter-efficient manner. While other methods for parameter-efficient finetuning exist, LoRA is worth highlighting because it's both elegant and very general.
- While the weights of a pretrained model have full-rank on pretrained tasks, ==the LoRA authors point out that LLMs actually have a low "intrinsic dimension" when they are adapted to a new task.== 
- ==The main idea behind LoRA is to decompose the weight changes, $\Delta W$ , into a lower-rank adaptation, which is more parameter efficient.==

![[Pasted image 20240415224635.png]]


### (13/19) Scaling Down to Scale Up: A Guide to PEFT (2022)
- This survey reviews more than 40 papers on PEFT methods (including many popular techniques like [[Prefix Tuning]], Adapters, and LoRA)
![[Pasted image 20240415230613.png]]

### (14/19) Scaling Language Models: Methods, Analysis, and Insights from training [[Gopher]] (2022)
- A particularly nice paper with tons of analysis to understand LLM training.
- Researchers trained 280B model with 80 layers on 300B tokens.
- Includes interesting architecture modifications like [[RMSNorm]] instead of [[Layer Normalization|LayerNorm]] (both of which are preferred over [[Batch Normalization]] since they don't depend on the batch size and don't require synchronization -- an advantage in distributed settings with smaller batch sizes)

Main focus of the paper is analysis of task performance for different scales.
- The evaluation of 152 diverse tasks reveal that increasing model sizes benefits task like comprehension, fact checking, and identification of toxic language the most - though tasks related to logical/mathematical reasoning benefitted less from architecture scaling.
	- ((Hard to read into this, because 300B isn't a large amount of tokens anymore))


### (15/19) Train Compute-Optimal Large Language Models (2022)
- The 70B [[Chinchilla]] paper that outperformed the larger [[GPT-3]] model, showing that GPT was wildly overparametrized relative to the number of tokens that it was trained on.
- ==This paper defines the linear scaling law for large language model training.==

![[Pasted image 20240415231142.png]]

### (16/19) Pythia: A Suite for Analyzing Large Language Models across Training and Scaling (2023)
- The [[Pythia]] suite open-source LLMs (70M-12B parameters) to study how LLMs evolve over the course of training (many checkpoints provided).
- Architecture is similar to GPT but with some improvements ([[FlashAttention]], [[Rotary Positional Embedding|RoPE]]). 300B tokens.
- Main insights:
	1. Training on duplicated data (ie multiple epochs) does not benefit nor hurt performance
	2. Training order does not influence memorization.  ((But you should still shuffle for other reasons, right?))
	3. Pretrained term frequency influences task performance
	4. Doubling the batch size halves the training time but doesn't hurt convergence ((Juice those batch sizes!))

# Alignment: Steering LLMs to intended goals/interests

### (17/19) Training Language Models to Follow Instructions with Human Feedback (2022)
- The so-called [[InstructGPT]] paper, in which researchers use an [[Reinforcement Learning from Human Feedback|RLHF]] mechanism to fine-tune it further, using supervised learning on prompt-response pairs generated by humans combined with subsequently training a reward model over human preference data and optimizing the model using [[Proximal Policy Optimization]] (PPO).
- This paper is known for describing the idea behind [[ChatGPT]], which (according to legend) was a scaled-up version of InstructGPT fine-tuned on a larger dataset.

![[Pasted image 20240415231812.png]]


### (18/19)  [[Constitutional AI]]: Harmlessness from AI Feedback (2022)
- In this papers, researchers take alignment on step further -- instead of direct human supervision, researchers propose a self-training mechanism based on a list of ==principles== provided by humans -- a constitution.
![[Pasted image 20240415232129.png]]
Includes both a finetuning (+AI critique) and RL portion, both incorporating AI-feedback ([[Reinforcement Learning from from AI Feedback|RLAIF]], in the latter case)


### (19/19) [[Self-Instruct]]: Aligning Language Model with Self-Generated Instruction (2022)
- Instruction finetuning is how we get from GPT-3-like pretrained base models to capable LLMs like ChatGPT
- How do we scale the generation of such instruction-tuning datasets? One was is by bootstrapping an LLM off of its own generations!

Four step process:
1. Seed a task pool with a set of human-written instructions (15, in this case) and sample instructions.
2. Use a pretrained LLM (eg GPT-3) to determine the task category ((Useful for reasons, see paper))
3. Given the new instruction and its category, let a pretrained LLM generate the response.
4. Collect, prune, and filter the responses before adding them to the task pool.

![[Pasted image 20240415233046.png]]
In practice, this seems to work well, based on ROUGE scores.
