---
aliases:
  - ReFT
---
April 4, 2024
Stanford NLP, including (Wu et al), including [[Christopher Manning|Chris Manning]], [[Dan Jurafsky]], [[Christopher Potts|Chris Potts]]
[ReFT: Representation Finetuning for Language Models](https://arxiv.org/abs/2404.03592)
....
Takeaway: ... (Haven't read paper)

References: 
- Paper Club Video: [Oxen AI How ReFT Works w/ Author Zhengxuan Wu](https://www.youtube.com/watch?v=to2oKwnknUk)

----


ReFT operates on a frozen base model, intervening in hidden representations to learn task-specific representations. A member of [[Parameter-Efficient Fine-Tuning|PEFT]], but instead of updating and operating on existing weights, it keeps them frozen and steers "representations". Does add some parameters at inference time, but 15-65x more parameter-efficient than LoRA.

Background
- LLMs are powerful tools, but aren't suited to every task out of the gate -- they can hallucinate facts that aren't true, or are out-of-date.
- Existing PEFT techniques include"
	- Prompting (In-Context Learning)
	- [[Low-Rank Adaptation|LoRA]] (one of the most popular methods for fine-tuning; don't require that you finetune *all* weights in the network, instead using a low-rank matrix that can optionally be merged back into pretraining weights when you're done, for no additional overhead)
	- [[Adapter]]-based methods (Require adding additional layers at the end or in-between layers that are learnable, adding a small amount of overhead during inference.)

ReFT uses a"interventions" between selected layer and weights to steer representations
- Inspired by a lot of work in model interpretability (In the Scaling Monosemanticity, they have a traditional transformer, but they stick a sparse auto-encoder into the middle of it to extract features from the middle of the network. It turns out you can turn on/clamp activations in specific dimensions in that autoencoder latent that activate different features (eg Golden Gate Claude)).