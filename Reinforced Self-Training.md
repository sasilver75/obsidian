---
aliases:
  - ReST
  - ReSTEM
---
December 11, 2023
[[DeepMind]]
Paper: [Beyond Human Data: Scaling Self-Training for Problem-Solving with Language Models](https://arxiv.org/abs/2312.06585)
#zotero 
Takeaway: A simple self-training method for when you don't have a large amount of data, but you have access to *binary* feedback on generations (eg math problems, where you can verify correctness). Involves generating samples, filtering them using binary feedback, fine-tuning on the remaining generations, and repeating the process.

Gulchere et al. (2023) proposed Reinforced Self-Training; Deepmind authors make some modifications and call their approach $ReST^{EM}$ , which can be viewed as applying expecation-maximization for reinforcement learning.

----

Notes:
- They make some modifications to the previously proposed ==Reinforced Self-Training== (Gulchere et al, 2023) algorithm, and call their approach $ReST^{EM}$, which can be viewed as applying expectation-maximization for RL. This algorithm alternates between the *expectation* and *maximization* steps:
	1. `Generate (E-Step)`: The language model generates multiple output samples for each input. Then, we filter samples using a binary reward, yielding a training dataset.
	2. `Improve (M-step)`: The original LM is SFT'd on the training dataset from the previous Generate step; the fine-tuned model is then used in the *next* Generate step.
- $ReST^{EM}$ and its various adaptations have had success in enhancing LMs across diverse domains, including:
	1. Machine translation
	2. Semantic parsing
	3. Preference alignment
	4. Elementary reasoning
- Our work aims to complement these efforts and investigate the effectiveness and scalability of model-generated synthetic data compared to human-generated data in the challenging domains of mathematical problem-solving (MATH) and code generation (APPS).

# Paper Figures

![[Pasted image 20240506174135.png]]
Above: See that we're able to improve performance on a Math benchmark and on a Coding benchmark! These are both examples of domains where automated binary reward is possible.