---
aliases:
  - PEFT
---
A family of methods that allow fine-tuning models without modifying all of the parameters in the network. Often you freeze the model, add a small set of parameters, and modify it.

Examples: [[Low-Rank Adaptation|LoRA]], [[ControlNet]], [[Prompt Tuning]], [[Prefix Tuning]], Adapters
- [[Low-Rank Adaptation|LoRA]]: A technique where adapters are designed to be the product of two low-rank matrices. Inspired that the idea that pre-trained LMs have a low intrinsic dimension, so weight updates during adaptation might also have low intrinsic rank. Recent studies have shown that (compared to full finetuning), LoRA learns less, but forgets less.
- [[Quantized Low-Rank Adaptation|QLoRA]]: Builds on the idea of LoRA, but instead of using the full 16-bit model during finetuning, it applies a 4-bit quantized model. Reduces the average memory requirements of fine-tuning a 65B model from >780GB memory to a more manageable 48GB, without meaningfully degrading predictive performance.
- [[Prefix Tuning]]: We prepend a small number of trainable parameters to the hidden states of all of our transformer blocks, and fine-tune them while keeping the rest of the network frozen.
- [[Prompt Tuning]]: We prepend a small number of trainable pseudo-word vectors to the beginning of our prompt, and fine-tune them while keeping the rest of the network frozen.
- Adapters: This method adds fully-connected network layers *twice* to *each transformer block*, after the attention layer and after the feed-forward network layer. Again, they're fine-tuned while keeping the rest of the network frozen.


![[Pasted image 20240120232049.png]]![[Pasted image 20240415230544.png]]
(From 2022 _**Scaling Down to Scale Up**_ survey paper)