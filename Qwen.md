September 28, 2023 -- [[Alibaba Research]]
Paper: [Qwen Technical Report](https://arxiv.org/abs/2302.03241) ("Tongyi Qianwen")

A family of chat models from Alibaba (1.8B, 7B, 14B, 72B) in various flavors. Their first public foray into LLMs.

Qwen (base pertrained language models)
Qwen-Chat (chat models finetuned with human alignment techniques)
Code-Qwen (finetuned on Code)
Code-Qwen-Chat (aligned, finetuned on Code)
Math-Qwen-Chat (aligned, finetuned on Math)

Later, a Qwen-VL multimodal model was released, as well as a Qwen-Audio
Later: a Qwen 1.5 MoE was released (14.3B, 27.B active): https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B

Uses [[Byte-Pair Encoding]] as their tokenization method, following GPT-3 and GPT-4, augmenting the vocabulary with commonly-used Chinese characters and words, as well as those in other languages.
Uses [[Rotary Positional Embedding|RoPE]] to incorporate positional information into the model (similar to [[PaLM]] and [[LLaMA]])
Uses [[RMSNorm]] rather than [[Layer Normalization|LayerNorm]].
Uses [[SwiGLU]] as an activation function (their experiments show that activations functions based on [[GLU]] generally outperform other baseline options, such as [[GeLU]])
Uses the standard [[AdamW]] optimizer for pretraining optimization, with cosine learning rate schedule (peak learning rate differing for each model size), decayed to 10% of the peak learning rate.
All models trained with BFloat16 for stability.
Incorporates LogN-Scaling and [[Sliding Window Attention]] ((I think; they refer to it just as "Window Attention"))

Abstract
> Large language models (LLMs) have revolutionized the field of artificial intelligence, enabling natural language processing tasks that were previously thought to be exclusive to humans. In this work, we introduce ==Qwen==, the first ==installment of our large language model series==. Qwen is a ==comprehensive language model series== that encompasses distinct models with varying parameter counts. It includes Qwen, the base pretrained language models, and Qwen-Chat, the chat models finetuned with human alignment techniques. The base language models consistently demonstrate superior performance across a multitude of downstream tasks, and the chat models, particularly those ==trained using Reinforcement Learning from Human Feedback (RLHF),== are highly competitive. The chat models possess advanced tool-use and planning capabilities for creating agent applications, showcasing impressive performance even when compared to bigger models on complex tasks like utilizing a code interpreter. Furthermore, we have developed coding-specialized models, Code-Qwen and Code-Qwen-Chat, as well as mathematics-focused models, Math-Qwen-Chat, which are built upon base language models. These models demonstrate significantly improved performance in comparison with open-source models, and slightly fall behind the proprietary models.

![[Pasted image 20240419144029.png]]