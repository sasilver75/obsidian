January 30, 2023 (12 months after [[CLIP]]) -- [[Salesforce Research]]
Paper: [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)

An improvement of [[CLIP]]
Bootstraps vision-language pre-training by producing its own captions, rather than relying on lazy ones from online.

Abstract
> The cost of vision-and-language pre-training has become increasingly prohibitive due to end-to-end training of large-scale models. This paper proposes ==BLIP-2==, a ==generic and efficient pre-training strategy that bootstraps vision-language pre-training from off-the-shelf frozen pre-trained image encoders and frozen large language models==. BLIP-2 bridges the modality gap with a lightweight Querying Transformer, which is pre-trained in two stages. The ==first== stage ==bootstraps vision-language representation learning from a frozen image encoder==. The ==second== stage ==bootstraps vision-to-language generative learning from a frozen language model==. BLIP-2 achieves state-of-the-art performance on various vision-language tasks, despite having significantly fewer trainable parameters than existing methods. For example, our model outperforms Flamingo80B by 8.7% on zero-shot VQAv2 with 54x fewer trainable parameters. We also demonstrate the model's emerging capabilities of zero-shot image-to-text generation that can follow natural language instructions.

![[Pasted image 20240420181558.png]]

![[Pasted image 20240420181705.png]]

![[Pasted image 20240420181716.png]]

