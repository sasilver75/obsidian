---
aliases:
  - Mosaic Pretrained Transformer
---
May 5, 2023 -- [[MosaicML]] (acq. [[DataBricks]])
Blog: [Introducing MPT-7B: A New Standard for Open-Source, Commercially Usable LLMs](https://www.databricks.com/blog/mpt-7b)
dels are open-source and commercially-usable LLMs pre-trained on 1T tokens.
MPT-7B, MPT-30B (with variants like MPT-30B-Instruct and MPT-30B-Chat)


Released a bunch of open-source tools (eg llm-foundry) for training models, used by Mosaic.
Mosaic's business was that they would offer (as mercenaries) to pretrain LLMs for people on their platform, using customer data. The MPT models were basically a proof-point saying: "Look, we can do this."

Leverages attention mechanisms like [[Attention with Linear Biases]] (ALiBi) and [[FlashAttention]]

Variants:
- MPT base
- MPT instruct: A model for short-form instruction following on a dataset they built and release, derived from Anthropic's Helpful and Harmless dataset.
- MPT chat: Chatbot-like model for dialogue generation
- MPT storywriter: Base models fine-tuned for 2500 steps on 65k-length-context excerpts of fiction books contained in the books3 corpus. Not released because [[Jonathan Frankle]] was worried about copywrite.

Summary
> Today, we at MosaicML are releasing a new model series called [MPT (MosaicML Pretrained Transformer)](https://github.com/mosaicml/llm-foundry) to address the limitations of the above models and finally provide a commercially-usable, open-source model that matches (and - in many ways - surpasses) LLaMA-7B. Now you can train, finetune, and deploy your own private MPT models, either starting from one of our checkpoints or training from scratch. For inspiration, we are also releasing three finetuned models in addition to the base MPT-7B: MPT-7B-Instruct, MPT-7B-Chat, and MPT-7B-StoryWriter-65k+, the last of which uses a context length of 65k tokens!
