Github: [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)

Compare: [[Unsloth]]

Axolotl is an LLM fine-tuner supporting SotA techniques and optimizations for a variety of common model architectures.

Features
- Allows training of various HugginfFace models like LLaMA, Pythia, Falcon MPT
- Supports full finetuning, [[Low-Rank Adaptation|LoRA]], [[Quantized Low-Rank Adaptation|QLoRA]], Relora, and GPTQ
- Allows customizable configurations using simple YAML files or CLI
- Integrated with xformer, [[FlashAttention]], [[Rotary Positional Embedding|RoPE]] scaling, and multipacking.
- Works with a single GPU or multiple GPUs via FSDP or Deepspeed
- Easily run with Docker locally or on the cloud
- Log results and optionally checkpoints to `wandb` or `mlflow`



![[Pasted image 20240108224754.png]]
![[Pasted image 20240108224828.png]]
(Above: Roughly 11/2023)


It's used by many of the leading open source models.








