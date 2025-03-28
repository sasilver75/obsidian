---
aliases:
  - QLoRA
---
May 23, 2023
Paper: [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
Authors include [[Tim Dettmers]]
See also: [[Low-Rank Adaptation|LoRA]], [[Parameter-Efficient Fine-Tuning]]

A technique that combines [[Low-Rank Adaptation|LoRA]] with quantization.

==This results in massive reductions in memory requirement -- enabling the training/fine-tuning of models as large as 70 billion parameters on just 2x NVIDIA RTX 3090s, which would originally take more than 16x A100-80GB GPUs!== It enables a finetuning of a 65B parameter model on a single 48GB GPU, while preserving full 16-bit fine-tuning task performance.

Abstract
> We present ==QLoRA==, an ==efficient finetuning approach== that reduces memory usage enough to ==finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance==. QLoRA backpropagates gradients through a frozen, 4-bit quantized pretrained language model into Low Rank Adapters~(LoRA). ==Our best model family, which we name== ==[[Guanaco]]==, outperforms all previous openly released models on the Vicuna benchmark, ==reaching 99.3% of the performance level of ChatGPT while only requiring 24 hours of finetuning on a single GPU==. QLoRA introduces a number of innovations to save memory without sacrificing performance: (a) 4-bit NormalFloat (NF4), a new data type that is information theoretically optimal for normally distributed weights (b) double quantization to reduce the average memory footprint by quantizing the quantization constants, and (c) paged optimziers to manage memory spikes. ==We use QLoRA to finetune more than 1,000 models==, providing a detailed analysis of instruction following and chatbot performance across 8 instruction datasets, multiple model types (LLaMA, T5), and ==model scales that would be infeasible to run with regular finetuning (e.g. 33B and 65B parameter models).== Our results show that QLoRA finetuning on a small high-quality dataset leads to state-of-the-art results, even when using smaller models than the previous SoTA. We provide a detailed analysis of chatbot performance based on both human and GPT-4 evaluations showing that GPT-4 evaluations are a cheap and reasonable alternative to human evaluation. Furthermore, we find that current chatbot benchmarks are not trustworthy to accurately evaluate the performance levels of chatbots. A lemon-picked analysis demonstrates where Guanaco fails compared to ChatGPT. ==We release all of our models and code==, including CUDA kernels for 4-bit training.


# Paper Figures
![[Pasted image 20240525153510.png]]

# Non-Paper Figures

![[Pasted image 20241215153704.png]]
