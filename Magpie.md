June 12, 2024
University of Washington, [[Allen Institute]] (*Xu et al.*) - Includes [[Yejin Choi]]
[Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
#zotero 
Takeaway: Built on the simple observation that if you only input the system template and a blank user template, language models will self-synthesize a plausible user query.


---
## Introduction
- The effectiveness of instruction tuning depends crucially on access to high-quality instruction datasets, but alignment datasets typically used to fine-tuned models like [[LLaMA 3]]-Instruct are typically private.


## Magpie: A Scalable Method to Synthesize Instruction Data


## Dataset Analysis


## Performance Analysis


## Related Work


## Limitations and Ethical Considerations


## Conclusions


Abstract
> High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a ==self-synthesis method for generating large-scale alignment data== named ==Magpie==. ==Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages==, thanks to their auto-regressive nature. We use this method to prompt ==Llama-3-Instruct and generate 4 million instructions along with their corresponding responses==. We perform a comprehensive analysis of the extracted data and ==select 300K high-quality instances==. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that ==in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT)== and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with [[UltraFeedback]]. This advantage is evident on alignment benchmarks such as [[AlpacaEval]], [[Arena-Hard]], and [[WildBench]].


# Paper Figures

![[Pasted image 20240723224506.png|500]]
- Given an empty user prompt, get an instruction
- Given a user prompt with the instruction and empty assistant prompt, get response
- Accumulate the instruction/response pair

Below: Examples of Magpie system prompts that can be used to condition the user prompts that get generated
![[Pasted image 20240723224043.png|500]]
![[Pasted image 20240723224057.png|500]]
![[Pasted image 20240723224114.png|500]]
![[Pasted image 20240723224132.png|500]]
Above: Examples of Magpie system prompts that can be used to condition the user prompts that get generated


Below: Generations from ==Magpie-Pro== across various task categories.
![[Pasted image 20240723223230.png|600]]
![[Pasted image 20240723223243.png|600]]
Above: Generations from ==Magpie-Pro== across various task categories.