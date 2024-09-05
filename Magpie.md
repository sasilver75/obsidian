June 12, 2024
University of Washington, [[Allen Institute]] (*Xu et al.*) - Includes [[Yejin Choi]]
[Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing](https://arxiv.org/abs/2406.08464)
#zotero 
Takeaway: Built on the simple observation that if you only input the system template and a blank user template, language models will self-synthesize a plausible user query. We can then feed that back into a model to get a response. Authors create large [Magpie-Air](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-300K-Filtered) and [Magpie-Pro](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-300K-Filtered)datasets doing this with LLaMA 3 8B-Instruct and LLaMA 3 70B-Instruct. The resulting 300k  instances (after filtering) result in a comparable finetune of LLaMA 3 8B-Base to the official LLaMA 3 8B-Instruct dataset (which was trained on 10M SFT and feedback learning). SFT with Magpie even surpasses what is considered to be one of the best open alignment datasets, [[UltraFeedback]]!
- Authors also use this technique to generate two multi-turn instruction datasets, [Magpie-Air-MT](https://huggingface.co/datasets/Magpie-Align/Magpie-Air-MT-300K-v0.1) and [Magpie-Pro-MT](https://huggingface.co/datasets/Magpie-Align/Magpie-Pro-MT-300K-v0.1).
- See full Magpie collection [here](https://huggingface.co/Magpie-Align)
- It seems from Figure 6 that Magpie data creates models good at Creative Tasks, Reasoning and Planning, and Info Seeking, but not Math and Data or Coding and Debugging.


	Update (August 2, 2024): Since then. LLaMA 3.1 was released, so HuggingFace folks used [[LLaMA 3.1]] 405B and [[Distilabel]] to create a new ==Magpie Ultra v0.1== ([described here](https://x.com/gabrielmbmb_/status/1819398254867489001)), the first public synthetic dataset created with LLaMA 3.1 405B. Includes 50k *unfiltered rows* with instruction-response pairs across {information seeking, reasoning, planning, editing, coding and debugging, math, data analysis, creative writing, advice-seeking, brainstorming, and more}. They are working to bring out a filtered version. The dataset can be used for SFT, but also for RLAIF, because they included both an instruct model and base model response for each instruction.

---
## Introduction
- The effectiveness of instruction tuning depends crucially on access to high-quality instruction datasets, but alignment datasets typically used to fine-tuned models like [[LLaMA 3]]-Instruct are typically private.
- ==Key Insight==: Authors observe that when you only input the pre-query template to an aligned LLM like LLaMA 3-Instruct, they self-synthesize a user query due to their autoregressive nature! 
	- Then you can feed that generated query back into the model to get a response, and now you have a (query, response) instruction pair!
	- Authors do this with both LLaMA 3 8B-Instruct and 70B-Instruct and generate two instruction datasets: ==Magpie-Air== and ==Magpie-Pro==, respectively.
	- Authors also generate two ==multi-turn instruction datasets==, ==Magpie-Air-MT== and ==Magpie-Pro-MT==!
		- ((This seems really cool as a way to generate multiturn datasets!))
- To compare the Magpie dataset with existing public instruction datasets (ShareGPT, WildChat, Evol-Instruct, UltraChat, OpenHermes, Tulu V2 Mix) and preference tuning strategies with UltraFeedback, we fine-tune the LLaMA-3-8B Base model with each dataset and assess the performance of the resultant models on LLM alignment benchmarks like AlpacaEval 2, Arena-Hard, and WildBench.
	- Models finetuned with Magpie achieve superior performance, ==even surpassing the official LLaMA-3-8B-Instruct model on AlpacaEval (when it's trained on Magpie-Pro, which is basically distillation)==
	- It beats (with SFT alone) prior public datasets that include both SFT and preference optimization (eg with UltraFeedback, whether using DPO, IPO, KTO, or ORPO!) 

## Magpie: A Scalable Method to Synthesize Instruction Data
- Instruction Generation
	- Magpie crafts an input query in the format of the predefined instruction template of the LLM. This query defines only the role of the instruction provider (eg user), but doesn't provide an instruction.
	- LM generates an instruction when the query crafted by Magpie is given as an input. This can be done many times to generate a set of instructions.
- Response Generation
	- The goal is to generate responses to the instructions obtained from Step 1. Magpie sends these to the LLM to generate the corresponding responses.
	- Authors do a tSNE Plop comparing Magpie-Pro with Alpaca/Evol-Instruct/Ultrachat; the t-SNE of Magpie-Pro encompasses the area covered by the other datasets' plots, implying a diverse range of topics.

## Dataset Analysis
- See figures

## Performance Analysis
- Authors compare to six open instruction-tuning datasets:
	- [[ShareGPT]] (human, multiturn, 112k)
	- [[WildChat]] (human, multiturn, 652k)
	- [[Evol-Instruct]] (synth)
	- [[UltraChat]] (synth, 208k sanitized)
	- [[OpenHermes Dataset]] (crowd-sourced, 243k)
	- [[Tulu-v2-sft-mixture]] (crowd-sourced, 326k)
- Authors compare models finetuned using Magpie-generated data with preference optimization baselines, including  [[Direct Preference Optimization|DPO]], [[Identity-Mapping Preference Optimization|IPO]], [[Kahneman-Tversky Optimization|KTO]], and [[Odds Ratio Preference Optimization|ORPO]]... They compare against models finetuned with [[UltraChat]] (IFT) and [[UltraFeedback]] (preference optimization).
- Models finetuned with Magpie data achieve comparable performance to official aligned models, but with fewer data.


Abstract
> High-quality instruction data is critical for aligning large language models (LLMs). Although some models, such as Llama-3-Instruct, have open weights, their alignment data remain private, which hinders the democratization of AI. High human labor costs and a limited, predefined scope for prompting prevent existing open-source data creation methods from scaling effectively, potentially limiting the diversity and quality of public alignment datasets. Is it possible to synthesize high-quality instruction data at scale by extracting it directly from an aligned LLM? We present a ==self-synthesis method for generating large-scale alignment data== named ==Magpie==. ==Our key observation is that aligned LLMs like Llama-3-Instruct can generate a user query when we input only the left-side templates up to the position reserved for user messages==, thanks to their auto-regressive nature. We use this method to prompt ==Llama-3-Instruct and generate 4 million instructions along with their corresponding responses==. We perform a comprehensive analysis of the extracted data and ==select 300K high-quality instances==. To compare Magpie data with other public instruction datasets, we fine-tune Llama-3-8B-Base with each dataset and evaluate the performance of the fine-tuned models. Our results indicate that ==in some tasks, models fine-tuned with Magpie perform comparably to the official Llama-3-8B-Instruct, despite the latter being enhanced with 10 million data points through supervised fine-tuning (SFT)== and subsequent feedback learning. We also show that using Magpie solely for SFT can surpass the performance of previous public datasets utilized for both SFT and preference optimization, such as direct preference optimization with [[UltraFeedback]]. This advantage is evident on alignment benchmarks such as [[AlpacaEval]], [[Arena-Hard]], and [[WildBench]].


# Paper Figures

![[Pasted image 20240723232541.png]]
Above: Comparison of the Magpie datasets to other popular open datasets.

![[Pasted image 20240723224506.png|500]]
- Given an empty user prompt, get an instruction
- Given a user prompt with the instruction and empty assistant prompt, get response
- Accumulate the instruction/response pair

![[Pasted image 20240723233730.png]]
Impressive coverage compared to other synthetic datasets... wonder how it compares to real/human datasets like ShareGPT.

![[Pasted image 20240723234210.png|300]]
This is pre-filtering, looking at the LLaMA-3-8B-Instruct model assessments of difficulty.

![[Pasted image 20240723234322.png|300]]
Again, pre-filtering, showing the similarity among instructions in embedding space (using `all-mpnet-base-v2`). Also shows the "reward difference", which is the difference between the reward assigned by a reward model to the *response in our dataset*, and the reward assigned to the response generated for the same response by LLaMa-3 Base.

![[Pasted image 20240724000116.png|500]]

![[Pasted image 20240724000134.png|500]]


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