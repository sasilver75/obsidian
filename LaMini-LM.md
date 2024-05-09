April 27, 2023 (6 months after ChatGPT, 1 month after [[Alpaca]])
Mohammad bin Zayed University of AI (MBZUAI) (in the UAE)
Paper: [LaMini-LM: A Diverse Herd of Distilled Models from Large-Scale Instructions](https://arxiv.org/abs/2304.14402) 🐂🐄🐎
#zotero 
Takeaway: The paper explores the power of distilling capable frontier models by using them to respond to large of instruction-tuning data (2.58M instructions, also released with the paper), and then using those responses to fine-tune a variety of smaller models of varying architectures, which all end up punching above their weight.

Relevance: I'm trying to figure this out still, because this came out a month after Alpaca, which similarly distill-fine-tuned LLaMA using GPT-3.5. Did we need to do this with N different model architectures to show that it worked for all of them? It's *just* training on data.

----

Notes:
- Authors create an instruction-tuning dataset of ==2.58M examples==, first curating it from instructions from diverse existing datasets, including [[Self-Instruct]], P3, [[FLAN v2]], and [[Alpaca]].
	- Authors augment the dataset using :
		- ==Example-Guided Instruction Generation==: Using `gpt-3.5-turbo` to generate additional diverse instructions that match human-written ones in style and quality.
		- ==Topic-Guided Instruction Generation==: Technique to enhance instruction *diversity* by incorporating specific topics of interest from Wikipedia.
	- Finally, for all instructions, we use `gpt-3.5-turbo` to generate responses for each instruction, resulting in the ==LaMini instruction dataset==.
- Afterwards, we fine=tune multiple smaller language models with sizes ranging from 61M to 7B, across a both encoder-decoder and decoder-only, conducting extensive experiments and analyses, setting our work apart from previous analyses.
- 

Abstract
> Large language models (LLMs) with instruction fine-tuning demonstrate superior generative capabilities. However, these models are resource-intensive. To alleviate this issue, we explore distilling knowledge from instruction-tuned LLMs into much smaller ones. To this end, ==we carefully develop a large set of 2.58M instructions based on both existing and newly-generated instructions==. In addition to being sizable, we design our instructions to cover a broad set of topics to ensure diversity. Extensive analysis of our instruction dataset confirms its diversity, and ==we generate responses for these instructions using gpt-3.5-turbo==. Leveraging these instructions, ==we fine-tune a diverse herd of models, collectively referred to as LaMini-LM, which includes models from both the encoder-decoder and decoder-only families, with varying sizes==. We evaluate the performance of our models using automatic metrics on 15 different natural language processing (NLP) benchmarks, as well as through human assessment. The results demonstrate that our proposed ==LaMini-LM models are comparable to competitive baselines, while being much smaller in size==.

# Paper Figures
![[Pasted image 20240508223359.png|250]]





# Non-Paper Figures
