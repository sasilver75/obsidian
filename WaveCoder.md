December 20, 2023 -- [[Microsoft Research]] (6-7 months after [[HuggingFace|HF]]'s [[StarCoder]] and [[Microsoft Research|MSR]]'s [[WizardCoder]], 7 days after unrelated [[Magicoder]])
Paper: [WaveCoder: Widespread and Versatile Enhance Instruction-Tuning with Refined Data Generation](https://arxiv.org/abs/2312.14187) (Widespread And Versatile Enhanced instruction tuning) 🌊
#zotero 
Takeaway: Paper introduces an LLM-based ==Generator-Discriminator== data process framework to generate diverse, high-quality code instruction data from open-source code. Authors release their dataset of 20k instruction instances ("==CodeOcean==" dataset) across 4 universal code-related tasks, as well as ==WaveCoder==, the resulting code generation model trained on the dataset.
- AUthors use StarCoder, CodeLLaMA, and DeepseekCoder as their base models

Relevance: ....

Related:
- The focus on generating high-quality code instruction data using open source code is very similar to [[Magicoder]], which is an unrelated (by authorship, organization) paper released one week prior that aimed to do something very similar.

----

Notes:
- Authors define the following code-related tasks:
	1. ==Code Summarization== (code-to-text): Create a brief summary of code.
	2. ==Code Generation== (text-to-code): Generate code based on user's instructions.
	3. ==Code Translation== (code-to-code): Convert between programming languages.
	4. ==Code Repair== (code-to-code): Correct code based on potential issues in given code.
- We propose a method that can 1) retain the diversity of raw code to the utmost extent, and 2) an LLM-based Generator-Discriminator framework to further improve the data quality of instruction instances.
- ==Raw Code Collection==
	- To ensure quality and diversity of raw code, we define some filtering rules and use a cluster method KCenterGreedy to get our raw code collection out from the open source code dataset.
	- We use CodeSearchNet, containing 2M `(comment, code)` pairs from open-source libraries.
	1. Manual filtering
		- Length of required code should not be too long or too short; Authors keep code with length between 50..800 characters.
		- Eliminated raw code containing words from blacklist (following Code Alpaca)
		- Resulting in 1.2M pairs remaining, down from 2M
	2. Coreset-based selection method
		- We employ KCenterGreedy algorithm based on code embeddings to reduce the amount of training data while *==maximizing data diversity* of raw code==, as much as possible,  using two steps.
		1. **Code Embedding**: We encode all raw code samples using a pre-trained language model, and take the \[CLS\] token as the code embedding for one input raw code.
		2. **Coreset sampling**: Our goal is to use a small number of samples to represent the distribution of the overall data, so we employ unsupervised clustering and focused on the data representations. After obtaining the code embeddings from the previous step, we use ==KCenterGreedy algorithm to select the code dataset and maximize data diversity, aiming to choose *k* center points such that we minimize the ***largest** distance* between a random data point and its nearest center, which has been proven efficient in obtaining a set of core samples of one distribution.== 
- ==LLM-based Generator-Discriminator Framework==
	- After we have raw code, we need to generate high-quality and diverse instruction data.
	- We propose a novel LLM-based Generator-Discriminator Framework, where the 

Abstract
> Recent work demonstrates that, after being fine-tuned on a high-quality instruction dataset, the resulting model can obtain impressive capabilities to address a wide range of tasks. However, ==existing methods for instruction data generation often produce duplicate data and are not controllable enough on data quality==. In this paper, ==we extend the generalization of instruction tuning by classifying the instruction data to 4 code-related tasks and propose a LLM-based Generator-Discriminator data process framework to generate diverse, high-quality instruction data from open source code==. Hence, we introduce CodeOcean, a dataset comprising 20,000 instruction instances across 4 universal code-related tasks,which is aimed at augmenting the effectiveness of instruction tuning and improving the generalization ability of fine-tuned model. Subsequently, we present ==WaveCoder, a fine-tuned Code LLM== with Widespread And Versatile Enhanced instruction tuning. This model is specifically designed for enhancing instruction tuning of Code Language Models (LLMs). Our experiments demonstrate that Wavecoder models outperform other open-source models in terms of generalization ability across different code-related tasks at the same level of fine-tuning scale. Moreover, Wavecoder exhibits high efficiency in previous code generation tasks. This paper thus offers a significant contribution to the field of instruction data generation and fine-tuning models, providing new insights and tools for enhancing performance in code-related tasks.


# Paper Figures
![[Pasted image 20240511004137.png]]
Above: The Pipeline. See that primary portions of it are an ==LLM Generator== and an ==LLM Discriminator==.

![[Pasted image 20240511005132.png|200]]
![[Pasted image 20240511005126.png|450]]




