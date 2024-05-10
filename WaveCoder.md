December 20, 2023 -- [[Microsoft Research]] (6-7 months after [[HuggingFace|HF]]'s [[StarCoder]] and [[Microsoft Research|MSR]]'s [[WizardCoder]], 7 days after unrelated [[Magicoder]])
Paper: [WaveCoder: Widespread and Versatile Enhance Instruction-Tuning with Refined Data Generation](https://arxiv.org/abs/2312.14187) (Widespread And Versatile Enhanced instruction tuning)
#zotero 
Takeaway: Paper introduces an LLM-based ==Generator-Discriminator== data process framework to generate diverse, high-quality code instruction data from open-source code. Authors release their dataset of 20k instruction instances ("==CodeOcean==" dataset) across 4 universal code-related tasks, as well as ==WaveCoder==, the resulting code 

Relevance: ....

Related:
- The focus on generating high-quality code instruction data using open source code is very similar to [[Magicoder]], which is an unrelated (by authorship, organization) paper released one week prior that aimed to do something very similar.

----

Notes:
- 

Abstract
> Recent work demonstrates that, after being fine-tuned on a high-quality instruction dataset, the resulting model can obtain impressive capabilities to address a wide range of tasks. However, ==existing methods for instruction data generation often produce duplicate data and are not controllable enough on data quality==. In this paper, ==we extend the generalization of instruction tuning by classifying the instruction data to 4 code-related tasks and propose a LLM-based Generator-Discriminator data process framework to generate diverse, high-quality instruction data from open source code==. Hence, we introduce CodeOcean, a dataset comprising 20,000 instruction instances across 4 universal code-related tasks,which is aimed at augmenting the effectiveness of instruction tuning and improving the generalization ability of fine-tuned model. Subsequently, we present ==WaveCoder, a fine-tuned Code LLM== with Widespread And Versatile Enhanced instruction tuning. This model is specifically designed for enhancing instruction tuning of Code Language Models (LLMs). Our experiments demonstrate that Wavecoder models outperform other open-source models in terms of generalization ability across different code-related tasks at the same level of fine-tuning scale. Moreover, Wavecoder exhibits high efficiency in previous code generation tasks. This paper thus offers a significant contribution to the field of instruction data generation and fine-tuning models, providing new insights and tools for enhancing performance in code-related tasks.