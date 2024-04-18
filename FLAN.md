September 3, 2021
Paper: [Finetuned Language Models are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) (FLAN = "Finetuned Language Network")

FLAN is both a model and a dataset (they only release a partial dataset though, as the [[Pile-T5]] authors discovered)

Abstract
> This paper explores a simple method for improving the zero-shot learning abilities of language models. ==We show that instruction tuning== -- finetuning language models on a collection of tasks described via instructions -- ==substantially improves zero-shot performance on unseen tasks==.  
> We take a ==137B parameter pretrained language model and instruction-tune it on over 60 NLP tasks== verbalized via natural language instruction templates. We evaluate this instruction-tuned model, which we call FLAN, on unseen task types. FLAN substantially improves the performance of its unmodified counterpart and surpasses zero-shot 175B GPT-3 on 20 of 25 tasks that we evaluate. FLAN even outperforms few-shot GPT-3 by a large margin on ANLI, RTE, BoolQ, AI2-ARC, OpenbookQA, and StoryCloze. Ablation studies reveal that number of finetuning datasets, model scale, and natural language instructions are key to the success of instruction tuning.

![[Pasted image 20240417224708.png]]

Formatted more than 60 high-quality NLP datasets into instruction-following datasets. Aims to capture the prior of understanding and respecting textual instructions.