October 25, 2023
[[HuggingFace]] H4 (Helpful, Honest, Harmless, Huggy) Team (including [[Nathan Lambert]])
Paper: [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)
#zotero 
Takeaway: Explores using a combination of [[Distillation]] and [[Direct Preference Optimization|DPO]] (which they call dDPO), where an ensemble of strong models give AI Feedback (AIF, rather than human feedback (HF)) regarding preference, and then authors use the resulting dataset to align [[Mistral 7B]]. 

Note: There's a model called Notus, which is a variant of Zephyr with better filtered/fixed data.
Q: What is Zephyr-Beta?

---

## Introduction
- Distillation has been a powerful tool for improving open model performance, but it stops short of letting open models reach the performance levels of their teachers, and there hasn't been much success in distillation of *alignment*-related behavior.
- This paper considers the problem of aligning a small open-LLM entirely through distillation, using AI feedback from an ensemble of strong models as preference data, and then applying a distilled DPO optimization technique. Authors combine [[Mistral 7B]] with the [[UltraChat]] and [[UltraFeedback]] datasets to produce [[Zephyr]] 8B.
- [[Self-Instruct]] and the [[Alpaca]] model started a trend of improving smaller model performance via distillation from larger models. This was followed by [[Vicuna]] and other models who primarily focused on the SFT stage of alignment, and other methods like [[WizardLM]]/[[Evol-Instruct]] that experimented with other methods of getting SFT data.
- We generate 



Abstract
> We aim to produce a smaller language model that is aligned to user intent. Previous research has shown that applying distilled supervised fine-tuning (dSFT) on larger models significantly improves task accuracy; however, these models are unaligned, i.e. they do not respond well to natural prompts. To distill this property, we experiment with the use of preference data from AI Feedback (AIF). ==Starting from a dataset of outputs ranked by a teacher model, we apply distilled direct preference optimization (dDPO) to learn a chat model with significantly improved intent alignment.== The approach requires only a few hours of training without any additional sampling during fine-tuning. ==The final result, Zephyr-7B, sets the state-of-the-art on chat benchmarks for 7B parameter models==, and ==requires no human annotation==. In particular, results on MT-Bench show that Zephyr-7B surpasses Llama2-Chat-70B, the best open-access RLHF-based model. Code, models, data, and tutorials for the system are available at [this https URL](https://github.com/huggingface/alignment-handbook).


# Paper Figures
![[Pasted image 20240712181603.png|400]]
Comparison of Zephyr-7B against much larger models. See that it seems to outperform LLaMA2-Chat-70B.

![[Pasted image 20240712183600.png|500]]
SFT then DPO on a dataset scored by GPT-4 in an LLM-as-a-Judge capacity.



# Non-Paper Figures
![[Pasted image 20240418171918.png]]
