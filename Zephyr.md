October 25, 2023
[[HuggingFace]] H4 (Helpful, Honest, Harmless, Huggy) Team (including [[Nathan Lambert]])
Paper: [Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944)
#zotero 
Takeaway: Explores using a combination of [[Distillation]] and [[Direct Preference Optimization|DPO]] (which they call dDPO), where an *ensemble of strong models* give AI Feedback (AIF, rather than human feedback (HF)) regarding preference, and then authors use the resulting *dataset* (pairs of (highest_score_generation, random_lower_score_generation)) to align [[Mistral 7B]] using DPO. 

> "This paper is mostly about making DPO mainstream" - Nathan Lambert


Note: The Zephyr in this paper is actually ==Zephyr-Alpha== (finetune of [[Mistral 7B]]); later, a ==[Zephyr-Beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)== (finetune of [[Mistral 7B]]) and a ==[Zephyr-Gemma](https://huggingface.co/HuggingFaceH4/zephyr-7b-gemma-v0.1)== (finetune of [[Gemma]] 7B)
Note: There's a model from [[Argilla]] called [Notus 7B](https://huggingface.co/argilla/notus-7b-v1), which is an analogue to Zephyr-Beta with a better preference dataset used for dDPO.

---

## Introduction and Related Work
- Distillation has been a powerful tool for improving open model performance, but it stops short of letting open models reach the performance levels of their teachers, and there hasn't been much success in distillation of *alignment*-related behavior.
- This paper considers the problem of aligning a small open-LLM entirely through distillation, using AI feedback from an ensemble of strong models as preference data, and then applying a distilled DPO optimization technique. Authors combine [[Mistral 7B]] with the [[UltraChat]] and [[UltraFeedback]] datasets to produce [[Zephyr]] 8B.
- [[Self-Instruct]] and the [[Alpaca]] model started a trend of improving smaller model performance via distillation from larger models. This was followed by [[Vicuna]] and other models who primarily focused on the SFT stage of alignment, and other methods like [[WizardLM]]/[[Evol-Instruct]] that experimented with other methods of getting SFT data.

## Method
- ==Distilled Supervised Fine-Tuning (dSFT)== : Given a set of seed prompts representing a diverse set of topical domains, we generate a dataset through iterative self-prompting, where ==the teacher model is used to BOTH respond to an instruction AND refine the instruction based on the response.== For each x0, we sample a response y0, and then sample a new instruction (using a prompt for refinement) x1. Distillation on this dataset is then performed by SFT.
	- Trained for 1-3 epochs. Uses a cosine learning rate scheduler with a peak LR of 2e-5 and 10% warmup. We train with a global bs=512 and use packing with a sequence length of 2048 tokens
- ==AI Feedback through Preferences (AIF)==: Human feedback is usually used to provide signal to align LLMs; in our dDPO, we're going to instead use AI preferences from the teacher model on generated outputs from other models. With a set of prompts, each prompt is fed to a collection of four models, each of which yield a response. These are then fed to the teacher model (eg GPT-4) which gives a score for the response. We save the highest scoring response as y_w and a random lower-scoring score as y_l; the final feedback dataset consists of triplets of (x, y_l, y_w).
	- ((I'm worried here about generated outputs from other models. We're climbing uphill or downhill from some selected or non-selected generation, except there's a distribution mismatch because the model we're selecting winning and losing completions form isn't the model being finetuned. It's pretty common to use third-party alignment datasets (which is basically what this is ), but distribution mismatch means that preferences learned might not be as applicable to our model.))
- ==Distilled Direct Preference Optimization (dDPO)==:  Goal is to refine the $\pi_{SFT}$ model by maximizing the likelihood of ranking the preferred $y_w$ over $y_l$.   This looks complex, but it implies a simple training procedure. We compute the probability of (x, y_w) and (x, y_l) for both the dSFT and dDPO model, and compute the below equation and backpropagate to update.
	- ![[Pasted image 20240712222112.png|600]]
	- They train the DPO model for 1-3 epochs, using a linear learning rate schedule with a peak LR of 5e-7 and 10% warmup steps. Trained with a bs=32 and Beta=0.1 to control deviation from reference model.

## Experimental Details
- They finetune [[Mistral 7B]], which at the time was the SoTA base LM at the 7B scale. They use the [[TRL|Transformer Reinforcement Learning]] (TRL) library from HF for fine-tuning, with DeepSped ZeRO-3 and FlashAttention-2 to optimize memory an d improve training speed. All models re trained with [[AdamW]] and no weight decay. They didn't experiment with PEFT techniques like [[Low-Rank Adaptation|LoRA]].
- Focus on two dialogue datasets distilled from a mix of open and proprietary models, previously been shown to produce strong chat models like UltraLM:
	1. [[UltraChat]]: A self-refinement dataset of 1.47M multi-turn dialogues generated by GPT-3.5-Turbo over 30 topics and 20 types of text material. Initially after running dSFT over the whole corpus, they got a model that would preface most of its answers with "I don't have personal experiences," and "as a LLM," even for questions where it doesn't make sense to do that. We applied filters to focus on grammatical error (5% of the dataset), helpfulness and remove undesired model responses, resulting in a dataset containing only 200k examples. ((Surprising that it's a such a low percentage!))
	2. [[UltraFeedback]]: Consists of 64k prompts, each of which have 4 LLM responses that are rated by GPT-4 according to criteria like instruction-following, honesty, and helpfulness. In this paper, we construct binary preferences from UltraFeedback by selecting the highest mean core as the "chosen" response, and one from the remaining three at random as the "rejected." We opt for random selection instead of selecting the lowest-scored response to encourage diversity and make the DPO objective more challenging.
- They evaluate on [[MT-Bench]] and [[AlpacaEval]], as well as on the [[Open LLM Leaderboard]], which at the time of writing includes [[Abstraction and Reasoning Corpus|ARC]], [[HellaSWAG]], [[MMLU]], and [[TruthfulQA]].

## Conclusions and limitations
- Our method avoids the use of sampling-based approaches like [[Rejection Sampling]] or [[Proximal Policy Optimization|PPO]] and distills conversational capabilities with [[Direct Preference Optimization|DPO]] directly from a set of AI feedback.
- Zephyr7B sets a new SoTA for 7B parameter chat models, outperforming even LLaMA2-Chat-70B on MT-Bench.
- Limitations: Use of GPT-4 as an evaluator for the AlpacaEval and MT-Bench benchmarks, which are known to be biased towards models distilled from it (and the usual LM-as-a-Judge biases).



Abstract
> We aim to produce a smaller language model that is aligned to user intent. Previous research has shown that applying distilled supervised fine-tuning (dSFT) on larger models significantly improves task accuracy; however, these models are unaligned, i.e. they do not respond well to natural prompts. To distill this property, we experiment with the use of preference data from AI Feedback (AIF). ==Starting from a dataset of outputs ranked by a teacher model, we apply distilled direct preference optimization (dDPO) to learn a chat model with significantly improved intent alignment.== The approach requires only a few hours of training without any additional sampling during fine-tuning. ==The final result, Zephyr-7B, sets the state-of-the-art on chat benchmarks for 7B parameter models==, and ==requires no human annotation==. In particular, results on MT-Bench show that Zephyr-7B surpasses Llama2-Chat-70B, the best open-access RLHF-based model. Code, models, data, and tutorials for the system are available at [this https URL](https://github.com/huggingface/alignment-handbook).


# Paper Figures
![[Pasted image 20240712181603.png|400]]
Comparison of Zephyr-7B against much larger models. See that it seems to outperform LLaMA2-Chat-70B.

![[Pasted image 20240712183600.png|500]]
SFT then DPO on a dataset scored by GPT-4 in an LLM-as-a-Judge capacity.

![[Pasted image 20240713000351.png|600]]
Comparison against similar 7B models, larger open models up to 70B, and against frontier models.



# Non-Paper Figures
![[Pasted image 20240418171918.png]]
