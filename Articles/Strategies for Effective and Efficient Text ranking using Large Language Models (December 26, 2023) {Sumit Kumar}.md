Link: https://blog.reachsumit.com/posts/2023/12/towards-ranking-aware-llms/

---

Let's take a closer look at some of the shortcomings of prompting methods, and explore the latest efforts to train ranking-aware LLMs.

## Current challenges with Prompting for Ranking
- The previous article [[Prompting-based Methods for Text Ranking Using Large Language Models (December 12, 2023) {Sumit Kumar}]] described some methods for using LLMs for the ranking task, and highlighted several issues.
	- ==Pointwise ranking== strategies do not work for generation APIs (eg GPT-4) ((?))
	- ==Pointwise ranking== also requires the model to output calibrated scores so that they can be used for comparisons in sorting. This is hard to achieve across promptings, and also unnecessary because ranking only requires relative ordering.
	- Comparing all possible pairs in a ==Pairwise ranking== strategy is computationally prohibitive.
	- ==Listwise ranking== tasks have shown to be the most difficult for lLMS, especially for smaller and even moderately-sized models, and context size limits make it unlikely to fit all possible candidate documents in the prompt (requiring strategies like sliding windows).

## Sensitivity to Initial Ordering
- Performance of LMs is significant affected by the position of relevant information ([[Positional Bias]]) and the relative ordering of candidate documents in the input context.
- Liu et al: LM performance is generally highest when relevant information/document occurs at either the very beginning (==Primacy Bias==) or end (==Recency Bias==); performance degrades when models must access and use information in the middle of their input context.
- Wang et al: Showed that GPT-4 tended to prefer the item in first position, while ChatGPT preferred the items in the second position. 
- Lu et al. Studied the effect of input order on in-context learning performance and found that the right sample order can make as much of a difference as the right templates.
- Tang et al. conducted conducted a passage ranking task analysis and concluded that different positional biases exist in reranking LLMS, varying by model and dataset.

## Potential Causes for Sensitivity to Order
- Model Architecture
	- When encoder-decoder models (_Flan-T5-XXL_, _Flan-UL2_) are evaluated on sequences that are shorter than their encoder’s training time maximum length, they are relatively more robust to changes in the position of the relevant information in their input context.
		- - ==The authors hypothesize that encoder-decoder models may make better use of their context windows because their bidirectional encoder allows processing each document in the context of future documents==, potentially improving relative importance estimation between documents, whereas decoder-only models may only attend to prior tokens.
- Query-Aware Contextualization:
	- ==When the query placed after documents, decoder-only models CANNOT attend to the query tokens when contextualizing documents, since the decoder-only models can only attend to prior tokens at each timestep.==
- Effect of Instruction Fine-Tuning
	- When doing IFT training,  task specification and/or instruction is commonly placed at the beginning of the input context in supervised instruction fine-tuning data, which might lead these instruction fine-tuned models to place more weight on the start of the input context.
- Model Size
	- Authors found that even large models had a "lost in the middle" phenomenon

## Mitigating Order Sensitivity
- Wang et al proposed a calibration framework to alleviate positional bias and achieve more reliable/fair evaluation results: ![[Pasted image 20240528182553.png]]
	- Three strategies:
	1. ==Multiple Evidence Calibration== (MEC): Authors design a template that requires the model to generate explanations first, and THEN give the core, so that scores can be calibrated with evaluation evidence.
	2. ==Balance Position Calibration== (BPC): Additional $k$ scores are calculated by swapping two responses in each sample, such as creating  query prompt (q,r2,r1) along with the original query prompt (q, r1,r2). The final calibrated scores of the two response are the AVERAGES of the scores.
	3. ==Human-in-the-Loop Calibration== (HITLC): Essentially a way of measuring the difficulty of each task (based on MEC/BPC results) and seeking human assistance when needed.
- Tang et al. apply the shuffle-aggregate paradigm of self-consistency framework to the decoding step for listwise ranking LLMs to achieve permutation invariance.
	- Candidates are shuffled to curate a diverse set of rankings. Each output ranking has positional bias, but the central ranking closest in Kendall tau distance to all of the sampled rankings is computed. Incurs additional costs due to multiple LLM calls (which can be paralellized)


## Towards Ranking-Aware LLMs
- Approaches that leverage prompt learning for LLM-based reranking have demonstrated effectiveness, but it's hard for them to outperform baseline rerankers trained/fine-tuned on benchmark datasets.
- There's a noteworthy disparity between the training objective of LLMs (centering around NTP) and the objective of evaluating query-document relevance. As a result, off-the-shelf LLMs don't understand ranking formulations well.


## Distilling LLMs for Reranking
- In RankGPT, Sun et al distilled the ranking capabilities of ChatGPT into a smaller specialized model (DeBERTa-large cross-encoder model and LLaMA-7B), using a permutation distillation scheme.
- In RankVicuna, Pradeep et al used the Vicuna model as a student model trained on ranking lists generated by RankGPT-3.5 as the teacher model. The model (7B) is on par with RankGPT-3.5 (175B).
- Authors of RankVicuna also propsed a 7B parameter RankZephyr model for listwise zero-shot reranking.


### Fine-Tuning LLMs for Reranking
- Ma et al. proposed a zero-shot multi-stage ranking pipeline with a dense retriever (RepLLaMA) and a point-wise reranker (RankLLaMA), both based on finetuning the latest LLaMa on MS MARCO datasets.
	- RepLLaMA follows the bi-encoder dense retriever architecture, but with the backbone initialized with LLaMA.
	- RankLLaMA takes a query and a candidate document and produces a score indicating relevance; model is optimized by contrastive loss, and hard negatives are sampled from the top-ranking results from the retriever.
- In RankingGPT, authors proposed a supervised training strategy to improve LLM ranking capability. They construct a large-scale dataset of weakly-supervised text pairs using web resources to continaully pretrain the model, and then later finetune using MS MARCO and contrastive learning.







