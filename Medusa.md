January 19, 2024
Princeton, [[Together AI]], UIUC, CMU, UConn (incl. [[Tri Dao]])
[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
#zotero 
Takeaway: ...

---

### Introduction
- Two options to speed up LLM inference include *increasing the arithmetic intensity* (ratio of FLOPs to total data movement), and *reducing the number of decoding steps.*
- [[Speculative Decoding]] uses a smaller draft model to generate token sequences, which are then refined/approved by the larger model for acceptable continuation -- but it's hard to obtain an appropriate draft model, and even harder to integrate it into a distributed system.
- Instead of a draft model, this paper introduced the concept of using ==multiple decoding heads== on top of the backbone model to expedite inference, concurrently predicting multiple tokens.
	- These heads are finetuned in a parameter-efficient manner and can be added to any existing model.
- Key insights of Medusa:
	1. The current approach of generating a single candidate continuation at each decoding step is an inefficient use of resources. We propose generating multiple candidate continuations and verifying them concurrently.
	2. We *could* reuse the [[Rejection Sampling]] scheme as used in [[Speculative Decoding]] in order to generate consistent responses with the same distribution as the original model. We *instead* introduce a *==typical acceptance scheme==* that selects *reasonable* candidates from the Medusa head outputs, and use temperature as a threshold to manage deviation from the original model's predictions, providing an efficient alternative to the usual rejection sampling method.
- Two distinct fine-tuning procedures to equip LLMs with Medusa heads:
	1. If you have limited computational resources, ==Medusa-1== requires minimal memory, but the full potential of the backbone model isn't utilized.
	2. In Medusa-2, for situations with ample computational resources, they use a training protocol that enables joint training of the Medusa heads and the backbone model, without compromising the model's predictive ability.
- Authors also propose a [[Self-Distillation]] approach for when a training dataset isn't available (or if the model has already underwent RLHF).
- Authors test Medusa on [[Vicuna]]-7B, Vicuna-13B, Vicuna-33B, and [[Zephyr]]-7B.

### Methodology
- Follows a similar framework to [[Speculative Decoding]], where each decoding step consists primarily of three substeps:
	1. Generating candidates
	2. Proposing candidates
	3. Accepting candidates
	- The difference being that in Medusa, (2) is realize by tree attention, and since medusa heads are on top of the original model, the logits calculated in (2) can be used for substep (1) for the next decoding step. The final step can be done by either [[Rejection Sampling]] or ==typical acceptance==.
- Medusa Heads
	- In speculative decoding, subsequent tokens are predicted by an auxiliary draft model that needs to be small yet effective enough to generate continuations that will be accepted by the original model -- this is a difficult balance, and often requires even separately *pre-training* ($, eg via distillation) a smaller model! If there's a distribution mismatch between draft and main model, it might lead to continuations that the main model doesn't accept!
	- Medusa heads in contrast are additional decoding heads appended to the last hidden states of the original model. We add K decoding heads, with the k'th head being used to predict the token in the t+k+1-th position of the next tokens.
		- ((Meaning if we have 3 heads, they're collectively predicting the next 3 tokens))
		- Each head utilizes a single layer of FFNNs with a residual connection for each head. Authors find this simple design is sufficient to achieve satisfactory performance.
		- They use the [[SiLU]] activation function.
	- Medusa draft heads are trained in conjunction with the original backbone model, ensuring that the distribution of the Medusa heads aligns with that of the original model:
		- Medusa-1: The backbone remains frozen while Medusa heads train
		- Medusa-2: The backbone is trained together with the Medusa heads
- Tree Attention
	- Through our Medusa heads, we obtain probability predictions for the subsequent few tokens, we consider multiple tokens for each position, resulting in multiple possible continuations/candidates.
	- To strike a balance between this multiple-continuation strategy and the higher computational demands, we employ a ==three-structured attention mechanism== to process multiple candidates concurrently.
	- Only tokens from the same continuation are regarded as historical data.
	- For a given k'th head, its top-s_k predictions serve as the basis for candidate formulation, where s_k is a designated hyperparameter.
	- Our candidate sequences are established by determining the Cartesian product of the top-s_k predictions from each head.
		- Within this tree, only a token's predecessors are seen as historical context, 


Abstract
> Large Language Models (LLMs) employ auto-regressive decoding that requires sequential computation, with each step reliant on the previous one's output. This creates a bottleneck as each step necessitates moving the full model parameters from High-Bandwidth Memory (HBM) to the accelerator's cache. While methods such as [[Speculative Decoding]] have been suggested to address this issue, their implementation is impeded by the challenges associated with acquiring and maintaining a separate draft model. In this paper, we present ==Medusa==, an efficient method that augme==nts LLM inference by adding extra decoding heads to predict multiple subsequent tokens in parallel==. Using a ==tree-based attention mechanism==, Medusa ==constructs multiple candidate continuations and verifies them simultaneously in each decoding step==. By leveraging parallel processing, Medusa substantially reduces the number of decoding steps required. We present two levels of fine-tuning procedures for Medusa to meet the needs of different use cases: ==Medusa-1==: Medusa is directly fine-tuned on top of a frozen backbone LLM, enabling lossless inference acceleration. ==Medusa-2==: Medusa is fine-tuned together with the backbone LLM, enabling better prediction accuracy of Medusa heads and higher speedup but needing a special training recipe that preserves the backbone model's capabilities.
> Moreover, we propose several extensions that improve or expand the utility of Medusa, including a [[Self-Distillation]] to handle situations where no training data is available and a typical acceptance scheme to boost the acceptance rate while maintaining generation quality. We evaluate Medusa on models of various sizes and training procedures. Our experiments demonstrate that Medusa-1 can achieve over 2.2x speedup without compromising generation quality, while Medusa-2 further improves the ==speedup to 2.3-3.6x==.


# Paper Figures
![[Pasted image 20240711144612.png|300]]
Medusa includes multiple prediction heads, each of which generates multiple predictions for its designated position. These predictions are assembled into sequence candidates, and a ==tree-based attention mechanism== is used to process them in parallel. It seems we can either use a standard [[Rejection Sampling]] scheme or a *typical acceptance* scheme.

# Non-Paper Figures
