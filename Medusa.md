January 19, 2024
Princeton, [[Together AI]], UIUC, CMU, UConn (incl. [[Tri Dao]])
[Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774)
#zotero 
Takeaway: A method for increasing speeding up output generation that maintains output quality comparable to real model while increasing speed ~2-3x and requiring minimal changes to the existing LLM architecture by adding additional decoding heads. The heads can be fine-tuned separately with a frozen backbone, or jointly trained with the backbone (with some smart learning rates/warmup). The multiple decoding heads each predict a future token/position in parallel with the main model's processing, and their token probability distributions (with thresholding) construct a tree-like structure of potential token sequences. (Unsure): We can then feed the candidate sequences back through the model to see if it agrees, effectively jumping our generation forwards a few steps in a single forward pass (there's also something like a confidence score associated with these candidate sequences, and we can threshold on that).

References:
- Video: [OxenAI's How Medusa Works](https://www.youtube.com/watch?v=Jjjn-J9SJ1s&t=4s) esp 43:00

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
			- ((Authors note that there might be other ways to construct a tree than with a Cartesian product))
		- Within this tree, only a token's predecessors are seen as historical context (based on how the attention mask is constructed).
		- By employing this mask and properly setting the positional indices for positional encoding, we can process numerous candidates simultaneously without the need to expand the batch size.
- Training Strategies
	- At the most *basic level*, they train Medusa heads by freezing the backbone model and fine-tuning medusa heads. (Medusa-1; low-compute)
	- But we can also train the backbone in conjunction with the Medusa heads to significantly enhance the accuracy of the Medusa heads. (Medusa-2; ample compute)
	- Medusa-1: Frozen Backbone 🧊
		- We use a [[Cross-Entropy]] loss between the prediction of the Medusa heads and the ground truth. Given a ground-truth token $y_{t+k+1}$ at position $t+k+1$, the loss for the kth head is $\mathcal{L_k} = -logp_t^k(y_t+k+1)$ ... where the $p_t^k(y)$ denotes the probability of token y predicted by the kth head.
			- Note that L_k is larger when k is larger, which is reasonable since the prediction of the kth head is more uncertain when k is larger. So we add a weight $\lambda_k$ to the loss to balance the loss of different heads. Add a total Medusa loss:![[Pasted image 20240711171755.png|500]]
			- In practice, they set $\lambda_k$ as the kth power of a constant, like .8.
	- Medusa-2: Joint Training 🚬
		- To further improve accuracy, we can train the medusa heads together with the backbone model -- but this requires a special training recipe to preserve the backbone's next-token-prediction capabilities and output quality. We use three strategies:
			1. ==Combined loss==: We add the cross-entropy loss of the backbone model to the Medusa loss, and also add a weight $\lambda_0$ to balance the loss of the backbone model and the Medusa heads, so the total loss is: $\mathcal{L}_{Medusa2}=\mathcal{L}_LM + \lambda_0\mathcal{L}_{Medusa1}$ 
			2. ==Differential learning rates==: Since the backbone model is already well-trained relative to the Medusa heads, we use separate learning rates for them to enable faster convergence of Medusa heads, while preserving backbone model's capability.
			3. ==Heads warmup==: Notice that at the beginning of training, Medusa heads have large loss, leading to a large gradient that might distort the backbone model's parameters. Thusly, they use a two-stage training process where in the first stage they only train the medusa heads (As in medusa-1), and in the second stage they train the backbone model and heads together, using a warmup strategy: They train the backbone model for a few epochs first, then train the Medusa heads together with the backbone model.
- Selecting the number of heads
	- Empirically, ==we found that five heads are sufficient at most== -- they recommend training with five heads and referring to the strategy described later to determine the optimal configuration of the tree attention. With optimizer tree attention, sometimes three or four heads might be enough for inference.
- Extensions
	- ==Typical Acceptance== strategy
		- In [[Speculative Decoding]] papers, authors usually employ [[Rejection Sampling]] to yield diverse outputs that align with the distribution of the original model. Subsequent papers reveal this strategy results in diminished efficiency as the sampling temperature increases!
			- Think: If the draft model were the same as the original one and using greedy decoding, all output of the draft model would be accepted, maximizing efficiency.
			- Conversely, rejection sampling introduces extra overhead, as the draft model and the original model are sampled independently -- even if their distributions align perfectly, the output of the draft model may still be rejected.
		- We propose that higher temperatures shuld result in *more* opportunities for the original model to accept the draft model's output... and that it's unnecessary to match the distribution of the original model.
		- Thus, we propose a typical acceptance scheme to select plausible candidates, rather than using rejection sampling.
		- Our objective is to choose candidates that are *typical* (not exceedingly improbable to be produced by the original model). We use the prediction probability from the original model as a natural gauge for this, and establish a threshold based on the prediction distribution to determine an acceptance.
		- Given a context x1...xn, and evaluating a candidate sequence x1...xn+k+1 (compose by top predictions of the original language model head and medusa heads), we consider the condition:
		- ![[Pasted image 20240711191435.png|300]]
			- H is the entropy function, epsilon and delta are the hard threshold and the entropy-dependent threshold, respectively.
			- Assumption: Tokens with relatively high probability are meaningful, and when the distribution's entropy is high, various continuations may be deemed reasonable.
		- Every candidate during decoding is evaluated using this criterion, and a *prefix* of the candidate is accepted if it satisfies the condition. To guarantee the generation of at least one token at each step, we apply greedy decoding to the first token and unconditionally accept it while employing typical acceptance for subsequent tokens.
		- The final prediction for the current step is determined by the ==longest accepted prefix== among all candidates.
		- Insights from this scheme:
			- When the temperature is set to zero, it reverts to greedy decoding, since only the most probable token possesses non-zero probability. As the temperature surpasses 0 the outcomes of greedy decoding will be consistently accepted with appropriate epsilon, delta since those tokens have the maximum probability, yielding maximal speedup.
- Self-Distillation
	- We assumed previously that we had some training dataset that matches the target model's output distribution, but this isn't always the case -- for example, the model owners may only release the model without the training data, or the model might already have gone through a RLHF process, making the output distribution from the model different from the training dataset.
	- To tackle this, we propose an automated [[Self-Distillation]] pipeline to use the model itself to generate the training dataset for Medusa heads, which matches the output distribution of the model!
	- Dataset generation process:
		- Take a public seed dataset from a domain similar to the target model (eg using [[ShareGPT]])
		- Take the prompts from the dataset and ask the model to respond to the prompts.
		- To obtain multi-turn conversation samples, we can sequentially feed the prompts from the seed dataset to the model, or, for models like [[Zephyr]] 7B trained on both roles of the conversation, they have the ability to "self-talk," and we can simply feed the first prompt and let the model generate multiple rounds of conversation.
		- For Medusa-1, this dataset is sufficient for training Medusa heads.
		- For Medusa-2, we observe that solely using this dataset for training the backbone *and* Medusa heads usually leads to lower generation quality! Even without training Medusa heads, training the backbone with this model leads to performance degradation -- this suggests that we also need to use the original model's probability prediction, instead of using the ground-truth token as the label for the backbone model -- similar to classic knowledge distillation works.
		- So the loss for the backbone model is: 
			- $\mathcal{L}_{LM-distill} = KL(p_{original,t}||p_t)$ ... where $p_{original,t}$ denotes the probability distribution of the original model's prediction at position t.
		- This requires maintaining two models during training, increasing the model memory requirements.
		- They use a PEFT adapter like [[Low-Rank Adaptation|LoRA]] to fine-tune the back-bone model; so the original model is simply the model with the adapter turned off -- so the self-distillation doesn't require additional memory consumption! Cool trick :) They note that it's preferable to use LoRA without quantization.
- Searching for the optimized tree construction
	- We noted earlier that the simplest way to construct the tree structure is just by taking the Cartesian product, but a regular tree structure might not be the best choice -- we can leverage an estimation of the accuracy to construct the tree structure...
	- Thinking about building a tree by adding nodes one by one...using teh accuracies of top predictions from different heads on a calibration dataset. We greedily add nodes to the tree by choosing the node that's connected to the current tree and has the highest accuracy. Repeat the process until a desired number of nodes is reached.


Abstract
> Large Language Models (LLMs) employ auto-regressive decoding that requires sequential computation, with each step reliant on the previous one's output. This creates a bottleneck as each step necessitates moving the full model parameters from High-Bandwidth Memory (HBM) to the accelerator's cache. While methods such as [[Speculative Decoding]] have been suggested to address this issue, their implementation is impeded by the challenges associated with acquiring and maintaining a separate draft model. In this paper, we present ==Medusa==, an efficient method that augme==nts LLM inference by adding extra decoding heads to predict multiple subsequent tokens in parallel==. Using a ==tree-based attention mechanism==, Medusa ==constructs multiple candidate continuations and verifies them simultaneously in each decoding step==. By leveraging parallel processing, Medusa substantially reduces the number of decoding steps required. We present two levels of fine-tuning procedures for Medusa to meet the needs of different use cases: ==Medusa-1==: Medusa is directly fine-tuned on top of a frozen backbone LLM, enabling lossless inference acceleration. ==Medusa-2==: Medusa is fine-tuned together with the backbone LLM, enabling better prediction accuracy of Medusa heads and higher speedup but needing a special training recipe that preserves the backbone model's capabilities.
> Moreover, we propose several extensions that improve or expand the utility of Medusa, including a [[Self-Distillation]] to handle situations where no training data is available and a typical acceptance scheme to boost the acceptance rate while maintaining generation quality. We evaluate Medusa on models of various sizes and training procedures. Our experiments demonstrate that Medusa-1 can achieve over 2.2x speedup without compromising generation quality, while Medusa-2 further improves the ==speedup to 2.3-3.6x==.


# Paper Figures
![[Pasted image 20240711144612.png|300]]
Medusa includes multiple prediction heads, each of which generates multiple predictions for its designated position. These predictions are assembled into sequence candidates, and a ==tree-based attention mechanism== is used to process them in parallel. It seems we can either use a standard [[Rejection Sampling]] scheme or a *typical acceptance* scheme.

![[Pasted image 20240711170016.png|500]]
It doesn't seem like we have to set the same s_k (?) for every k head -- in this example, they're considering two for the first position, and three for the second position. It sort of makes sense that as uncertainty increases (as we prognosticate further into the future), we consider more tokens/a wider tree. Above picture shows 2 candidates for the first medusa head, and the second 
- "I'm not going to attend to all tokens, I'm only going to attend to tokens that came before on my "path" within the tree.
![[Pasted image 20240711234004.png|300]]
If, in our verification step, the model predicts "it" for the first position... we can discard the left side of the tree here, because the model didn't verify "I".
- We do this speculative decoding to guess tokens the future in parallel (say 5 at a time), and we verify with the model. But instead of verifying a single tree path at a time, we want to verify the whole tree of candidates; the Tree attention mask above lets us do this whole verification simultaneously... 



![[Pasted image 20240711200838.png|600]]
See that Medusas gives a ~2.2-2.8x speedup over standard Vicuna, and that the speedup isn't uniform across MT-Bench categories.

![[Pasted image 20240711201049.png|500]]

![[Pasted image 20240711201128.png|400]]

![[Pasted image 20240711201144.png|300]]

![[Pasted image 20240711201419.png|300]]



# Non-Paper Figures
![[Pasted image 20240711230442.png|500]]
From the OxenAI walkthrough; Essentially we have our 0 token to predict 1.... and 2 and 3 are Medusa speculation
We feed the sequence 0,1,2,3, back to the model, and the LM predicts 1,2,3,4 for the next token for each. So it verifies it, cool. That's the rough idea.
There's a strategy on how you actual accept things... We might not accept the entire proposal:
![[Pasted image 20240711230626.png|500]]
How do we actually accept these tokens, though?
We *could* use Rejection sampling.... which at a simple level does the following:
![[Pasted image 20240711230717.png]]
Given a speculated token (2,3,4), they have an associated probability distribution (we're considering a greedy decoding setting, so only showing the top probability for the token that was actually accepted) from the respective medusa head.  If, upon feeding the sequence (with speculation) back to the main model, the main model has a higher probability, we accept. If it has a lower probability, we reject it with a certain probability, given as a ratio between the model/speculation... so the bigger the gap, the less likely we are to accept a token. This is basically the idea of [[Rejection Sampling]]

![[Pasted image 20240711231445.png]]
You can subtract the probability distributions, and sample from the remaining positive terms (as an alternative). The deepmind/google papers.  

So Rejection Sampling is one way of doing it, but there's another way of doing it that the authors used. This process of being able to generate and then select is critical to Medusa's speedups and performance.

But note there's one complication where every head isn't just picking its top token; instead, each medusa head is offering some top k tokens that are that combined in a cartesian product with the candidates from other medusa heads. Increasing the number of possible generations like this increases the probability of generating some longer accepted sequence... 

----

Preserving a great explainer of tree-attention from a [user on reddit](https://www.reddit.com/r/LocalLLaMA/comments/16g27s0/comment/k077ukn/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button):

> My understanding is that the tree attention by itself enables the parallel validation of each edge of the tree of speculated continuations, which is made up of the combinations of the top-k predictions of the Medusa heads produced in the previous inference step.
> An example of 2 medusa heads with their top-2 predictions:
> Head 1: ["I", "The"]
> Head 2: ["love", "hate"]
> Which produce a tree like this:
> * 　　　　　->I　　　　　　　->love
> 　　　　　　　　　　　　　　->hate
> 　　　　　　->The　　　　　->love
> 　　　　　　　　　　　　　　->hate
> 
> Each node of the tree consist of a token.
> Each edge(arrow) points to a speculated next token continuation from each node.
> Parallel validation works by simultaneously producing the model's next token for each node in the tree.
> For example:
> *(The)　　　->I(hate)　　　->love(the)
> 　　　　　　　　　　　　　　->hate(those)
> 　　　　　　->The(love)　　->love(of)
> 　　　　　　　　　　　　　　->hatred(between)
> 
> The tokens in bold are the speculations that fit the output of the model. Starting from the root and going through the path of valid speculations, we can infer that the next three tokens should be "The", "love" and "of".
> 
> Now for a slightly different example, where the model output "desire" for the token "The" :
> *(The)　　　->I(hate)　　　->love(the)
> 　　　　　　　　　　　　　　->hate(those)
> 　　　　　　->The(desire)　->love(of)
> 　　　　　　　　　　　　　　->hatred(between)
> 
> The path of valid speculations here end at "The", because no speculated continuations from that node (the top-2 predictions of the Medusa head in the next position in the previous inference step) fit the model's continuation for that node (desire).
> So only 2 tokens, "The" and "desire" are accepted in this example.



