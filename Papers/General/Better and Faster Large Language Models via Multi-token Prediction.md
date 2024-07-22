April 30, 2024
[[Meta AI Research]], CERMICS Ecole des Ponts ParisTech, LISN Université Paris-Sacley
[Better and Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737)
#zotero 
Takeaway: A multi-token prediction paper, in which a language model has multiple predictive heads with a shared trunk, and is pretrained to predict the N next tokens, rather than just the next one. During inference the model can operate using just its Next-Token-Prediction (NTP) head , or use the other heads in a form of [[Self-Speculative Decoding]] to speed up decoding ~2.5x-3x on code tasks.
- Authors seem to get really great speedups (6x) and performance *especially* in the case of using a byte-level model, allowing us to fully-compensate the cost of longer byte-level sequences at inference time, still being faster than a vanilla NTP model by ~2x.

This paper came out a few months after the [[Medusa]] paper, and is similar in the sense that they both involve multi-token prediction, but Medusa explores finetuning the ability into language models, while this paper explores how using it in pretraining (and optionally at inference time) can resulting in the learning of better representation in shared trunk.

----
## Introduction
- Next-token-prediction remains an inefficient way of acquiring language, world knowledge, and reasoning capabilities.
- Training LLMs to predict multiple tokens at once (the n future tokens from each position, in parallel) will drive models towards better sample efficiency.
- Authors propose a simple multi-token prediction architecture with no train time or memory overhead, and show that models at 13B solve around 15% more coding problems on average, as a result.
	- It also enabled *==Self-Speculative Decoding==* (see [[Speculative Decoding]]), making models up to 3x faster across a wide range of batch sizes.

## Methods
- Standard language modeling implementing a next-token prediction task, with the learning objective of minimizing cross-entropy loss: $L_1 = - \sum_tlogP_\theta(x_{t+1}|x_{t:1})$    so as to maximize the probability of $x_{t+1}$ as the next future token.
- We generalize this to a multi-token prediction task, where at each position of the training corpus, we predict *n* future tokens at once! This translates into $L_1 = - \sum_tlogP_\theta(x_{t+n:t+1}|x_{t:1})$ , where we're predict not just the t+1'th token, but the t+1 ... t+n range of tokens.
	- We assume our LLM employs a shared trunk to produce a latent representation of the observed context, which is then fed into *n* independent heads to predict in parallel each of the *n* future tokens, leading to the following factorization of the multi-token prediction cross-entropy loss (where $z_{t+1}$ is the latent representation shared among heads):
![[Pasted image 20240722130822.png|400]]
See the second summation sums across the prediction heads.
- A big challenge in training multi-token predictors is reducing their ==GPU memory utilization==
	- Recall that the vocabulary size V is much larger than the dimension d of the latent representation... so the logit vectors (over V) become the GPU memory usage bottleneck.
	- Naive implementations of multi-token predictors that materialize all logits and their gradients (both of shape n,V) sverely limit the allowable batch-size and average GPU memory utilization.
	- In our architecture, we propose to carefully adapt the sequence of forward and backward operations
		- After the forward pass through the shared trunk, we *sequentially* compute the forward *and* backward passes of each independent output head, accumulating gradients at the trunk.
		- This requires the long-term storage of only the d-dimensional trunk gradient, reducing peak GPU memory utilization from O(nV + d) to O(V+d) at no expense to runtime 
			- ((Huh? We have to compute the heads serially... I know they're relatively small parts of the model, but how is there no runtime expense?)).
- Inference
	- During inference, the most basic use of the proposed architecture is just vanilla next-token autoregressive prediction using the next-token prediction head, while discarding all others.
		- (Here, the next-token prediction head benefits from the trunk having been updated during training with this "forward looking" objective)
	- However, additional output heads can be used to speed up decoding from the NTP head with ==[[Self-Speculative Decoding]]== methods such as ==blockwise parallel decoding==, a variant of speculative decoding without the need for an additional draft model, or with [[Speculative Decoding]] with [[Medusa]]-like tree attention.


## Experiments on Real Data
- Benefits scale with model size
	- Authors train 6 models in the 300M-13B range, and show meaningful improvements, especially and increasingly for the larger sized models (>3B). The small models (<3B) actually have adverse performance impacts! ==This *usefulness only at scale* is a reason why multi-token prediction has so far been largely overlooked==.
- Faster inference
	- Authors implement greedy [[Self-Speculative Decoding]] with heterogenous batch sizes using ==xFormers== (2022) and measure decoding speeds on a 7B 4-token prediction model, and observe a speedup of 3x on code, with an average of 2.5 accepted tokens out of 3 suggestions on code, and a 2.7x speedup on text.
		- On an 8-byte prediction model (where the vocabulary is bytes), inference speedup is 6.4x? See Table 1.
- Learning global patterns with multi-byte prediction
	- Authors go to the extreme case of byte-level tokenization by training a 7B parameter byte-level transformer on 314B bytes, which is equivalent to around 116B tokens.
		- The 8-byte prediction model achieves astounding improvements compared to next-byte prediction, solving 67% more problems on MBPP pass@1, and 20% more on HumanEval Pass@1.
	- ==So multi-byte prediction is a very promising avenue to unlock efficient training of byte-level models!==
		- ==Self-speculative decoding achieves speedups of 6x for the 8-byte prediction model, allowing us to fully-compensate the cost of longer byte-level sequences at inference time, and even be faster than a NTP model by ~2x!==
- Searching for the optimal $n$ 
	- Authors do a number of ablations on 7B param/200B token models, and try n=1,2,4,6,8... results in Table 1 show that ==training with 4-future tokens outperforms other models consistently.== (Very likely this depends on data distribution though)
- Training for multiple epochs
	- Even when training on multiple epochs of the same data, Multi-token training maintains an edge of NTP (though the improvements diminish slightly).
- Finetuning multi-token predictors
	- Performance improvements persist through finetuning.
- Multi-token prediction on natural language
	- Authors train 7B param/200B token models on natural language with 1,2,4-token prediction objectives, and benchmark resulting checkpoints on 6 standard NLP benchmarks...
	- Authors find that the 2-future-token prediction model performs on-par with NTP, and the ==4-future-token prediction model suffers a *performance degradation!*==
	- Authors contest that these multiple-choice and and likelihood-based benchmarks are really good ways to discern the *generative capabilities* of language models... So they conduct evaluations on 8 summarization benchmarks and natural language math benchmarks (GSM8k) too....
		- For 200B training tokens, the n = 2 model clearly outperforms the next-token prediction baseline, while the pattern reverses after 500B tokens and ==n = 4 is worse throughout==.


## Ablations on Synthetic Data
- Authors show that multi-token prediction leads to qualitative changes in model capabilities and  generalization capabilities.
	- For smaller models, induction capability either only forms when using multi-token prediction as training loss, or it is vastly improved by it.
	- ...

## Why does it work? Speculation
- Author's intuition is that multi-token prediction ==mitigates the distributional discrepancy between training time teacher forcing and inference-time autoregressive generation.==
- Multi-token prediction implicitly assigns weights to training tokens depending on how closely they're correlated with their successors... assigning higher weights to *consequential tokens*.
	- Authors think some transitions (next tokens) are considered inconsequential, and some are consequential choice points -- inconsequential transitions following a choice point are thus hard to predict in advance.
	- Authors believe that the quality of text generation depends on picking the right decisions at choice points, and that n-token prediction losses promote those.

## Related Work
Language modeling losses
- ...
Multi-token prediction in language modeling
- Multi-token prediction encourages planning, improves representation, and prevents the overfitting on local patterns that can result from teacher-forced training...
- Medusa proposes model finetunings with multi-token prediction for faster inference, but don't study the effects of such a loss during pretraining.
Self-speculative decoding
- [[Medusa]] present a more elaborate self-speculative decoding scheme that uses the top-k predictions of each head instead of the best one only. It can be used with the multi-token prediction models we train.
Multi-target prediction
- ==Multi-task learning== is the paradigm of training neural networks jointly on several tasks to improve performance on the tasks of interest.
- Learning with auxiliary tasks allows models to exploit dependencies between target variables, and can even ben preferable in the case of independent targets.
- Most modern deep learning approaches rely on large shared model trunks with separate prediction heads for respective tasks.


## Conclusion


Abstract
> Large language models such as GPT and Llama are trained with a next-token prediction loss. In this work, we suggest that training language models to predict multiple future tokens at once results in higher sample efficiency. More specifically, ==at each position in the training corpus, we ask the model to predict the following n tokens using n independent output heads, operating on top of a shared model trunk==. Considering multi-token prediction as an auxiliary training task, we measure improved downstream capabilities with no overhead in training time for both code and natural language models. The method is increasingly useful for larger model sizes, and keeps its appeal when training for multiple epochs. Gains are especially pronounced on generative benchmarks like coding, where our models consistently outperform strong baselines by several percentage points. Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models. Experiments on small algorithmic tasks demonstrate that multi-token prediction is favorable for the development of induction heads and algorithmic reasoning capabilities. As an additional benefit, models trained with 4-token prediction are up to 3 times faster at inference, even with large batch sizes.


# Paper Figures

![[Pasted image 20240721234532.png|300]]
Paper uses multiple prediction heads, where each n head predicts the t+1+n token during training. At inference time, only the NTP head can be used (benefitting from updated trunk representations), or the other heads can be used to speed up decoding.

![[Pasted image 20240722132617.png|300]]
The authors note that one of the challenges of multi-token prediction is around the calculation of logits, which are large and Vocabulary-dimensional, instead of d-dimensional like the hidden activations. These activations (and gradients) take up a lot of memory, and that's multiplied when you have multiple decoding heads! So the authors do the forward pass of the shared trunk, then do (in serial) the forward and backward pass for each head, accumulating the gradients at the end of the d-dimensional trunk, avoiding overuse of memory.

![[Pasted image 20240722133414.png|250]]
Training with this multi-token prediction objective results in a model (that, at inference, only predicts the next token) that performs better at coding tasks on [[MBPP]] and [[HumanEval]].
From abstract: "Our 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models"
It's ==very interesting to note that the small models (< 3B) seem to actually be *hurt* by this multi-token-prediction objective!==


![[Pasted image 20240721234647.png|500]]
2 different "types" of models (token vocabs, byte vocabs), with different toke prediction objectives ... the authors were really excited about how much the 8-byte (n=8 heads) version of the byte-level model improved performance.

![[Pasted image 20240722143146.png|200]]
Checking whether NTP or 4-token prediction model finetunes better on CodeContests dataset; both ways of finetuning the 4 token prediction model outperform the NTP prediction baseline... ==the best way weirdly seems to be using a NTP finetuning on top of the 4 token prediction model.==

![[Pasted image 20240722143156.png|200]]
(Using 7B model): Evolution of average accuracy of 6 standard NLP benchmarks... the 2 token prediction model has the same performance as the NTP, but ==the 4 token model *regresses performance*.== 
- Authors question whether we would still see this using a larger model 🤔

![[Pasted image 20240722143445.png|250]]
- See that the 2 and 4 token prediction models seem to do a little better on summarization than the NTP model.

![[Pasted image 20240722143515.png|300]]
For small model sizes, induction capability either only form when using multi-token prediction as a training loss, or it is vastly improved by it.

![[Pasted image 20240722143916.png|200]]

![[Pasted image 20240722144256.png|300]]
An explanation of how this method implicitly assigns weights to training tokens depending on how correalted they are with their successors.
- One transition is hard to predict choice point, while the other transitions are "inconsequential"; these inconsequential transitions following a choice point are hard to predict in advance!
- ==We believe that the quality of text generations depends on picking the *right decisions* at choice points, and that *n-token* prediction losses promote those.==
















