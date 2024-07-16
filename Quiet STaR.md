March 14, 2024
Stanford, Notbad AI Inc (lead [[Eric Zelikman]])
[Quiet-STaR: Language Models can Teach Themselves to Think Before Speaking]()
#zotero 
Takeaway: ...

Similar earlier papers include: 
- [[Self-Taught Reasoner|STaR]] (same author, March 28, 2022)
- [[Self-Taught Optimizer]] (STOP, same author, October 3, 2023)
- [[Reinforced Self-Training]] (ReSTEM, Dec 11, 2023)

> "Generalization of STaR that is the first work explicitly training LMs to reason generally from *text*, rather than on curated reasoning tasks or collections of reasoning tasks." - Authors

Questions: 
- It doesn't seem like the thoughts generated between output tokens A and B are useful for the generation of C, or even of the thoughts between B and C.
- I was thinking that at the end of the training process, our mixing head should basically be weighting the representation at the end of the intermediate thought... at 100%? Or no? Not clear to me 🤔. I think we only attend to the end-of-token logit (and then the previous real token outputs).

----

## Introduction
- Reasoning about the implications of text to predict later text (eg [[Chain of Thought|CoT]]) has consistently been shown to improve LM performance on a variety of tasks, but methods for allowing LMs to learn from their reasoning (eg [[Self-Taught Reasoner|STaR]]) have focused on solving *individual tasks* or a predefined set of tasks. 
- We instead ask: *If reasoning is implicit in all text, why shouldn't we leverage the task of language modeling to teach reasoning?*
- [[Self-Taught Reasoner|STaR]] showed that LMs can bootstrap their reasoning ability on question-answering datasets by sampling rationales to attempt to answer questions, and trining on rationales that led to a correct final answer, and then repeating this iteratively to solve more difficult problems.
	- Training from curated/labeled QA datasets limits the scale and generalizability of the rationales -- high-quality QA datasets require thoughtful curation and will only ever cover a subset of reasoning tasks!
	- As a result, we extend STaR to [[Quiet STaR]], which helps an LM infer future text from a large unlabeled internet text corpus! We leverage the LM's pre-existing reasoning ability to generate rationales, and train the LM on them with a [[REINFORCE]]-based reward!
- Quiet-STaR proceeds by generating rationales after every token to explain future text (*think*), mixing future-text predictions with and without rationales (*talk*), and then learning to generate better rationales using [[REINFORCE]] (*learn*).
	- We apply it to [[Mistral 7B]] using the webtext datasets OpenWebMath and [[C4]], and find that Quiet-STaR results in improvements to zeros-hot direct-reasoning abilities on [[CommonsenseQA]] (36.3 -> 47.2%) and [[GSM8K]] (5.9% -> 10.9%), and that these improvements predictably increase with the number of tokens used in LM's internal thoughts!
- Contributions:
	- Generalization of STaR that is the ==first work explicitly training LMs to reason generally from *text*==, rather than on curated reasoning tasks or collections of reasoning tasks.
	- They use a ==parallel sampling algorithm== that makes the training procedure scalable, generating rationales from *all token positions* in a given string.
	- Introduce ==custom meta-tokens at the start and end of each thought== to let the LM learn that it should be generating a rationale, and when it should make a prediction based on that rationale.
	- We apply a ==mixing head== to retrospectively determine how much to incorporate the next-token prediction from a given thought into the current next-token prediction.
	- We show that a ==non-myoptic loss== (including multiple tokens ahead for language modeling) improves the effect of thinking.
	- We demonstrate that the thinking allows the LM to predict difficult tokens better than one trained on the same text.

## Related Work
- CoT solutions and scratchpads have been shown to be useful.
- Wang and Zhou (2024) showed that for commonsense QA, one could force a LM to leverage CoT reasoning by preventing it from emitting any valid answer tokens unless it was confident.
	- (But it uses heuristics to identify this)
- We use relative improvement in the log-likelihood of the target text across rationales as an estimate of quality, but we simply subtract the mean reward and don't incorporate more complex control variates.
- Some authors have trained LMs on mined reasoning traces or reasoning-like data, which is effective but come with the usual drawbacks of manual annotations (sensitivity to and capped by annotator capability, expensive, is necessarily [[Off-Policy]] for the language model, harder to scale).
- Another direction relies on a LM's own generated reasoning, which can be seen as building on a body of [[Self-Play]] literature, including methods like [[Self-Taught Reasoner|STaR]], *Uesato et al 2022* (a [[Process Reward Model|PRM]] paper, iirc) (demonstrated additional usefulness to "process-based" supervision where incorrect reasoning traces were filtered), and [[V-STaR]], which demonstrates that training a verifier to guide generation also improves performance, as well as [[TRICE]], which maximizes the marginal likelihood of the correct answer given several reasoning traces per problem.
- There's a growing body of work that demonstrates the usefulness of custom tokens optimized to perform specific functions in the context of neural networks -- they've been referred to as "==function vectors==".
	- Includes [[Prompt Tuning]], [[Prefix Tuning]], and others have appleid meta-tokens to compress long prompts for efficiency, and "pause tokens" (essentially representing each token as two tokens) improves LM performance.

## Problem Statement
- We introduce an auxiliary "rationale" variable between each pair of observed tokens in the sequence.
- We aim to optimize a LM with parameters $\theta$ with the ability to generate intermediate thoughts (or rationales) such that:
	- $\theta^* = \underset{\theta}{argmax}E_x[logp_{\theta}(x_{i:n}|x_{0:i}, rationale_{\theta}(x_{0:i})]$ 
- Note that we formulate the objective as accurately predicting *the entire remaining sequence* rather than only the next token. We find this *non-myopic* formulation leads to a more effective loss for learning rationales.

## Quiet-STaR
- Three main steps:
	1. ==Parallel rationale generation==: In parallel across n tokens x_i in an input sequence x_{0:n}, we generate r rationales of length t for each x_i, resulting in ==n x r rationale candidates==. We inset learned `<|startofthought|>` and `<|endofthought|>` tokens to mark each rationale's start and end.
	2. ==*Mixing* post-rationale and base predictions==: From the hidden state output after each rationale, we train a "==mixing head==" shallow MLP producing a weight ==determining how much the post-rationale next-token predicted logits should be incorporated compared to the *baes* LM predicted logits.==
		- This eases distribution shift early in finetuning, due to introducing rationales.
	3. ==*Optimizing* rationale generation==: We ==optimize the rationale generation parameters== (start/end tokens and LM weights) to ==increase the likelihood of rationales that make future text more probable==. We use [[REINFORCE]] to provide a learning signal to rationales based on their impact on future-token prediction. To reduce variance, we apply a [[Teacher Forcing]] trick to include in the loss the likelihood of predicting not only the token *after* the logits, but also *later tokens.*
- Parallel Generation
	- A key challenge in Quiet-STaR is efficiently-generating rationales at each token position in the input sequence. Naively, this would require a separate forward pass foreach token, which becomes computationally intractable for longer sequences.
	- We allow for highly-parallel generation by first observing that an inference pass of a LM produces a probability distribution over the next tokens for *all input tokens.*
		- If one has generated a successor from each token (eg given `<bos> the cat sat`, having `yes orange saw down`)... each successor is itself a reasonable token to a prefix of the sequence... We can leverage these to generate hidden thoughts for each observed token.
		- ==We cache each forward pass and concatenate a diagonal attention matrix to the previous attention mask.== Each generated token now attends to all of the tokens that were used to generate it, as well as to itself (but not to tokens on other "counterfactual" paths).
	- ==We construct an attention mask that lets all thought tokens pay attention to themselves, preceding thought tokens in the same thought, and the preceding text== to generate continuations of all thoughts in parallel. ((See figure; this is sort of hard to grok, reminds me of the "tree attention" in [[Medusa]]))
- "Mixing" (Residual) Heads
	- Starting with a pre-trained model, thoughts will initially be out of distribution (?... Because producing rationales isn't something that the pre-trained model was trained to do). To smooth the transition to thinking, we introduce a learned interpolation between LM predictions *with* and *without* thoughts.
	- Given end-of-thought token's hidden state and the hidden state of the original text token, the mixing head (shallow MLP of 3 layers with ReLU) outputs a weight that determines the extent to which the post-thought prediction logits will be used.
- Optimizing Rationale Generation
	- Optimizing Start-of-Thought and End-of-Thought tokens
		- These tokens serve as ***learned meta-tokens*** that control the model's rationale generation. Optimization the representation of these (especially the start of thought token) is crucial but challenging due to the discrete nature of the rationale tokens.
		- We initialize the start/end token embeddings to the embeddings corresponding to the "em dash" (--), which often appears in text data to denote a pause or thought.
		- We apply a weight (hyperparam) to the gradient of these embeddings during the update step to let them be optimized more quickly.
	- Non-myopic Scoring and Teacher Forcing
		- We don't expect thoughts to be useful in predicting *every token*, so we'd prefer the model's reward to depend less on the *exact* next word in the text following the thought, and more on the following semantic content.
		- We use the parallel attention mask to compute the log probabilities of the true next tokens, applying [[Teacher Forcing]] by assuming the model selected the correct next ground-truth token...
			- The number of future tokens included in the loss is a hyperparameter
- Objective
	- We use [[REINFORCE]] to optimize the likelihoods of the rationales based on their usefulness!
		- The log-likelihood of the $n_true$ true next tokens X_{j+1:j+n_true+1} under the language model given previous observed tokens and a particular rationale (p_j:j+n_true^talk).
		- To reduce variance, we generate *multiple* rationale continuations for each token in the input sequence (loosely inspired by [[TRICE]]). So we define the reward for each rational T as the difference between $p^{talk}_{j:j+n_{true}}$ and the average across rationales for that token.
![[Pasted image 20240716153743.png]]
This loss term encourages the model to generate rationales that improve its predictions of future tokens compared to the average prediction across all generated rationales for that token. The gradients from this loss are used to update both the LM parameters and the start of thought and end of thought toke n embeddings, with a hyperparameter weight applied to the gradients of the SoT and EoT token embeddings to accelerate their optimization.
We also include a log likelihood loss to ensure that the LM learns to optimize the talking heads and also receives a next-token prediction signal for the base LM head.

## Experiment and Result
- Intuitively, not all tokens require equal amounts of thought. Consider "the person is run-" ... it's likely that the next token is "ing". Additional thinking is unlikely to improve a well-trained model's prediction.
- We conjecture that for most chunks of most online text, additional thought has little to no impact.
- So we design our experiment to investigate whether our approach is useful in predicting that DO require thought!
- We find thaat on average there is little improvement in the LM's ability to predict arbitrary tokens.
- There are parallels between [[Chain of Thought]] and Quiet-STaR, but they'er actually orthogonal and complementary techniques. 
	- CoT lets a model think out loud using its ordinary production distribution, whereas Quiet-STaR instead allows a model to think quietly at every token.
	- Accuracy (unsure on whaat) over 8 samples increases from 40.6% to 47.7% using Chain of Thought with Quiet-STaR.
- There's no explicit regularization in Quiet-STaR for thoughts to be human-interpretable... though they're generated from the same transformer used to model langauge, so they're likely to be at least partially understandable.


## Limitations and Conclusion
- Authors only applied Quiet-STaR to a 7B model, albeit  powerful one. 
	- It's often observed that similar techniques yield even better results when applied to stronger models.
- Quiet-STaR results in substantial overhead, generating many tokens before generating every additional token.
	- It's a way of leveraging additional compute to enhance next token predict.


## Appendix



Abstract
> When writing and talking, people sometimes pause to think. Although reasoning-focused works have often framed reasoning as a method of answering questions or completing agentic tasks, reasoning is implicit in almost all written text. For example, this applies to the steps not stated between the lines of a proof or to the theory of mind underlying a conversation. In the [[Self-Taught Reasoner]] (STaR, Zelikman et al. 2022), useful thinking is learned by inferring rationales from few-shot examples in question-answering and learning from those that lead to a correct answer. This is a highly constrained setting -- ideally, a language model could instead learn to infer unstated rationales in arbitrary text. We presen==t [[Quiet STaR]], a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions==. We address key challenges, including 1) the computational cost of generating continuations, 2) the fact that the LM does not initially know how to generate or use internal thoughts, and 3) the need to predict beyond individual next tokens. To resolve these, ==we propose a tokenwise parallel sampling algorithm, using learnable tokens indicating a thought's start and end, and an extended [[Teacher Forcing]] technique.== Encouragingly, generated rationales disproportionately help model difficult-to-predict tokens and improve the LM's ability to directly answer difficult questions. In particular, after continued pretraining of an LM on a corpus of internet text with Quiet-STaR, we find zero-shot improvements on [[GSM8K]] (5.9%→10.9%) and [[CommonsenseQA]] (36.3%→47.2%) and observe a perplexity improvement of difficult tokens in natural text. Crucially, these improvements require no fine-tuning on these tasks. Quiet-STaR marks a step towards LMs that can learn to reason in a more general and scalable way.


# Paper Figures

![[Pasted image 20240716122115.png|600]]

![[Pasted image 20240716123827.png]]
It's interesting how much having additional thoughts increases the performance of the model on GSM8K, and that it seems like this mostly saturates on CommonsenseQA at like 24. I don't understand why the CommonsenseQA baseline's performance decreases after training?...

![[Pasted image 20240716140022.png|600]]
Reminds me of the "tree attention" from the [[Medusa]] paper

![[Pasted image 20240716145308.png|600]]
- Lol at the use of the kitchen mixer emoji for the mixing head. This is visualizing predicting three tokens ahead...
- Dashed lines indicate teacher forcing. This just seems like usual teacher forcing, right? Though what does teacher forcing look like for thoughts, if that's what I'm seeing? Since we aren't using a labeled dataset.

![[Pasted image 20240716154322.png|600]]
See improvement that Quiet Star seems to make over CoT...

![[Pasted image 20240716155249.png|600]]
An example where we recall that one should start with magnesium to produce magnesium nitride, which allows it to better predict that the first step of the procedure involves heating magnesium. ((Makes me think: I bet the thoughts for the previous word were pretty similar... but I don't think we get to take any advantage of them.))

![[Pasted image 20240716155414.png|600]]
An example in which the most useful thoughts seem to be near-continuations that correspond more closely to the target text.

![[Pasted image 20240716155656.png|600]]
An example of a thought that occurs during the middle of reading a question (hence not used to predict the final answer). ((Interesting that we seem to limit the length of thoughts. I wonder how much that hyperparameter matters?))
