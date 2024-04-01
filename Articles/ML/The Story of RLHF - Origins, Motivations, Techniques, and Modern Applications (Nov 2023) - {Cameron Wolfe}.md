#article 
Link: https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations

----

For a long time, the AI community has leveraged different styles of language models to automate both ==generative== and ==discriminative== natural language tasks!
- There was a surge of interesti n 2018 with [[Bidirectional Encoder Representations from Transformers|BERT]], which demonstrated the power of a combination of:
	- The [[Transformer]] architecture
	- Self-supervised Pre-training
	- Transfer learning

BERT set a new SoTA benchmark in every benchmark to which it was applied!
With the proposal of [[T5]], we saw that supervised transfer learning was effective for *generative* tasks as well (BERT wasn't a generative model).

Despite these accomplishments, models pale in comparison to the generative capabilities of LLMs like GPT-4 that we have today. To create a model like this, we need training techniques that go far beyond supervised learning.

Modern generative language models are a result of numerous notable advancements in AI research, including:
- The decoder-only transformer
- Next-token prediction
- Prompting
- Neural scaling laws
- ==Alignment menthods==: Aligning our models to the desires of human users.

==Alignment was primarily made possible by directly training LLMs based on human feedback via reinforcement learning from human feedback ([[Reinforcement Learning from Human Feedback|RLHF]] )==
- Using this approach, we can teach LLMs to surpass human writing capabilities, follow complex instructions, avoid harmful outputs, cite their sources, and much more!
- RLHF enables the creation of AI systems that are more safe, capable, and useful!
	- (([[Anthropic]] uses the 3 H's of ==Helpful, Harmless, and Honest==))

Let's develop a deep understanding of RLHF, its origins/motivations, the role it plays in creating powerful LLMs, the key factors that make it so impactful, and how recent research aims to make LLM alignment even more effective.

# Where did RLHF come frmo?
- Prior to learning about RLHF and the role that it plays in creating powerful language models, we need to understand some basic ideas that preceded and motivated the development of RLHF, like:
	- Supervised learning (and how RLHF is different)
	- The LLM alignment process
	- Evaluation metrics for LLMs (and the [[ROGUE]] score in particular

### Supervised Learning for LLMs
- Most generative LLMs are trained via a pipeline that includes:
	1. (Self-supervised) Pretraining
	2. Supervised Finetuning ([[Supervised Fine-Tuning|SFT]])
	3. [[Reinforcement Learning from Human Feedback]] 
	- And *maybe* some additional finetuning depending on our application; see above.

Before RLHF:
- RLHF is actually a recent addition to the training process -- previously, language models were trained to accomplish natural language tasks (e.g. summarization or classification) via a simplified transfer learning procedure that includes just:
	- Pretraining
	- Finetuning
![[Pasted image 20240330182111.png]]
Above:
- Bottom-left: BERT; Bottom-right: T5 multitask text-to-text using those "trigger words" (mine).

Notable examples of models that follow this approach include BERT and T5, which are first pretrained using a self-supervised learning objective over a large corpus of unlabeled textual data, then finetuned on a a downstream dataset in a supervised learning fashion from labeled examples. 

#### Is supervised learning enough?
- Supervised finetuning works well for closed-ended/discriminative tasks like classification, regression, question answering, retrieval, and more
- Generative tasks, however, and slightly less compatible with supervised learning
	- Training language models to generate text in a supervised manner requires us to manually write examples of desirable outputs, and then train the underlying model to maximize the probability of generating these provided examples.
	- Even before the popularization of generative LLMs, such an approach was heavily utilized for tasks like summarization, where we could teach a pretrained LM to write high-quality summaries by simply training on example summaries written by humans.

^ The above strategy of supervised learning on human examples of 'generated' texts *does* work, but ==there's a misalignment between this fine-tuning objective (maximizing the likelihood of human-written text examples) and what we care about -- *generating high-quality outputs as determined by humans*!==

Namely, we train this model to maximize the probability of human







