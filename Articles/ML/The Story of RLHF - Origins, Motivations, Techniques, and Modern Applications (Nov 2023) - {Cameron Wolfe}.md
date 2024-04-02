#article 
Link: https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations

----

For a long time, the AI community has leveraged different styles of language models to automate both ==generative== and ==discriminative== natural language tasks!
- There was a surge of interest in 2018 with [[Bidirectional Encoder Representations from Transformers|BERT]], which demonstrated the power of a combination of:
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

Namely, we train this model to maximize the probability of *human written generations,* but what we want is a model that produces high-quality outputs, by whatever definition. These tow objectives aren't always aligned -- does a better finetuning approach exist?

## Language Model Alignment

![[Pasted image 20240401234357.png]]
- If we examine the outputs of a generative LLM immediately after pretraining, we see that, ==despite the model obviously possessing a lot of knowledge, the model generate repetitive and uninteresting results.==
- Even if this model can accurately predict the next token, this doesn't imply the ability to generate coherent and interesting text!
	- ==This is a result of a misalignment between the objective used for pretraining -- next token prediction -- and what we actually want -- a model that generates high-quality outputs.==

What's ==alignment==?
- Refers to the idea of teaching an LLM to produce output that aligns with human desires

We usually start by defining a set of "criteria" that we will want to instill within the LLM. For example, common alignment criteria might include:
- *==Helpfulness==: The model fulfills user requests, follows detailed instructions, and provides information requested by the user.
- *==Safety==*: The model avoids responses that are "unsafe"
- *==Factuality==*: That the model doesn't generate factually incorrect information

To align an LLM, we can finetune the model using SFT and RLHF in sequence, as previously described.

### The Role of RLHF
- Unlike pretraining and SFT, ==RLHF is not a supervised learning technique -- it leverages [[Reinforcement Learning]] to directly finetune an LLM based on feedback that is provided from humans==.
	- Put simply: *Humans just identify which outputs from an LLM that they prefer, and RLHF will finetune the model based on this feedback.*
	- This approach is fundamentally different from supervised learning techniques due to the fact that we can directly train the LLM to produce high-quality outputs.

> Just identify outputs that we like, and the LLM will learn to produce more outputs like this!


# Evaluating Language Models (and the [[ROGUE]] score)
- Before learning more about RLHF, e need to understand how LLMs are evaluated, including the recall-oriented understudy for gisting evaluation (ROGUE) score.

We should note that there are many ways that we could evaluate the output of generative language model.

For example, we should prompt a powerful language model like GPT-4 to evaluate the quality of a model's output, or even leverage the reward model from RLHF to predict a preference/quality score.

### Traditional Metrics
- Prior to the LLM revolution, several popular evaluation metrics existed for language models that operated by comparing the model's generated output to a reference output -- usually manually written by humans.
- [[ROGUE]] score is one of these classical metrics, and ==it works by counting the number of words== -- or the number of *n-grans* for ROGUE-N -- ==in the reference output that also occur in the model's generated output==.

![[Pasted image 20240401235520.png]]

Going further -- there are other metrics that work by comparing models' outputs to known reference outputs:
1. ==Bilingual Evaluation Understudy ([[BLEU]]) score==
	- Commonly used to evaluate translation tasks by counting the number of matching n-grams between the generated output and the reference, then dividing this number by the total number of n-grams within the generated output.
2. BERTScore
	- Generates an embedding for each n-gram in the generated output and reference output, and then uses cosine similarity to compare n-grams from the two textual sequences.
		- ((This seems like a pretty good tool -- instead of comparing that your model EXACTLY matches some reference output, try to see if it semantically (as per embedding similarity) matches the reference))
3. MoverScore
	- Generalizes BERTScore from requiring a one-to-one matching between n-grams to allowing many-to-one matches, thus making the evaluation framework more flexible.

Do we need something else?
- ==These metrics work reasonably well for tasks like summarization or translation, but they don't work for more open-ended generative tasks like information-seeking dialogue.==
- It tends that there is a ==poor correlation between these metrics and human judgement of a model's output.==

ROGUE and BLEU quantify the extent to which a model's output matches some reference output.
But for many problems, there are *numerous* outputs that a model could produce that are all equally viable -- fixed metrics like the above can't account for this.

![[Pasted image 20240402000522.png|450]]

Again, we have a misalignment between what is being measured (the overlap between two textual sequences) and what we *really* want to measure -- *output quality.*
- ((I'll not that we haven't defined output quality concretely yet, and I'm not sure that a human rater would be a consistent rater of this quality. It can be difficult to compare the quality between two good outputs, past a certain level of performance.))

As a result, the research community was forced to attempt to find more flexible metrics to address the poor correlation between these traditional metrics and the true quality of a model's output.


# Learning from Human Feedback
- Next token prediction is a training objective used almost universally for LLMs (i.e. during pretraining and supervised finetuning).
- By learning to maximize the log-probabilities of human-written sequences of text within a dataset, this technique provides us with the ability to train language models in a self-supervised fashion over large corpora of text data.
- ==RLHF provides us with the ability to directly optimize an LLM based on human feedback, avoiding the misalignment that comes from the normal LLM training objective and the true goal of learning -- generating high-quality output as assessed by human reviewers.==

![[Pasted image 20240402002022.png]]

Within this section, we dive deeper into the ideas behind RLHF to better understand:
1. The role of RLHF in the LLM training process
2. The benefits of RLHF compared to supervised learning


## How does RLHF work?
- RLHF was first applied to LLMs in
	- "Learning to Summarize with Human Feedback" (2020, Stiennon, Nissan)
	- "Training language models to follow instructions with human feedback" (2022, Ouyang, Long)
- These extend
	- "Fine-Tuning Language Models from Human Preferences" (2019)
		- Learns a reward model from human comparisons of model outputs, and uses this reward model to finetune language models on a variety of different natural language tasks.

However, work doesn't copy these techniques directly.
The modern approach to RLHF has been adapted to improve the efficiency of both human annotation and LLM finetuning.


![[Pasted image 20240402004050.png]]
### Understanding RLHF
- We will now look at the implementation of RLHF used in the two papers above.
- Typically, RLHF is done in tandem with SFT -- the model that is obtained after both pretraining and SFT serves as the starting point for RLHF.

==The RLHF framework is comprised of three standard steps:==
1. ==Collect human comparisons==
	- Human feedback is collected offline in large batches prior to each round of RLHF.
	- A dataset of human feedback is comprised of prompts, several LLM-generated responses to each prompt, and a *ranking* of these responses based on human preference.
2. ==Train a reward model==
	- A reward model is trained over the dataset of human comparisons to accurately predict a human preference score when given an output generated by an LLM as input.
3. ==Optimize a policy according to the reward model==
	- The policy is finetuned using reinforcement learning ([[Proximal Policy Optimization|PPO]] in particular) to maximize reward based on human preference scores generated by the reward model.


Data collection
- The approach above is *highly dependent on the quality of feedback provided by human annotators!*
	- It's important to keep close correspondence between your annotators and AI researchers, with extensive onboarding, shared communication, and a close monitoring of agreement rates between annotators and researchers.

==The rate of agreement between researchers and human annotators should be maximized to ensure that human preference data accurately reflects the desired alignment criteria==

![[Pasted image 20240402011359.png]]

### Training the reward model
- The reward model shares the underlying architecture of the LLM, but the *classification head* that is usually used for next-token prediction is removed and replaced with a *regression head* that predicts a preference score (above picture, right side).

The reward model is typically initialized with the same weights as the LLM (either the pretrained model or the model trained by SFT), thus ensuring that the reward model shares the same knowledge base as the underlying LLM.

![[Pasted image 20240402011621.png]]
Above: 
- 
