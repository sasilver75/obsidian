#article 
Link: https://cameronrwolfe.substack.com/p/the-story-of-rlhf-origins-motivations

----

For a long time, the AI community has leveraged different styles of language models to automate both ==generative== and ==discriminative== natural language tasks!
- There was a surge of interest in 2018 with [[BERT|BERT]], which demonstrated the power of a combination of:
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
	- Evaluation metrics for LLMs (and the [[ROUGE]] score in particular

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


# Evaluating Language Models (and the [[ROUGE]] score)
- Before learning more about RLHF, e need to understand how LLMs are evaluated, including the recall-oriented understudy for gisting evaluation (ROUGE) score.

We should note that there are many ways that we could evaluate the output of generative language model.

For example, we should prompt a powerful language model like GPT-4 to evaluate the quality of a model's output, or even leverage the reward model from RLHF to predict a preference/quality score.

### Traditional Metrics
- Prior to the LLM revolution, several popular evaluation metrics existed for language models that operated by comparing the model's generated output to a reference output -- usually manually written by humans.
- [[ROUGE]] score is one of these classical metrics, and ==it works by counting the number of words== -- or the number of *n-grans* for ROUGE-N -- ==in the reference output that also occur in the model's generated output==.

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

ROUGE and BLEU quantify the extent to which a model's output matches some reference output.
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
- ((Why is this? Is this obvious?))

![[Pasted image 20240402011621.png]]
Above: 
- To train the reward model we:
	- Take pairs of ranked responses as input
	- Predict the preference score of each response
	- Apply a ranking loss

The purpose of this ranking loss is to train the reward model (basically just a classifier) to output a higher preference score for the preferred output and vice versa.
- ==We're attempting to train a model that can accurately predict a human preference score, given a prompt and response pair as input!==


### Finetuning via RL
- To finetune the language model, we can formulate generating text via the LM as a [[Reinforcement Learning]] problem!
	- Our ==policy== is the LLM and its corresponding parameters.
		- ((Usually, the policy predicts a distributions over actions, given a state))
	- Each token generated by the LLM corresponds to a *==single time step== in the environment*
	- An entire ==episode== is completed when the LLM outputs the \[EOS\] token
	- The ==state== is given by the sequence being outputted by the LLM
	- There is ==no explicit transition function==, as we simply add each output token to the generated sequence.
	- At the end of each episode, we receive a single ==reward== -- ***generated by the reward model***, based on the perceived quality of the generated sequence.

![[Pasted image 20240407194631.png]]
Above:
- We generate the sequence over "time steps," sampling words one-by-one.
- When the \[EOS\] token is generated, we can then consider our final output sequence and calculate a *reward* for it, using our reward model.
- Given the reward, we can use [[Proximal Policy Optimization]] to update our policy (the LLM and its parameters).
	- PPO was adopted due to its simplicity, robustness, and efficiency -- but there are many other algorithms that have been explored and adopted in recent months.


![[Pasted image 20240407195004.png]]
Above:
- You can see (left to right, within a chart) the impact of multiple iterations of RL-driven fine-tuning on both *Harmlessness* and *Helpfulness*. Regardless of which Judge is used (Meta, GPT-4), it seems that the improvement in both dimensions continues.
- Note from this diagram that RLHF is typically not applied only once -- rather, most works tend to collect data in large batches and finetune the model via RLHF in an online fashion, in multiple rounds.
	- LLaMA-2 performed 5 successive rounds of RLHF
	- Anthropic (at some time) finetuned its HH LLM via RLHF on a weekly cadence as new batches of preference data were collected.


### Learning to Summarize from Human Feedback (Paper 1)
- The problem of abstractive summarization, or using a language model to understand important aspects of a text and produce a human-readable summary, has been studied for a long time.
- Prior to the popularization of RLHF, most approaches to this task trained the language model in a *supervised manner* based on human-written reference summaries, and then evaluation was performed via the [[ROUGE]] score.
	- This works relatively well, but supervised learning on human-written summaries and using a ROUGE score is just a *proxy for what is ACTUALLY desired:* A model that writes high-quality summaries, according to humans.
	- ((In other words, there are a bunch of high-quality summaries that a model could write that aren't identical to the one that a human wrote -- it shouldn't be penalized for not maximizing the ROUGE score, necessarily.))
- If we explore supervised learning with RLHF, we allow pretrained language models to be finetuned to produce high-quality summaries directly based on human feedback.
	- This actually ended up with better results, as ranked by human reviewers (when compared to SFT, which totally makes sense).
	- ((Obama giving Obama a reward meme))

Beginning with a pretrained LLM, we iteratively:
1. Collect human comparison data
2. Train a reward model to predict human-preferred summaries
3. Use the model as a reward function for finetuning via RL

#### The Methodology
- The LLM is *first* trained using supervised finetuning over human reference summaries, producing a baseline from supervised finetuning that we'll later improve via RLHF.
- The methodology for RLHF is comprised of three steps:
	1. A dataset of human feedback is collected by:
		- Grabbing textual input to summarize from the dataset
		- Using several policies to sample a variety of summaries for the input
		- Grabbing two summaries from from the set of sampled responses
		- Asking a human annotator to identify the better of the two summaries
	2. Once the data has been collected, we use the comparison data to train a reward model that accurately predicts a human preference score given a summary produced by the LLM.
		- ((I'm curious -- if we're creating a dataset of "good" and "bad" responses, how does it learn to classify a single response, when we use it as the reward model during the finetuning?))
	3. From here, we use RL to finetune model: The authors use the PPO algorithm to finetune based on preference scores output by the reward model.
		- Authors also add a [[Kullback-Leibler Divergence]] term to the objective optimized by RL
			- This penalizes the policy from being *too different* from the supervised baseline policy during RLHF -- such an approach encourages exploration without [[Mode Collapse]], and prevents summaries written by the LLM from becoming *too different* from those that are seen during training.
![[Pasted image 20240407201953.png]]

Authors note that despite the ability of PPO to train an LLM that jointly models the policy and value function - they use separate models for the value function and policy.

Looking forward:
- Although RLHF was explored only in the context of summarization, the authors of this paper had an incredible amount of foresight about what to come.
- The approach proposed later became a standard methodology for aligning LLMs, as we will soon see with [[InstructGPT]].

## Training language models to follow instructions with human feedback (Paper 2)
- Going beyond the summarization domain, authors explore the use of RLHF for language model alignment by directly learning from human feedback -- the resulting model, [[InstructGPT]], is the sister model and predecessor to [[ChatGPT]].
- Given that this model is outlined and explained in detail, this work grants us significant insight into how LLMs from OpenAI are trained.

##### Approach
- We start with a set of prompts that are either written by human annotators or collected from OpenAI's API.
- We can then have annotators write responses to these prompts and finetune a pretrained LLM -- GPT-3 in particular -- in a supervised manner.
- Using this model, we can then *collect comparison data by asking humans to select preferred outputs from the LLM and apply the same RLHF process that is outlined for finetuning.*
- The resulting model is heavily preferred by humans (see above) and much better at following detailed instructions provided within each prompt.
![[Pasted image 20240407203514.png]]

##### The Methodology
- Authors curate a team of 40 human annotations screened with a test to judge their annotation quality to collect finetuning data for the LLM.
- The approach for RLHF used matches the approach used in the first paper almost completely.
- Using a pretrained LLM and a set of prompts used for finetuning, the alignment process proceeds according to the following steps:
	1. Collect human demonstrations of responses for each prompt.
		- (A dataset of over 13k prompts and responses is constructed)
	2. Train the model in a supervised fashion over human demonstrations.
	3. Collect comparison data.
	4. Train a reward model.
		- The reward model is trained over 33k prompts
		- Human annotators are shown 4-9 responses to a prompt, allowing them to rank these responses and generate a larger amount of comparison data more efficiently.
	1. Optimize the underlying LLM/policy with PPO.
	2. Repeat steps 3-5

![[Pasted image 20240407204152.png]]
- Above:
	- The distribution of use-cases that make up the data that's used to train the reward model. ((?))

Beyond the basic RLHF methodology, we see a few interesting tricks used to improve the finetuning process:
1. A [[Kullback-Leibler Divergence]] term is added to the training objective used for RL, which keeps the resulting model from diverging too much from the SFT model.
2. ==More pretraining updates are "mixed in" to the optimization process during RLHF, which mitigates the [[Alignment Tax]] and maintains the model's performance across a wide set of natural language benchmarks.==

> *"We were able to mitigate most of the performance degradations introduced by our fine-tuning. If this wasn't the case, these performance degradations would constitute an ==alignment tax== -- an additional cost for aligning the model."*

The final finetuning objective used for RLHF, including both the added pretraining updates and the KL divergence term:

![[Pasted image 20240407204653.png]]
Above:
- Honestly this is pretty hard to parse, for parts of it!

Experimental findings
- Authors train three models with 1.3 billion parameters, 6 billions parameters, and 15 billion parameters.
- We learn that human annotators prefer InstructGPT outputs over those of GPT-3, even for models with 10x fewer parameters.
- Such a result has similarities to observations, where we also see that finetuning via RLHF enables much *smaller* models to outperform lager models trained in a supervised manner.

==Notably, outputs from InstructGPT-1.3B are preferred to those of GPT-3, which has 100x more parameters!==
- And InstructGPT-175B produces outputs that are preferred to GPT-3 85% of the time.


# Modern Variants of RLHF
- RLHF was shown to be highly effective in the two above papers, but these were early works in the space of LLM alignment
- Over the last year, numerous modifications to the RLHF methodology have been proposed, including completely new algorithms for aligning LLMs.

[[LLaMA 2]] was one of the first open-source LLMs to invest extensively into alignment via RLHF.
- Human annotators only compare two responses to a prompt at once, instead of 4-9 in the case of InstructGPT
- Instead of collecting *binary preference data*, human annotators are instructed to identify responses that are *significantly* or *moderately* better than others 
	- ((Human labeling/feedback is more granular than binary "good" or "bad"))
- Comparisons are collected with respect to a single alignment criteria at a time (eg human annotator may receive a prompt and response pair focused on minimizing harmfulness in particular)

The approach above:
- Makes it easier for human annotators
- yielded more accurate comparisons
- Collected more granular feedback on model outputs.

To incorporate the *non-binary* feedback into RLHF, we can simply add a *margin* into the training objective for a reward model, which encourages responses with a large difference in quality to be pushed further apart in their preference scores.

![[Pasted image 20240407210049.png]]
Going further, authors explore two different reinforcement algorithms for finetuning via RLHF:
- [[Proximal Policy Optimization|PPO]]
- [[Rejection Sampling]]

After performing multiple rounds of RLHF, we see that ==the combination of these two learning algorithms in tandem== can drastically improve learning efficiency.

## Safe RLHF
- A recently-proposed modification to the basic RLHF algorithm. As previously discussed, the alignment process for LLMs requires the definition of several alignment criteria.
	- But the alignment criteria that we define might ==conflict== with each other sometimes -- for example, harmlessness and helpfulness are two commonly-used alignment criteria that tend to conflict with eachother.
	- ==*Typically*, these cases are addressed by training a separate reward model to capture each alignment criteria==

However ==Safe RLHF== proposes a new learning algorithm that better balances conflicting alignment criteria via the definition of rewards and costs within the alignment process -- see below:
![[Pasted image 20240407210202.png]]

## Pairwise PPO
- One interesting aspect of RLHF is the manner in which the reward model is trained and used -- namely, the reward model is trained based on *relative/comparative scores* -- "We want the preferred response to be scored higher than the other response."
- During optimization with PPO, we directly use the scalar output of the reward model as a training signal -- *we aren't using the reward model's output in a comparatively similar manner*
	- To address this, authors in Pairwise PPO paper propose a variant of PPO (shown above) that is modified and optimized to work better with the comparative human preference data that is collected for RLHF.

![[Pasted image 20240407210904.png]]


### Reinforcement Learning from AI Feedback ([[Reinforcement Learning from from AI Feedback|RLAIF]])
- Although RLHF is useful, a major downside of this technique is that it requires a lot of human preference data to be collected.
- LLaMA-2 uses over 1M human preference examples for alignment via RLHF, which is very onerous!
	- ==To mitigate this onerous requirement upon human data annotation, a recent line of work has explored automating human preference annotations with the use of a generic LLM.==
	- ==In other words, we perform RLHF with feedback provided by an AI instead of humans!==
		- ((We basically replace the bit where we train a reward model, and instead use a pretrained LLM!))


### Direct Preference Optimization ([[Direct Preference Optimization|DPO]])
- Another downside of RLHF is that it's an arguably complicated training procedure that's oftentimes unstable, relies upon reinforcement learning, and trains an *entirely separate reward model* to annotate human preference scores!
- As a solution, authors propose DPO, which is simpler, but appears to be equally performant. It's more stable and eliminates the need for a reward model by finetuning the LLM using human preferences directly.


![[Pasted image 20240407211053.png]]


## What makes RLHF so impactful?
- We have now seen RLHF successfully used in several domains, providing clear empirical evidence that RLHF is an effective finetuning technique.
- But why does RLHF work so much better than supervised learning?
	1. Human annotation process
		- In SFT, we train the LLM using human-provided *demonstrations of high-quality output*
			- This means that for each example, a human must manually write a full, high-quality response to the given prompt.
		- In RLHF, we generate responses to prompts automatically via an LLM, and simply ask the human annotator to rank several responses to the same prompt.
			- Ranking outputs is an easier task compared to writing outputs from scratch, so the annotations strategy for RLHF lessens the cognitive load of human annotators, leading to several notable benefits:
				1. Annotations are higher quality (i.e. more accurate)
				2. Collecting annotations is faster and more efficient
				3. Individual annotations can be focused on a particular alignment principle (eg Harmlessness)
	2. Beyond human quality
		- The annotation strategy for RLHF has benefits beyond annotation efficiency; we should notice that all responses used for collecting comparison data within RLHF are generated automatically by an LLM. ==This means that RLHF can train an LLM over responses *that go beyond the writing capabilities of human annotators, and therefore has the ability to surpass human performance!*==
			- ((It's probably true that the ceiling of discrimination/preferring is higher than the ceiling of generation, but there's still probably a level of generation that a human can't really discriminate well between, and we definitionally wouldn't be sure where that is, if we don't have an objective way of assessing a generation))
	3. Accurately capturing response quality
		- We see that the reward model that's created for RLHF is surprisingly accurate at capturing the quality of model responses.
		- Compared to automatic metrics like ROUGE, reward models can provide a more consistent and accurate evaluation of model output quality, as judged by the agreement rate with human annotators.
		- Optimizing the LLM based on the *preference score* from this reward model tends to produce a model that performs quite well.

We see that reward models tend to obey scaling laws that are somewhat similar to those of LLMs
- In particular, the quality of the reward model improves as we increase the size of the model and the amount of comparison data used for training.


# Closing Thoughts
- We should now have a deep understanding of RLHF and its impactful role in the training of generative language models.
- RLHF is a key innovation -- *along with several others* -- that catalyzed the recent generative AI boom.
- By enabling us to train generative LLMs directly from human feedback, RLHF empowered AI researchers to create generative models that were shockingly informative/useful and could even exceed the writing capabilities of humans.

#### Limits of SFT
- Collecting a supervised training dataset is hard, from humans -- it requires that humans write high-quality reference responses from scratch.
- There's a notable misalignment between the training objective used for supervised learning and what we actually want -- we want models that produce *high quality* outputs, not just outputs that are necessarily similar to the training data!

#### RLHF provides a solution
- To solve this misalignment, why not directly train the language model to produce outputs that we (the user) like?
- This is exactly the idea that inspired RLHF! To apply RLHF, we collect human preference data by having humans identify preferable responses, train a reward model to learn patterns in human preferences, then finetune the LLM with reinforcement learning to produce outputs that are more preferable to humans.




