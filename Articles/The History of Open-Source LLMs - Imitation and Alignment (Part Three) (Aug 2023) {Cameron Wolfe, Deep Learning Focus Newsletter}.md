Link: https://cameronrwolfe.substack.com/p/the-history-of-open-source-llms-imitation
See also:
- [[The History of Open-Source LLMs - Early Days (Part One) (Jul 2023) {Cameron Wolfe, PHD}]]
- [[The History of Open-Source LLMs - Better Base Models (Part Two) (Jul 2023) {Cameron Wolfe, Deep Learning Focus Newsletter}]]

-----

![[Pasted image 20240214175911.png]]

A majority of prior research on open LLMs focused heavily on creating better pre-trained base models -- but these models necessarily haven't undergone any fine-tuning or alignment, so they'll ultimately fail to match the quality of top closed-source LLM (eg [[ChatGPT]], [[Claude]]) do to their lack of alignment.
- Paid models are aligned extensively using [[Supervised Fine-Tuning]], [[Reinforcement Learning from Human Feedback]] ((and now, [[Direct Preference Optimization|DPO]])), which greatly enhances their usability.

This overview will take a look at recent research that aims to improve the quality of open-source LLMs via more extensive fine-tuning and alignment.

We'll go from initial models like OPT to the incredibly high-performing open-source LLMs that we have today (eg LLaMA-2-Chat).

![[Pasted image 20240214180312.png]]

The alignment process
- This overview will study the fine-tuning and alignment process for open-source LLMs. Prior to studying research here, we need to understand **what alignment is, and how it's accomplished!**
	- After pre-training, the model is able to accurately perform next-token prediction on arbitrary documents from its training corpus, but its output, if resampled over and over, might be repetitive and uninteresting.
	- Therefore, the model needs to be fine-tuned to improve its ==*alignment*, or its ability to generate text that aligns with the desires of a human user (e.g. follow instructions, avoid harmful output, avoid lying, produce interesting or creative output, etc.)==

![[Pasted image 20240214182227.png]]
SFT
- Alignment is commonly accomplished via two fine-tuning techniques: Supervised fine-tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF)
	- ((SFT is just more training, using specially selected "golden data," it seems. In the sense that I think it's just the same next-token prediction task on a higher-quality set of data.))
	- ((In comparison, RLHF is about two parts; First, we have a prompt, and we generate a number of possible outputs from it. The outputs are ranked by a labeler model (often a human), and the ranked outputs are used to train a *reward model*. In the second step, that reward model is used to scale this process, stepping in for the human in much the same way -- a prompt is selected, the policy generates an output, the reard model calculates a reward for the output, and the reward is used to update the policy using (eg) [[Proximal Policy Optimization|PPO]]))

[[Supervised Fine-Tuning]]
- Simply fine-tunes the model using a ==standard language-modeling objective over examples of *high-quality prompt and response pairs, often curated by human experts.*==
- Incredibly simple and effective, but requires careful curation of a dataset that captures "correct" behaviors.

[[Reinforcement Learning from Human Feedback|RLHF]]
- Trains the ==LLM directly on feedback from human annotations -- humans identify outputs that they like, and the LLM learns how to produce more outputs like this.==
- First, we obtain a set of prompts and generate several different outputs for the LLM for each prompt. We use a *group* of human annotators to score/rank these responses, based on their perceived quality. These scores can then be used to train a *==reward model==* (i.e. ==just a fine-tuned version of our LLM with an added regression head==) to predict the score of a response.
- Then, RLHF fine-tunes the model to maximize this score by using a reinforcement learning algorithm called PPO.
- ==Typically, the highest-performing LLMs are aligned by performing *both* SFT and RLHF in that order.==

# Imitation Learning
![[Pasted image 20240217004715.png]]
- With the release of [[LLaMA]], the open source research community finally had access to powerful base LLMs that could be fine-tuned and aligned for a variety of applications!
	- One of the most common directions of research during this time was *==imitation learning==*, which fine-tunes an LLM over outputs from another, more powerful LLM. Such an approach is inspired by the idea of [[Distillation|Knowledge Distillation]]
		- Note on Distillation: It's where we use a (large) fully-trained NN as a training signal for another (smaller) NN. Many different types of knowledge distillation exist, but the idea behind them remains the same -- if we train a NN on both t he normal training data *and* the output of the larger, more powerful NN, we will typically arrive at a better result than training a NN over the data alone. We *distill* some of the information from the larger network into the smaller "student" network.
	- The question posed by open-source imitation learning research was simple:
		- "Can we create a model that's just as powerful as ChatGPT or GPT-4 by just fine-tuning on responses from these models?"
			- ((Well... those specific models didn't (at the time) return probabilities, did they? Okay, I see that imitation learning doesn't require the probabilities as implied by knowledge distillation... we could call imitation learning also just creating synthetic data using a more powerful model and fine-tuning on that))
		- The approach to answer this was:
			1. Collect dialogue examples from these models (eg using the OpenAI API)
			2. Perform supervised fine tuning on this data using a normal language modeling objective of predicting the next token.

#### Initial efforts in imitation learning
- After LLaMA was released, researchers quickly began to release a variety of imitation models using dialogue derived from ChatGPT, obtained via the OpenAI API or through places like ShareGPT (where you share interesting GPT responses).
- In chronological order:
	- [[Alpaca]] finetunes [[LLaMA]]-7B using the [[Self-Instruct]] framework to automatically collect a fine-tuning dataset from GPT-3.5 for a cost of only $600 (data and finetuning cost)
		- ![[Pasted image 20240219140526.png]]
		- ![[Pasted image 20240219142940.png]]
		- The ==Self-Instruct== framework pioneers the idea of using LLMs to train themselves by generating synthetic [[Instruction Tuning]] data that can be used for fine-tuning.
			- Recall: ==Instruction fine-tuning==: The goal is to fine-tune an LLM over a set of "instructions," which are comprised of supervised data examples (i.e. input prompt + desired output), each paired with a description of the task being solved.
			- ![[Pasted image 20240219142704.png]]
			- ![[Pasted image 20240219142920.png]]
	- [[Vicuna]] then fine-tuned LLaMA-13B over 70k dialogue examples frmo ChatGPT (i.e. derived from ShareGPT) for a cost of only $300.
	- Koala fine-tuned LLaMA-13B on a large dataset of dialogue examples from both the Alpaca fine-tuning set and a variety of other sources like ShareGPT, HC3, OIG, Anthropic HH, and OpenAI's WebGPT/Summarization.
		- Compared to prior imitation models, Koala is fine-tuned over a larger dataset and evaluated more extensively.
	- GPT4ALL finetunes LLaMA-7B over 800k chat completions from `GPT-3.6-turbo` . 

The impact of imitation
- These models were published in close succession and claimed to achieve comparably quality to top proprietary models like ChatGPT and GPT-4 -- Koala was even found to match or exceed the quality of ChatGPT in some use cases!
	- Such findings seemed to indicate that model imitation could be used to distill the capabilities of any proprietary model into a smaller, open-source LLM.


#### Are imitation models a false promise?
- ![[Pasted image 20240219143719.png]]
- Above: See that fine-tuning via imitation in this case resulted in the imitation model mimicking the *structure* of the former model's response, it still got many of the factual bits wrong. It absorbed the style, not the knowledge.


### Experimental Setup - Is imitation learning useful?
- To determine the utility of imitation learning, authors curate a dataset of ~130k diverse dialogue examples form ChatGPT. Then, several different sizes of language models are fine-tuned over various amounts of imitation data before having their performance measured... ==We found, regarding imitation models==:
	- The amount of imitation data used for fine-tuning does not improve model quality in human evaluation trials.
	- Imitation models' performance on standardized benchmarks is often worse than the base model (and deteriorates as more imitation data is used)
	- Increasing the size of the base model consistency improves the quality of the resulting imitation models.
- What's going on here?
	- Imitation models do not actually match the quality of models like ChatGPT. Compared to proprietary LLMs, these models have a less extensive knowledge base, as revealed by the performance improvement observed with larger base models.
	- So why did it *seem like* these models were performing well?
		- ==Like in the picture example above, the imitation models learned to mimic the *style* of a model like ChatGPT -- as such human workers can be *tricked* into perceiving the model as high-quality, even if it generates factually incorrect information more frequently==!
			- # ==**THIS IS SO IMPORTANT!**==



#### So... is imitation learning actually useful?
- After the above learnings, the research community wasn't sure!
- Authors proposed two paths forward:
	1. ==Generating a much bigger and more comprehensive imitation dataset==
	2. Creating a ==better base model== to use for imitation learning

Interestingly, both of these recommendations were explored extensively by subsequent research and found to yield positive results.

[[Orca]] 🐋 is an imitation model based upon LLaMA-13B 
- Compared to prior work on imitation learning, ==Orca== is trained over a higher-quality, more-detailed, and more comprehensive dataset collected from ChatGPT and GPT4. 
	- Prior datasets can be considered "shallow" -- they're just examples of prompt and response pairs.
	- In contrast, ==Orca attempts to augment imitation datasets generated== (eg by GPT4) with:
		1. ==Explanation traces==
		2. ==Step-by-step thought processes==
		3. ==Complex instructions==
	- To do this, the model being imitated is prompted to provide detailed explanations of its response via specific prompts... As a result, when learning from powerful LLMs, Orca sees more than just the model's response -- it sees the detailed explanations and "thought processes" generated along with the model's response on complex prompts.
![[Pasted image 20240219144752.png]]Above: See that we use prompts regarding "thinking step by step", "providing a detailed answer", and "justify your steps" in the prompts. This imitation-learning process is sometimes called ==Explanation-tuning==.

- Orca ended up significantly narrowing the gap between open source imitation models and proprietary LLMs, but GPT-4 still consistently outperformed it -- even an improved imitation approach is not enough to fully match the quality of top proprietary models. But it still showed that improving the quality of datasets is incredibly important.


## Aligning Open-Source LLMs
- Imitation learning attempting to improve the quality of open-source base models by fine-tuning over the responses (and explanations) of proprietary LLMs. This worked, but it's (obviously) not the manner in which the top *proprietary* models were trained -- imitation is a just a (mostly effective) *shortcut* for creating more powerful open-source models.
- ==If we really want open-source LLMs to rival the quality of proprietary models, we need to invest significantly into alignment.==

> Closed product LLMs are heavily fine-tuned to align with human preferences, which greatly enhances their usability and safety. This step can require significant costs in compute and human annotation, and is often not transparent or easily reproducible.

So what's the hold up?
- ==The alignment process used by (eg) GPT-4 requires extensive compute and expert human annotation ==resources, and depends on proprietary ==data that is very expensive to collect.==

#### Prior work on open-source alignment
- Falcon, and the MPT models both underwent SFT on public alignment datasets -- open source LLMs haven't avoided alignment altogether, but in comparison to top proprietary models, we're still scratching the surface as of the time of this writing.

### LIMA: Data-Efficient Alignment
> A model's knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which *subdistribution* of formats should be used when interacting with users.

The [[LIMA]] paper: Less is More for Alignment.

- For quite some time, open-source LLMs mostly performed alignment via [[Supervised Fine-Tuning|SFT]] on public alignment datasets.
- Researcher studied the impact of SFT on pre-trained LLMs:
	- Authors constructed a dataset of ==1,000 dialogue examples to use for SFT==. This might not seem like very much, but the examples were *carefully curated* to ensure quality by using diverse prompts and a uniform output style or tone.
	- ==The dataset used to train LIMA is small of but of incredibly high quality. We see that LIMA performs surprisingly well when fine-tuned over this dataset; even **approaching the performance of state-of-the-art LLMs like GPT-4 or Claude**!==

Such a result shows that language models can be effectively aligned via a small number of carefully chosen examples, and that ==data quality is seemingly the most important factor in alignment via SFT.==

### LLaMA-2: Improving Transparency in Alignment Research
> LLaMA-2-Chat is the result of several months of research and iterative applications of alignment techniques, including both instruction tuning and RLHF, requiring significant computational and annotation resources.

- The recently-released [[LLaMA 2]] suite of LLMs is comprised of several open-source models with sizes ranging from 7-70B parameters.
- Compared to predecessors (ie LLaMA-1), it differentiates itself by pretraining over 40% more data (ie 2T tokens), and using an architecture that is optimized for fast inference ([[Grouped Query Attention]]).
- However, the LLaMA-2 suite contains more than just pre-trained LLMs! Authors invest heavily into the alignment process by fine-tuning each model (using both SFT and [[RLHF]]).
- The refined versions of LLaMA-2 perform incredibly well, emphasizing:
	1. Helpfulness
	2. Safety
- To ensure that the aligned model is both helpful and safe, data curated for both SFT and RLHF is filtered, collected, and annotated according to these principles.
![[Pasted image 20240219153325.png]]
- Supervised Fine-Tuning
	- The first step in LLaMA-2's alignment process is fine-tuning with SFT -- similar to other open-source LLMs, LLaMA-2 is first fine-tuned over publicly-available instruction tuning data -- but such data tends to lack in diversity and quality (see: importance of this in LIMA, above). As a result, the authors focused on filtering this data to collect a smaller set of high-quality data for SFT. ==Ultimately, 27,540 high-quality dialogue examples were used for SFT.==
		- Authors observed that collecting more data (beyond 27k, in their case) provided diminishing benefits. We don't need a ton of data for SFT, but it should be of high-quality!
- RLHF
	- LLaMA-2 is further fine-tuned using RLHf over a dataset of >1M examples of human feedback.
	- To collect this feedback, a binary protocol was adopted, in which human annotators were asked to write a prompt and choose the better of two generated responses from the LLM. Here, human preference data is collected according to both helpfulness and safety standards.
	- Human feedback is collected in batches, and LLaMA-2 is fine-tuned via RLHF between each batch. So there are several versions of each LLaMA-2-Chat models (5 in total) that are iteratively created after each trial of RLHF.
	- ==In total, LLaMA is fine-tuned on over 1M instances of human feedback throughout the entirety of the iterative RL process.==
![[Pasted image 20240219153753.png]]
((Above: See that the quality of LLaMA-2-Chat, as measured via harmlessness and helpfulness by two judges, increased over those successive versions/iterations of each model, as they were iteratively trained. It's interesting to me that they stopped where they did.))

Performance:
- The LLaMA-2-Chat models are currently (as of writing) the SoTA for open-source LLMs as per the Open LLM leaderboard.
- Found to even perform comparably to top proprietary models like ChatGPT when evaluated in terms of helpfulness and safety! Put smiply, these results heavily indicate that the quality of alignment performed for the LLaMA-2-Chat models is high.

Importance of LLaMA-2:
- It adopted a fundamentally different approach to prior work.
- Prior open source models mostly leverage SFT and public sources of dialogue data, but LLaMA-2 invested extensively into the alignment process, curating a great deal of *high-quality* dialogues and human preferences for both SFT and RLHF.

# Closing Remarks
- We've studied the journey of open-source language models from OPT to LLaMA-2 (only a year apart!)
- The open source community modes quickly, huh? Keep up!









