#article 
Link: https://cameronrwolfe.substack.com/p/data-is-the-foundation-of-language

-----

Large language models (LLMs) have been around for quite some time, but only recently has their impressive performance warranted significant attention from the broader AI community!

What was it that actually made recent models so impressive compared to their predecessors?
- It's likely our ==ability to perform alignment!== We trained LLMs to not just output the most likely next word, but to output text that will satisfy the goals of a human, whether it be by following an instruction or retrieving important information!

The overview will study the role and impact of alignment, as well as the interplay between alignment and pre-training.
- We will learn that the alignment process, though critical, primarily teaches an LLM ==steerability== and correct behavior or style, while *most* of the *knowledge* is gained during pre-training.
	- Here, steerability refers to our ability to control or modify an LLM's behavior in-context -- things like asking the LLM to assume different roles/personas, follow directions in a certain way, speak in a certain tone, etc. GPT-3.5 wasn't very steerable, but GPT-4 is way more reliable and capable of following detailed instructions. Steerability seems to come purely from RLHF; OpenAI invested heavily into hiring experts (chemical experts, software engineers) to evaluate model behavior and provide better data for RLHF. The result is a model that's more steerable, safer, and less likely to hallucinate.


# the LLM Training Pipeline

> LLMs are trained in ((at least)) two stages: 
> 1. Unupervised pretraining from raw text, to learn general-purpose representations
> 2. Large-scale instruction tuning and reinforcement learning, to better align to end tasks and user preferences.

Although LMs have been studied from a variety of perspectives, the creation of these models in 2023 does seem to follow a relatively standardized process:

![[Pasted image 20240219155502.png]]
Above: ((We might replace RLHF with PPO, etc.))

# Language Model Pre-training
![[Pasted image 20240219155832.png]]
- The pretraining process is the most computationally expensive step in the creation of the LLM, bevcaues it's churning through a huge corpus of unlabeled textual data and training itself using the standard language modeling objective of next-token prediction.
	1. Sample some text from the dataset
	2. Train the model to predict the next work

# The Alignment Process

![[Pasted image 20240219160003.png]]
After pre-training is complete, we have a base model that doesn't have any specialized abilities. To endow it with the ability to conduct interesting conversations, follow instructions, and more, we need to ==align== the model, or train it to replicate the behavior that's desired by a human user.

In most cases, the alignment procedure is based on two primary techniques:
1. [[Supervised Fine-Tuning]]
2. [[Reinforcement Learning from Human Feedback]]

These techniques can either be done individually or combined in sequence (as originally proposed by [[InstructGPT]], the predecessor to ChatGPT).

SFT
- A simple alignment approach -- we obtain examples of desired behavior, and directly fine-tune the LLM (using a language modeling objective) on this data.
	- ((This makes sense that it's SUPERVISED fine-tuning. You start with a dataset that already has what you want, and you just fit a curve to it (or slightly adjust your existing curve to it)))
- If we want to teach a base LLM to follow directions, we can obtain many examples of accurate responses to instruction-based prompts, and use those as our training examples!
- ==This technique is both simple and powerful, but is highly-dependent on curating a high-quality training dataset.== (See [[LIMA|Less is More for Alignment]] (LIMA))

RLHF
- Provide us with the ability to optimize an LLM's parameters based on feedback provided by humans. 
- Process
	1. Starting with a set of prompts, we first use the LLM to generate several potential outputs for each prompt. `(Prompts -> LLM Outputs)`
	2. Given these outputs, human annotations rank the quality of these responses, and use the rankings to train a reward model. `(LLM Outputs + Human Ratings -> Trained Reward Model)`
	3. We use the reward model (and its outputted scalar rewards) to optimize the LLM to maximize this reward via the [[Proximal Policy Optimization|PPO]] algorithm `(LLM + Trained Reward Model + PPO -> Reward-model-optimizing LLM)`

![[Pasted image 20240219171530.png]]
Above:
- The beauty of the RLHF process above is that =="human preferences" can be used to capture a variety of different properties.== 
	- For example, ==maybe we want to model to follow instructions better, output more interesting content, or even stop hallucinating (making up fake information). ALL of these can be optimized using RLHF, making it very robust.==

For those not familiar with [[Reinforcement Learning]], RLHF may be difficult to understand without some background reading. Links added to my reading list.

> "The model's capabilities seem to come primarily from the pre-training process -- RLHF doesn't improve exam performance (Without active effort, it actually degrades it). But steering of the model comes from the POST-TRAINING process -- the base model requires prompt engineering to even know that it should answer the question."
> - From GPT-4 blog


What is the purpose of alignment?
- Alignment is an incredibly active area of research. 
- Currently there's a discussion in the research community around *better understanding the role/purposes of alignment*. 
	- In the analysis section of GPT-4, we see that the role of alignment techniques like RLHF is to make the LLM more ==interesting, steerable, helpful, and harmless.==

We know that most *knowledge* in the model comes from the pre-training process, and that it seems like high-quality alignment can be done using a small amount of data and SFT (in the case of [[LIMA]])
- Such a result is especially interesting given the massive amount of human and computational resources that have been invested into aligning popular proprietary language models.


# Applying the LLM
![[Pasted image 20240219173136.png]]
- Once an LLM has underwent pre-training and alignment, it's more or less ready to be used in downstream applications -- but if you want to use it on a specific task, you might want to do some ==domain-specific fine-tuning== of the model. Furthermore (or alternatively), you might want to do some ==in-context learning==/prompting.

Domain-specific fine-tuning
- If deploying the model into a specialized domain (eg medical, legal, software, etc.), then it might make sense to further fine-tune the model over the types of data that it will see in this domain.
- This is quite simple -- it's just a continuation of training, but using a domain-specific corpus.

In-context learning
- Once we're ready to deploy the model (with or without specific fine-tuning), we should leverage in-context learning, which uses textual prompts to instruct/guide the model towards some desired behavior, to more accurately solve downstream tasks. 
- These prompts may include examples of correct solutions (i.e. few-shot examples), but this data is only used by the model as *context* when generating output -- it isn't used for training.

# LIMA: Less is More for Alignment
- Authors study the relative importance of pre-training versus alignment by training [[LIMA]], a derivative of LLaMA-65B that undergoes SFT (without RLHF) over a curated alignment dataset of only 1,000 well-curated examples of question-answering.
![[Pasted image 20240219174419.png]]
- When trained on these examples, we see that LIMA performs quite well, and even approaches SoTA performance compared to proprietary models like GPT-4 and Claude.
- ==LIMA reveals that language models can be effectively aligned via a small number of carefully-chosen examples, which emphasizes the role of data quality and diversity in training and aligning powerful language models.==

The Superficial Alignment Hypothesis
- Along these lines, the authors propose the [[Superficial Alignment Hypothesis]], which says that a models' capabilities are almost entirely learnt during pre-training, and alignment teaches it which subdistribution of formats should be used when interacting with users.
- The SAH simply states that alignment can be learned in a data-efficient manner given a set of examples with sufficient quality and diversity.

Curating Data for Alignment
- The dataset used for alignment is constructed from a combination of community QA forums and manually authored eamples.
- Unlike recent work that attempts to *automate* the curation of data for SFT (eg [[Self-Instruct]]), we see that data from both of these sources mentioned before is carefully and oftentimes manually filtered for both quality and diversity.
- ==Although manual curation takes time, it boosts the quality of the resulting dataset, which is found to be highly beneficial.==
	- ((It's clear to me that generating a shit-ton of okay-examples via Self-Instruct is maybe not optimal... And I'm guessing that the claims of "LIMA does alignment almost as good as GPT-4" is kind of like "Phi-2 is almost as good as 13B models" -- it's half-true. So what's the best mix between the cost effectiveness of generated data and the accuracy of expert-curated human-generated data? How do we optimally generate and discriminate examples? The Orca-2 paper might have some answers.))
	- ((Later, the article does talk about how they tested the generalizability of the alignment by testing it on a bunch of out-of-distribution prompts... and... LIMA did well! Huh.))

Sourcing the data
- The breakdown of LIMA's training data is shown in the table above; in the training set, 750 examples are sourced from community QA forums... 
	- To ensure data quality, these examples are either filtered manually or via "up vote" metrics.
- The remaining 250 examples are manually written by the authors -- 200 from scratch, and 50 from ==SuperNaturalInstructions==, a [[HuggingFace]] dataset.

When manually creating data, the authors maximize diversity and uniformity by making sure that:
1. Responses are *stylistically aligned* to the behavior of a helpful AI agent
	- Remember from the imitation-learning examples that imitation learning (SFT) is really mostly about acquiring style, rather than about acquiring knowledge. In a sense, this is the same thing as imitation learning -- it's just training.
2. Prompts are as diverse as possible

Can we automate this?
- In recent work on imitation learning with open-source LLMs, we usually see, we usually see that the data for fine-tuning is automatically curated -- it might be downloaded from online sources (eg ShareGPT) or obtained directly from LLM APIs.
- This approach is incredibly *efficient* compared to manual curation, and in some cases it even works well! [[Orca]] is an example of how to do this in a good way.

LIMA uses alternative approach to alignment that invests into the curation of high-quality data. Instead of automatically obtaining a large amount of data, we instead manually filter and select fewer examples. This smaller-scale (and more labor-intensive) selection process allows the diversity and quality of data to be controlled.


# The bigger picture
- In recent open-source language models (Alpaca, Vicuna, Koala), we have seen a variety of different LLMs that have adopted an automatic approach for curating data for SFT. In particular, these models use an *==imitation approach==* that:
	1. Collects a large amount of dialogues from other LLMs
	2. Performs supervised fine-tuning over this data.
- These models initially seemed to perform well, but we saw in more targeted evaluations that the actual quality of their alignment was poor -- they had absorbed the *style* of the larger proprietary model, but hadn't absorbed its *knowledge*.
- With this in mind, what seems to make LIMA's approach more effective?
	- ==Quality > Quantity==
		- Even in studies on imitation models, we see that increasing the *amount* of data in the fine-tuning set yields minimal and diminishing impact in the underlying model's performance.
		- We decided after learning that that we needed to either:
			- Create a more powerful base model to learn from
			- Create a better alignment dataset
		- ==LIMA studies how better alignment datasets can be created==, and that they have a huge impact on alignment via SFT.
- We see that LIMA tends to generalize well and oftentimes outperforms imitation models like Alpaca, which indicates that high-quality alignment data is still incredibly beneficial to LLM performance.

# The impact of Data Beyond Alignment
- We see that the quality of data is incredibly important for effectively aligning a language model -- however the importance of data quality and diversity goes beyond alignment alone.

Pre-training
- Across a variety of models, we've seen that the quality of data used for pre-training is important.
- Authors in [[Galactica]], a scientific model, found that ==training on smaller, heavily-curated datasets of high-quality science information yielded the best possible performance==. Other models like BioMedLM model are pre-trained over smaller-curated corpuses of technical content ((Also see Phi-2)). In the [[Falcon]] paper, the authors invested significant effort into developing a novel and efficient pipeline for extracting high-quality pre-training data from the web.

Alignment
- The [[Orca]] model heavily studies the role of data quality in solving the alignment problem.
	- The authors train a model using an imitation approach, but ==*augment* the data used for SFT with detailed information from the model about *how each problem was solved==!* They used prompting techniques like Chain-of-Thought, etc instead of just asking it to classify answers, etc.
		- Including this extra data in the alignment dataset is found to produce imitation models that are much more robust compared to Alpaca or Vicuna.

In-context learning
- The data used for in-context/few-short learning can massively impact performance.
- Recent research on few-shot learning shows us that factors such as:
	1. Ordering
	2. Distribution
	3. Format
- ...of examples can impact a model's performance. 
- The diversity of the exemplars given is incredibly important.


# Closing Remarks

The major conclusions from work covered within this overview are twofold:

1. _The Superficial Alignment Hypothesis_: LLMs learn their knowledge during pre-training and alignment teaches them how to properly interact with users.
    
2. The quality and diversity of data is incredibly important to the alignment process (much more so that data scale).





