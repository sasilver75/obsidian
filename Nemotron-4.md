June 17, 2024
[[NVIDIA]]
Paper: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704v1)
- Note: There's also a [Nemotron-4 15B Technical Report](https://arxiv.org/abs/2402.16819)
#zotero 
Takeaway: ...

> "[[Nemotron-4]]'s entire paper is focused on the training of a very good reward model, to then use as their data filterer."

> "Good for synthetic generation because of its permissive use license, compared to models like GPT-4." - Zeta Alpha folks

----

### Introduction
- Recent efforts have focused on pretraining on more, higher-quality tokens. [[LLaMA 2]] was trained on 2 trillion tokens, while [[LLaMA 3]] was trained on 15 trillion tokens. The Nemotron-4 340B base model was trained with 9 trillion tokens on a high-quality dataset.
- The model is aligned with [[Supervised Fine-Tuning|SFT]], [[Direct Preference Optimization|DPO]], then [[Proximal Policy Optimization|PPO]].
- Authors release `Nemotron-4-340B-Base`, `Nemotron-4-340B-Instruct`, and also the [[Reward Model]], `Nemotron-4-340B-Reward` as open-access models with a permissive license.
	- Because of the permissive license, the authors note that these would be great models to use for synthetic data generation.
- They use synthetic data heavily to create `Nemotron-4-340B-Instruct` -- ==over 98% of their training data was synthetically generated throughout the alignment process.==
	- ((I think this just means that the fine-tuning data was synthetically generated, not necessarily the pretraining data?))
	- They also ==release their synthetic data generation pipeline, which includes synthetic prompt generation, response and dialogue generation, quality filtering, and preference ranking.==
		- "Going forward, we will share more tools such as NVIDIA Inference Microservices (NIMs) for synthetic data generation"
### Pretraining
- Our pretraining data blend consists of three different types of data:
	- 70% English natural language data
		- web documents, news articles, scientific papers, books, etc.
	- 15% Multilingual natural language data
		- 53 natural languages from both monolingual and parallel corpora
	- 15% Source code data
		- 43 programing languages
- We train for 9T tokens on this data, with the first 8T in the formal pretraining phase, and the last 1T in a *==continued pretraining phase.==*
- Nemotron-4-340B-Base is similar in architecture to Nemotron-4-15B-Base; it's a standard decoder-only Transformer architecture.
	- [[Rotary Positional Embedding|RoPE]]
	- [[SentencePiece]] tokenizer
	- [[Squared ReLU]] activations
	- No bias terms
	- [[Dropout]] rate of zero ((Meaning no dropout))
	- Untied input-output embeddings
		- (Tying I/O embeddings refers to a technique where the input embedding layer and output embedding layers share the same wight matrix; often called [[Weight Tying]]). Untied embeddings cost more parameters but allow for more flexibility, because input and output representations can evolve separately.
	- 96 Transformer Layers
	- 18432 Hidden dimension
	- 96 Attention heads
	- 8 KV heads
	- 4096 Sequence length
	- 256,000 Vocabulary size
- Continued training: ==Authors find that switching the data distribution and learning rate decay schedule at the end of model training significantly improves data quality.==
	- After having pretrained for 8T tokens, they use the same loss objective and perform continued training on 1T additional tokens.
	- In this additional phase of continued training, we use ==two distinct data distributions==
		1. (Majority of continued training tokens) Utilizes tokens that have already been introduced during pre-training, but with a distribution that places larger sampling weight on higher-quality sources.
		2. A small number of QA-style alignment examples to better allow the model to respond to such questions in downstream evaluations, while also up-weighting data sources that come from areas of lower model accuracy.
	- The idea is that ==such an ordering and style of data distributions allows for the model to *gently transition* from the pre-training dataset and better learn from the data introduced during the final stage of training==.
### Alignment
- The reward model plays a pivotal role in model alignment, serving as a crucial judge for preference ranking and quality filtering 
- To help develop a strong reward model, they collect a ==dataset of 10k human preference data== called [[HelpSteer2]], following a methodology similar to the one described in [[HelpSteer]]. 
- Unlike pairwise ranking models used in other work, they find that *==multi-attribute regression reward models==* are more effective at disentangling real helpfulness from irrelevant artifacts (eg [[Verbosity Bias]]).
	- They replace the final softmax layer on Nemotron-4-340B-Base with a new regression head that maps the hidden states of the last layer into a five-dimensional vector of HelpSteer attributes (Helpfulness, Correctness, Coherence, Complexity, Verbosity). During inference, these attribute values are then aggregated by a weighted sum to be an overall reward.
	- The authors note that the model tops the leaderboard on [[Nathan Lambert]]'s [[RewardBench]].
- Authors note that existing permissive open-source datasets are increasingly inadequate for training the most well-aligned models, and that collecting high-quality data from humans is a time-consuming and costly endeavor.
	- During the entire alignment process, they rely only on ~20k human-annotated data (with the remaining 98% of the data being synthesized)
		- 10k for SFT
		- 10k [[HelpSteer2]] data for reward model training and preference tuning
- The first step in synthetic data generation is generating synthetic prompts.  This enables us to control the prompt distribution to cover a diverse set of scenarios.
	- Task diversity (writing, Open Q&A, closed Q&A)
	- Topic diversity (stem, humanities, daily-life)
	- Instruction diversity (JSON output, # paragraphs, Yes-or-No answers)
- To ensure prompt diversity, they follow a similar approach to the generation of the [[UltraChat]] dataset and CAMEL -- using the Mistral-8x7B-Instruct-v0.1 model as their generator to generate synthetic prompts separately for the tasks including openQA, writing, closed QA, and match&coding.
- For each prompt task, they seed generation with a diverse set of topics or keywords so that the prompts cover a wide variety of topics.
- They also generate *==instruction-following prompts==* in which they explicitly define the format of the anticipated response (eg "The output has to be in the JSON format."). 
- They generate ==*two-turn prompts*==, which include the user-assistant interaction history, to boost model's conversation skills.
- ==Synthetic single-turn prompts==
	- To collect diverse topics, they prompt their prompt generator to output a diverse set of macro-topics, and related subtopics for each of the synthetic macro-topics. Including synthetic macro topics, synthetic subtopics, and manually-collected topics, we gathered 3K topics in total.
		- We generate ==open Q&A prompts== by prompting the generate to generate questions related to each topic, and then refine the question to be more detailed and specific, since we observe that the initially-generated questions are usually very short.
		- For ==writing prompts==, the prompts include instructions about the generation of certain types of documents (newsletters, essays) about the given topic, and again ask the generator to refine the generated tasks to include more details.
		- For ==closed Q&A prompts==, they start with the documents in the [[C4]] dataset, and ask the generator to output instructions like "summarize the given text," or "based on the given text, what is xxx?" Then they concatenate the document with the manually-generated instruction using manually-defined templates.
		- For ==math and coding prompts==, we collect a diverse set of keywords (eg division, loop, lambda function) from math to python programming. They generate high-level topics and subtopics for math and python programming, then prompt the generator to classify whether Wikipedia entities are related to math or python programming, respectively. We prompt the generator to generate problems related to each keyword.
- ==Instruction-following prompts==
	- We want to generate prompts like "*Write an essay about machine learning. Your response should have three paragraphs.*"
	- Starting with a random set of synthetic prompts, we randomly generate a synthetic instruction ("Your response should have three paragraphs.") from the "verifiable" instruction templates described in [Zhou et al (2023)](https://arxiv.org/abs/2311.07911), before concatenating the prompt and instruction together with manually-defined templates.
	- They also generate *multi-turn instruction-following prompts* where the instruction applies to all future conversations ("Answer the question and all following questions according to {instruction}")
	- In addition, they construct *==second-turn instruction-following prompts==*, in which the user requests previous of the previous response, according to a given instruction.
- ==Two-turn prompts==
	- While datasets used for SFT are usually multi-turn, the preference data for preference fine-tuning is usually single-turn!
	- The construct two-turn prompts: They source the first user prompts from [[ShareGPT]] and generate the assistant response and the next turn question with our intermediate instruct models. ("User: XXX; Assistant: XXX; User: XXX")
- Real-world LMSYS prompts
	- To better mirror real-world user requests, they also draw prompts from the [[LMSYS-Chat-1M]] dataset, combining all prompts and dividing them into two distinct/non-overlapping sets:
		- For SFT (removing prompts that are flagged as potentially unsafe)
		- For human preference learning (retaining those flagged as potentially unsafe)
	- Authors do a little analysis using their reward model and see that it's easier to be "helpful" for synthetically-generated prompts than it is for these human-generated LMSYS prompts, making them interesting to train on.
- Synthetic Dialogue Generation
	- SFT enables models to learn to interact with users in a dialogue format.
	- To foster multi-turn capabilities, they design each dialogue to comprise *==three turns==*, creating a more dynamic and interactive conversation flow.
		- The model alternates between simulating the Assistant's and User's roles.
		- In order to elicit the desired behavior in user turns, ==we find it essential to provide the model with explicit prompts that describe user personalities.== They also post-process user turns to mimic real-world user questions by ==*excluding polite statements*== ("Thank you for ...", "Sure I'd be happy to...").
	- They use Nemotron-4-340B-Reward to assess the quality of dialogues, assigning a score to each sample and ==filtering== out those that fall below a predetermined threshold.
- Synthetic Preference Data Generation
	- They use their 10k human-annotated [[HelpSteer2]] preference data to train Nemotron-4-340B-Reward, but we also need preference data with a *more diverse* domain of prompts (than in HelpSteer2), with higher-quality responses from top-tier models... so we 



Abstract
> We release the Nemotron-4 340B model family, including ==Nemotron-4-340B-Base==, ==Nemotron-4-340B-Instruct==, and ==Nemotron-4-340B-Reward==. Our models are open access under the NVIDIA Open Model License Agreement, a permissive model license that allows distribution, modification, and use of the models and its outputs. These models perform competitively to open access models on a wide range of evaluation benchmarks, and were sized to fit on a single DGX H100 with 8 GPUs when deployed in FP8 precision. We believe that the community can benefit from these models in various research studies and commercial applications, especially for generating synthetic data to train smaller language models. ==Notably, over 98% of data used in our model alignment process is synthetically generated==, showcasing the effectiveness of these models in generating synthetic data. To further support open research and facilitate model development, we are also open-sourcing the synthetic data generation pipeline used in our model alignment process.


# Paper Figures
![[Pasted image 20240710202313.png]]
See evaluations of each of the three released models compared with 3 comparable models (LLaMA3-70B, Mixtral8x22, Qwen-2 72B).

![[Pasted image 20240710204520.png]]
Model hyperparameters

![[Pasted image 20240710204602.png]]
Schedule for Batch Size rampup

![[Pasted image 20240710205053.png]]
See that Nemotron-4-340B-Base compares favorably to Mistral 8x22B, LLaMA-3-70B, and Qwen-2-72B. It's also a much larger model than the rest of these, to be fair.

![[Pasted image 20240710205858.png]]
Showing the Nemotron-4-340B-Reward model's performance on RewardBench datasets; at the time of release, it had the best performance. It's powered by the [[HelpSteer2]] dataset.

![[Pasted image 20240710213137.png]]
Showing the process for *prompt generation* across four tasks.

![[Pasted image 20240710213312.png]]



# Non-Paper Figures