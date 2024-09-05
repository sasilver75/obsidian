June 17, 2024
[[NVIDIA]]
Paper: [Nemotron-4 340B Technical Report](https://arxiv.org/abs/2406.11704v1)
- Note: There's also a [Nemotron-4 15B Technical Report](https://arxiv.org/abs/2402.16819)
#zotero 
Takeaway: A 340B parameter model train trained on 9T tokens - notably 98% of the finetuning data (both SFT and preference-tuning) is *synthetically generated*, and they share a lot of information (and tooling) relevant to this process. Introduces the [[HelpSteer2]] preferences dataset, an "Iterative Weak-to-Strong Alignment" process, [[Genetic Instruct]] as an [[Evol-Instruct]]-like technique to inflate code data, and [[Reward-Aware Preference Optimization]] (RPO), which is applied in this paper after [[Direct Preference Optimization|DPO]], and takes advantage of the relative difference in Reward Model-given score between the chosen and rejected generations (which DPO is blind to). They open-source the base, instruct, and reward models with permissive licenses.

> "[[Nemotron-4]]'s entire paper is focused on the training of a very good reward model, to then use as their data filterer."

> "Good for synthetic generation because of its permissive use license, compared to models like GPT-4." - Zeta Alpha folks

----

### Introduction
- Recent efforts have focused on pretraining on more, higher-quality tokens. [[LLaMA 2]] was trained on 2 trillion tokens, while [[LLaMA 3]] was trained on 15 trillion tokens. The Nemotron-4 340B base model was trained with 9 trillion tokens on a high-quality dataset.
- The model is aligned with [[Supervised Fine-Tuning|SFT]], [[Direct Preference Optimization|DPO]], then [[Reward-Aware Preference Optimization]].
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
	- They use their 10k human-annotated [[HelpSteer2]] preference data to train Nemotron-4-340B-Reward, but we also need preference data with a *more diverse* domain of prompts (than in HelpSteer2), with higher-quality responses from top-tier models... so we generate synthetic preference data.
	- The preference data contains synthetic single-turn prompts, instruction-following prompts, two-turn prompts, as well as real-world prompts including ShareGPT prompts, LMSYS prompts, and prompts from [[GSM8K]] and [[MATH]].
		- For each prompt, we generate responses using multiple random intermediate models for diversity of responses (and even from the completed model, so it's later used to improve itself!).
	- Given multiple responses for each prompt, we need to judge their preference ranking and choose the chosen and rejected response.
		- Some tasks can be evaluated using ground-truth labels (eg GSM8K and MATH training dataset) or verifiers (instruction-following response can be validated with a Python script)
		- Other tasks don't come with an objective answer, so we consider using either *LLM-as-a-Judge* and *Reward-Model-as-a-Judge*. They end up choosing to use ==[[Reward-Model-as-a-Judge]]==.
- ==Iterative Weak-to-Strong Alignment==
	- High-quality data is essential for model alignment. 
	- What model should we use as a generator, and how does generator strength relate to data quality? How can we improve the dat generator?
	- *Inspired* by [[Weak-to-Strong Generalization]] (Burns et al., 2023), they develop a novel iterative approach to incrementally refine their data towards optimality.
		1. An initial aligned model is employed as the generator for both dialogue nad preference data.
		2. The data is then used for aligning a better base model using SFT and preference-tuning.
		3. We find that the teacher model doesn't impose a ceiling on the student model -- as the base model and alignment data are refined, the newly-aligned model is able to surpass the initial model by a significant margin.
	- Note that ==alignment procedures are performed in parallel with base model pretraining!== (In the first iteration, they use Mixtral-8x7B-Instruct-v0.1 as the initial aligned model.) This is sort of iterative [[Weak Supervision]].
		- Generated data is leveraged to train an intermediate checkpoint of Nemotron-4-340B-Bae, which we call `340B-Interm-1-Base`.
	- This iterative process creates a self-reinforcing flywheel effect, where improvements can be attributed to two aspects:
		1. When using the same dataset, the strength of the base model has a direct impact on the instruct model, with stronger base models yielding stronger instruct models.
		2. When using the same base model, the quality of the dataset plays a critical role in determining the effectiveness of the instruct model, with higher-quality data leading to stronger instruct models.
- Additional Data Sources
	- We incorporate several supplementary datasets to impart specific capabilities to the model, as listed below:
		- ==Topic following== (Incorporates the [[CantTalkAboutThis]] training set; dialogues intentionally interspersed with distractor turns to divert the chatbot from the main subject)
		- ==Incapable tasks== (Eg those that require access to the internet or real-time knowledge; we ask the model to respond with rejections)
		- ==STEM datasets== (Open-Platypus dataset, used to train the Platypus2 model, to improve STEM and logic knowledge. Also include PRM800K, SciBench, ARB, openbookQA)
		- ==Document-based reasoning and QA== (Document-grounded QA is important for LLMs; they leverage the FinQA dataset to improve numerical reasoning capability, and the wikitablequestions dataset to strengthen the model's understanding of semi-structured data)
		- Function calling (a subset of samples from Glaive AI)
- Alignment Algorithms
	- SFT
		- Experimental results showed that learning multiple behaviors concurrently can lead to conflicts between them, preventing the model from achieving optimal alignment on all tasks. So they use a ==two-stage SFT strategy==:
			1. Code SFT: Authors develop ==[[Genetic Instruct]]==, an approach that mimics evolutionary processes and is inspired by [[Self-Instruct]] and [[Evol-Instruct]], generating a large number of synthetic samples from a limited number of high-quality seed examples. Uses a fitness function employing an LLM to assess the correctness and quality of generated instructions and solutions.
			2. General SFT: A blended dataset of 200k samples that encompasses a variety of tasks.
	- Preference Fine-tuning
		- Their preference fine-tuning stage involves multiple iterations of model improvement using both [[Direct Preference Optimization|DPO]] and their new alignment algorithm, ==[[Reward-Aware Preference Optimization]]== (RPO).
		- Direct Preference Optimization (DPO)
			- Seems like they had a difficult time with DPO, and noticed that improvement of one metric usually came with degradation of other metrics. 
			- They add an additional SFT loss to prevent the policy network from shifting a lot away from the preference data.
			- The use a preference dataset of 160k examples including a variety of tasks.
		- Reward-Aware Preference Optimization (RPO)
			- The majority of their preference data is synthetic, with their preference ranking judged by Nemotron-4-340B-Reward.
			- DPO only uses the binary order between two responses -- meanwhile, the rewards contain more information! 
				- ==Some rejected responses will be only *slightly* worse than the paired chosen response, while in other cases the rejected response will be *far* behind. DPO is ignorant of this quality gap, leading to both overfitting and unnecessarily "unlearning" high-quality rejected responses.==
			- RPO attempts to approximate the reward gap using the implicit reward defined by the policy network, leading to the following new loss function:![[Pasted image 20240710233504.png]]
			- Above: $\pi$ is the policy network to train, $\pi_{ref}$ is the reference policy, $(x, y_c, y_l)$ are the prompt, chosen response, and rejected response. $r^*(x, y_c)$ and $r^*(x, y_l)$ are the rewards of the chosen and rejected responses by the reward model, respectively.
				- ((I think eta is some sort of shaping function for the reward difference.))
			- Compared to DPO, RPO learns to approximate the reward gap, which prevents the overfitting issue described earlier. Depending on the choice of the distance metric $\mathbb{D}$ , RPO is related to existing approaches like [[Direct Nash Optimization|DNO]], [[Identity-Mapping Preference Optimization|IPO]], Distill DPO, BRAINn. Authors used ![[Pasted image 20240710233852.png]]
			- "We use a preference dataset of 300k examples with a less-harsh quality filtering on chosen responses"; 3 iterations of RPO training
- Instruct Model Evaluation
	- Benchmarked against a bunch of the usual suspects, performed well.
	- Used a team of trained annotators to assess Nemotron-4-340B-Instruct against GPT-4-1106-preview, and found that Nemotron performed pretty well against it (though they usually tied).





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

![[Pasted image 20240710224136.png]]
The "Iterative Weak-to-Strong Alignment" workflow is inspired by [[Weak-to-Strong Generalization]] (2023).

![[Pasted image 20240711003257.png]]
Benchmark performance against open and closed models.

![[Pasted image 20240711003507.png]]
Showing the model's performance across two SFT phases, DPO, and 3 rounds of iterated RPO.

![[Pasted image 20240711004722.png]]
Interesting that it performed so poorly on the "Rewrite" category against GPT-4

![[Pasted image 20240711004909.png]]
Human evaluation results regarding length of generations.


# Non-Paper Figures