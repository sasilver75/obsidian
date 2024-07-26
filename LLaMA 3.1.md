July 23, 2024
[[Meta AI Research]]
[The LLaMa 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)
#zotero 
Takeaway: ...


---

# Introduction
- Authors present the LLaMA 3 Herd of models, which natively support *multilinguality, coding reasoning, and tool use.*
	- Largest model: 405B parameters, 128K token context
	- The results in this paper are for the LLaMA 3.1 models, but we'll refer to them as LLaMA 3 throughout for brevity.
- Authors seek to optimize three levers in the development process:
	1. Data: Compared to prior LLaMA versions, we improve quantity and quality of data used for both pre-training and post-training. ==15.6T multilingual tokens for LLaMA 3, compared to 1.8T tokens for LLaMA 2==.
	2. Scale: Our 3.1 405B model uses 3.8x10^25 FLOPs, almost ==50x more FLOPs than the largest version of LLaMA 2.==  The ==405B model is approximately compute-optimal size for our training budget==, and the smaller models (8b, 70b) are trained well-beyond compute optimality.
		- We use the flagship model to further improve the quality of smaller models during post-training via [[Distillation]] ((continuing the trend with recent [[Gemeni 1.5]] Flash, [[GPT-4o Mini]] and [[Gemma 2]] models))
	3. Managing complexity: 
		- We make design choices that seek to maximize our ability to scale the model development process. 
			- We opt for a ==standard dense Transformer model== architecture with minor adaptations, rather than for a [[Mixture of Experts]] models, to maximize training stability.
			- We use a relatively simple post-training procedure using [[Supervised Fine-Tuning|SFT]], [[Rejection Sampling]], and [[Direct Preference Optimization|DPO]], ==as opposed to using more complex RL algorithms that tend to be less stable and harder to scale.==
- The 405B model performs on par with leading language models like [[GPT-4]] turbo, [[Claude 3.5]] Sonnet, [[GPT-4o]], etc.
- Authors release base and post-trained versions of the models, as well as a new version of the LLaMa Guard series of models, [[LLaMA Guard 3]], for input and output safety.
- Authors are currently still working on multimodal extensions to the models that enable image recognition, voice recognition, and speech-understanding capabilities.

# General Overview
- Development of the 3.1 models is broken into two main stages:
	1. Pre-training
		- Self-supervised learning, obtaining large amounts of knowledge about the world. The 405B is pretrained on 15.6T tokens using a context window of 8K tokens, which is then followed by a [[Continued Pretraining]] stage that increases the supported context window to 128K tokens.
	2. Post-training
		- We align the model with human feedback in *several rounds,* each of which involves
			- [[Supervised Fine-Tuning|SFT]]
			- [[Rejection Sampling]]
			- [[Direct Preference Optimization|DPO]]
		- We also integrate new capabilities:
			- Tool-use
			- Coding
			- Reasoning
			- ...
		- Safety mitigations are also incorporated into the model at post-training stage.
		- We also perform experiments in which we add image, video, and speech capabilities in a ==compositional approach==.
			- Multi-modal encoder pre-training (images, speech)
				- The image encoder is trained on large amounts of image-text pairs, teaching models the relation between image/text.
				- The speech encoder is trained using self-supervised approach that masks out parts of speech inputs and tries to reconstruct masked-out parts via discrete-token representation.
			- Vision Adapter training
				- We train an adapter integrating the pre-trained mage encoder into the pre-trained LM; adapter consists of a series of [[Cross-Attention]] layers feeding image-encoder representations into the LM. During adapter training they also update the parameters of the image encoder, but we intentionally do not update the LM parameters.
				- Authors also train a video adapter on top of the image adapter for paired video-text data, enabling the model to aggregate information across frames.
			- Speech Adapter training
				- Speech encoder is integrated into the model via an adapter converting speech encodings into token representations that can be fed directly into the finetuned LM.
				- The parameters of the adapter and encoder are jointly-updated in a SFT stage to enable high-quality speech understanding.
				- Authors also integrate a text-to-speech system.
		- ==These multimodal experiments are still under development and not yet ready for release.==

# Pretraining
- Pretraining involves:
	1. Curation/filtering of large-scale training corpus
	2. Development of a model architecture and scaling laws to determine model size
	3. Development techniques for efficient pretraining at scale
	4. Development of a pre-training recipe

## Pre-Training Data
- Authors use pre-training data from a variety of sources, with a knowledge cutoff at the end of 2023, applying deduplication and cleaning mechanisms (PII, bad domains) on each data source to obtain high-quality tokens.

### Web Data Curation
- We implement filters designed to remove data from websites likely to contain unsafe content or high volumes of PIPI.
- We process the raw HTML content to extract high-quality, diverse text, building a ==custom parser that extracts HTML content,== optimizing for precision in boilerplate removal and content recall.
	- ((Rather than use, eg, [[Common Crawl]]'s pre-processed WET version of the crawl))
	- Authors evaluate the parser's quality in human evaluations, comparing it with popular third-party HTML parsers (eg [[Trafilatura]] or [[Resiliparse]]) and find that it performs "favorably".
	- Authors are careful about processing HTML pages with math/code content to preserve the structure of that content, maintaining the image `alt` attribute text, since math content is often represented as pre-rendered images, where the math is also provided in the `alt` attribute.
	- We find markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so ==we remove all markdown markers==.
- Deduplication (==URL, Document, Line-level deduplication==)
	- URL-level deduplication (keeping the most recent version for pages)
	- Document-level deduplication (performing global [[MinHash]] to remove near-duplicates)
	- Line-level deduplication (aggressively, similar to [[CCNet]]. We remove lines that appear more than 6 times in each bucket of 30M documents.)
- Heuristic Filtering
	- We develop heuristics to remove low-quality documents, outliers, and documents with excessive repetitions.
	- Use ==duplicated n-gram coverage ratio== to remove lines consisting of repeated content like logging or error messages.
	- Use "dirty word" counting to filter out adult websites.
	- Use a token-distribution [[Kullback-Leibler Divergence|KL-Divergence]] to filter out documents containing excessive numbers of outlier tokens compared to training corpus distribution.
- ==Model-based Quality filtering==
	- We *experiment* with various model-based quality classifiers to sub-select high-quality tokens. These include using fast classifiers like [[fastText]] trained to recognize if a given text is Wikipedia-like, as well as more ==compute-intensive [[RoBERTa]]-based classifiers trained on LLaMA 2 predictions.==
		- ((Which, if you recall, was used in the original LLaMA 3 paper for quality filtering of data, I believe?))
	- We create a training set of cleaned web documents, describe the quality requirements, and instruct LLaMA 2-chat to determine if the documents meet these requirements.
	- ==We use [[DistilRoBERTa]] to generate quality scores for each document for performance reasons== ((unclear, is this DistilRoBERTa finetuned on the LLaMA 2-chat quality score ratings? Yes, they make it clear in the next paragraph that this is what's up: "DistilledRoberta models trained on web data annotated by LLaMA 2"))
- Code and reasoning data
	- Similar to [[DeepSeek-Coder-V2]], we build domain-specific pipelines that extract code/math web pages, using [[DistilRoBERTa]] models trained on web data annotated by [[LLaMA 2]]. 
	- Since the token distribution of code and math is substantially different than that of natural language, thees pipelines implement domain-specific HTML extraction.
- Multilingual data
	- The multilingual text processing pipeline has some unique features:
		- A [[fastText]]-based language identification model to categorize documents into 176 languages.
		- We perform document-level and line-level deduplication within data for each language.
		- We apply language-specific heuristics and model-based filters to remove low-quality documents.

### Determining the data mix
- Our main tools in determining the proportion of different data sources in the pretraining mix are *knowledge classification* and *scaling law experiments*
	1. ==Knowledge classification==: We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to *downsample* data categories that are over-represented on the web (eg arts, entertainment).
	2. ==Scaling laws for data mix==: To determine the best data mix, we perform scaling law experiments in which we train several small models on a single data mix, and use that to predict the performance of a larger model on that mix. We do this for many different data mixes before choosing one to train as a larger model.
- In summary, the ==final pretraining datamix contains ~50% general knowledge, 25% math and reasoning tokens, 17% code tokens, and 8% multilingual tokens.==
	- ((This doesn't seem to be their most granular taxonomy though, if they're downsampling arts and entertainment. And what kind of multilingual tokens are they?))

### Annealing Data (More later in Section 3.4.3)
- We find that ==annealing== on small amounts of high-quality code and math data can boost the performance of pre-trained models on key benchmarks ((for the smaller 3.1 models)).
	- We perform this annealing with a data mix that upsamples high-quality data in select domains.
	- We find that annealing improves the performance of a pre-trained LLaMA 3 *8B* model on the [[GSM8K]] and [[MATH]] validation sets by 24% and 6.4$, respectively... however the improvements on the *405B* model were negligible.
- Using ==Annealing enables us to judge the value of small domain-specific datasets== -- we anneal the learning rate of a 50% trained LLaMA 3 8B model linearly to 0 on 40B tokens. We assign 30% weight to the new dataset and the remaining 70% weight to the default data mix.
	- ==Using annealing to evaluate new data sources is more efficient than performing scaling law experiments for every small datasets.==
	- ((The learning rate being gradually reduced during this process is similar to the metallurgical process of annealing where heat is gradually reduced))

### Model Architecture
- LLaMA 3 uses a standard dense Transformer architecture that doesn't deviate strongly from previous LLaMA models. ==Our performance gains are driven primarily by improvements in data quality, diversity, and scale==.
- Some small modifications:
	- Use of [[Grouped Query Attention]] (GQA) with 8 KV heads to improve inference speed and reduce size of KV Cache during decoding.
	- Vocabulary size of 128K tokens, combining 100K tokens from the [[Tiktoken]] tokenizer with 28K new tokens to better-support non-english languages. 
		- Improves compression rates on a sample of English data from 3.17 to 3.94 characters per token, ==enabling the model to read more text for the same training compute==.
		- ((This is important! It's not just about inference-time compute, but also making better use of compute during training time, enabling better scaling!))
	- We increase [[Rotary Positional Embedding|RoPE]] base frequency hyperparameter to 500,000, enabling us to better support longer contexts.

### Scaling Laws
- ==We develop our own scaling laws to determine optimal model sizes for our flagship model given our pre-training compute budget.==
- A major challenge is to forecast the flagship model's performance on downstream benchmark tasks, due to a couple of issues:
	1. ==Existing scaling laws typically predict only NTP loss ([[Perplexity]]) rather than benchmark performance.==
	2. Scaling laws can be noisy and unreliable because they're developed based on pre-training runs with small compute budgets.
1. We implement two-stage methodology to develop scaling laws that accurately predict downstream benchmark performance:
	1. Establish a correlation between compute-optimal model's negative log-likelihood on downstream tasks and the training FLOPs.
	2. Correlate negative log-likelihood on downstream tasks with task accuracy, using both the scaling law models and older models trained with higher compute flops...
- ... We use the resulting compute-optimal models to forecast performance of L3 on benchmark datasets. We find this two-step scaling law prediction to be quite accurate.

### Infrastructure, Scaling, and Efficiency
- LLaMA 1 and 2 models were trained on Meta's AI Research Supercluster; the training for LLaMA 3 was migrated to Meta's production clusters, which optimizes for production-grade reliability, which is essential as we scale up training.
- Compute: L3.1 405B was trained on  ==16,000 H100 GPUs==, with training jobs scheduled using MAST, Meta' global-scale training scheduler.
- Storage: Tectonic, Meta's general purpose distributed filesystem, is used to build a storage fabric fro L3 pretraining, offering 240PB of storage out of 7,500 SSD-equipped servers, supporting a sustainable throughput of 2TB/s and a peak of 7 TB/s.
	- ==A major challenge is supporting highly bursty checkpoint writes that saturate the storage fabric for short durations.==
- Network: L3.1 405B used RDMA over Converged Ethernet (RoCE) fabric... smaller models in the L3 family used Nvidia Quantum2 Infiniband fabric. Both RoCE and Infiniband clusters leverage 400Gbps interconnects between GPUs.
- Parallelism for model scaling:
	- We use ==4D parallelism==, a combination of four types of parallelism methods, to shard the model. Combines:
		- [[Tensor Parallelism]] (A type of [[Model Parallelism]]): Splits individual weight tensors into multiple chunks on different devices.
		- [[Pipeline Parallelism]]: Partitions the model vertically into stages by layers, so different devices can process in parallel different stages of the full model pipeline.
		- [[Context Parallelism]]: Divides the input context into segments, reducing memory bottleneck for very long sequence length inputs.
		- [[Data Parallelism]] ([[Fully Sharded Data Parallelism|FSDP]]): Shards the model, optimizer, gradients while implementing data parallelism, which processes data in parallel on multiple GPUs and synchronizes after each training step.
- ==Authors achieve an overall BF16 Model FLOPs Utilization ([[Model FLOPs Utilization|MFU]]) of 38-43%==.
	- ((This is a pretty interesting fact. We can't expect to get 100% GPU utilization! Avoid the "bubble!"))
- ...
- When training with 16K GPUs, the space of failure scenarios is large, and the synchronous nature of training makes us less fault-tolerant; a single GPU failure may require a restart of the entire job.
	- Despite this, they achieved ==higher than 90% effective training time==, while supporting automated cluster maintenance (firmware, Kernel upgrades).
	- ==During a 54-day snapshot period of pretraining, we experienced a total of 466 job interruptions== (47 of which were planned interruptions due to automated maintenance operations.) 78% of unexpected interruptions are attributed to hardware issues like GPU or host component failures, with GPUs accounting for 58.7% of all unexpected issues.
	- ==Despite the large number of failures, significant manual intervention was required only three times during this period, with the rest of issues handled by automation.==
- To increase effective training time, we reduce job startup and checkpointing time, and develop tools for fast diagnosis and problem resolution.
- Impact of environmental factors on training performance: For 405B, ==we noted a diurnal 1-2% throughput variation based on time-of-day, the result of higher mid-day temperatures impacting GPU dynamic voltage and frequency scaling.==

### Pre-Training Recipe
- Three main stages: ==initial pre-training, long-context pre-training, annealing==.
Initial Pretraining
- Use a ==cosine learning rate schedule==, with a peak of 8e-5, with a linear warm up of 8000 steps and a decay to 8e-7 over 1,200,000 training steps.
- We use ==batch size ramping==, starting with a lower batch size early in training to improve training stability, and increase it subsequently to improve efficiency.
Long-Context Pretraining
- In the final stages of pretraining, they train on long sequences to support context windows of up to 128K tokens.
	- (They don't do it earlier because of quadratic self-attention cost)
- ==We increase it in increments==, pretraining until the model has successfully adapted to an increased context length (performance on short-context evaluations has recovered completely, and model perfectly solves [[Needle in a Haystack|NIAH]] tasks up to that new length).
Annealing
- During pre-training on the final 40M tokens, we linearly anneal LR to 0, maintaining a context of 128K tokens; we also adjust the datamix to upsample data sources of very high quality.
- ((I wonder, is this basically doping))

# Post-training Data
- We produce an aligned L3 model by applying several rounds of post-training, with each round involving [[Supervised Fine-Tuning|SFT]] followed by [[Direct Preference Optimization|DPO]] on examples collected either from humans or generated synthetically.
- We first train a reward model on top of the pretrained checkpoint using human-annotated preference data, and then finetune pre-trained checkpoints with SFT and further align them with DPO.
- ==Because L3 has new capabilities like tool use, we design a new multi-message chat protocol/chat dialog format that uses various special headers and termination tokens.==
- We train a RM covering different capabilities on top of the pre-trained checkpoint. In addition to standard preference pair of (chosen, rejected) response, annotations also create a third "edited response" for some prompts). So each preference ranking sample has either two or three responses with clear ranking (eg edited>chosen>rejected).
- Reward model is used to perform [[Rejection Sampling]] on human-annotation prompts; together with synthetic data and other sources, we finetune pretrained LM using a standard [[Cross-Entropy]] loss on the target tokens.
- We further train SFT models with [[Direct Preference Optimization|DPO]] for human preference alignment, primarily using the most recent batches of preference data collected using the best performing models from previous alignment rounds.
	- We mask out special formatting tokens (like header and termination tokens) from both chosen and rejected responses.
- We average models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage.



- Preference data annotation process is similar to LLaMA 2; deploy multiple models for annotation after each round and sample two responses from two different models for each user prompt.
	- We ask annotators to rate the strength of their preference by categorizing it into one of four levels (==significantly better, better, slightly better, marginally better==).
		- For DPO and reward modeling
	- ==We also incorporate an editing step after preference ranking to encourage annotators to further improve the preferred response -- allowing them to edit the chosen response *or* prompt the model with feedback to refine its own response.==
		- ((On the LS podcast episode, they said this was basically to get the model out of a "hole" of crappy performance, especially where neither of the generations are good and (eg) DPOing on either wouldn't help much.))
		- ((I assume they re-prompt with the critique and then SFT/DPO/etc on the response?))
SFT Data
- Finetuning data is comprised of:
	1. Prompts from human annotation collection with [[Rejection Sampling]] responses
	2. Synthetic data targeting specific capabilities
	3. Small amounts of human-curated data
- Rejection sampling
	- For each prompt collected during human annotation, we sample ==10-30 outputs== from the latest chat model policy and use our reward model to select the best candidate. We use system prompts to steer RS responses to conform with desirable tone/style/formatting for different capabilities.
	- We adopt [[PagedAttention]] to increase the efficiency of rejection sampling... it enhances memory efficiency through dynamic KV cache allocation, supporting arbitrary output lengths by dynamically scheduling requests based on current capacity. ==Leads to a throughput improvement of over 2x during rejection sampling.==
		- ((?? I don't really understand this))
- Overall data composition
	- SFT and preference data have overlapping domains, but they're curated differently.
	  In each round of post-training, ==we adjust our overall data mix carefully along the axes of topic, complexity, quality of data samples.==
Data Processing and Quality Control
- Given that most of our training data is model-generated, it requires careful cleaning and quality control.
- Data Cleaning
	- In early rounds, we observe a number of undesirable patterns common in our data, like ==excessive use of emojis, exclamation points, or overly-apologetic tonal issues==, so we used a series of rule-based data removal/modification strategies.
- Data Pruning
	- Apply a collection of model-based techniques to remove low-quality training samples and improve overall model performance.
		- Topic classification using LLaMA 8b finetuned into a topic classifier (both coarse-grained "math reasoning" and fine-grained "geometry and trigonometry").
- Quality Scoring
	- Use the RM and LLaMA-based signals to obtain quality scores for each generation, and retaining 
			- English data (Accuracy, Instruction Following, Tone/Presentation) and coding data (Bug identification, User intention) are scored with different rubrics
			  The RM and LLaMA-based scores have ==high disagreement rates==, and we find that combining the signals yielded the best recall on test set.
- Difficulty scoring
	- We score data using two measures of difficulty ([[Instag]] and LLaMA-based scoring.)
		- For Instag, we prompt LLaMA 3 70B to perform intention-tagging of SFT prompts, where more intentions implies more complexity.
		- For LLaMa-based, we prompt it to measure the difficulty of dialogs on a three-point scale.
- ==Semantic duplication==
	- We cluster complete dialogues using [[RoBERTa]] and then within each cluster sort by quality score/difficult score. We then greedily select by iterating through all sorted examples, only keeping the ones that have maximum cosine similarity below some threshold.

### Capabilities
Code
- They improve coding capabilities via training a code expert, generating synthetic data for SFT, improving formatting with system prompt steering, and creating quality filters to remove bad samples from our training data.
- Code expert training
	- We train a code expert to help us collect high-quality human annotations for code... They branch the main pretraining run and do [[Continued Pretraining]] on a 1T token mix of 85% code data... for the last several thousands steps performing long-context finetuning (LCFT) to extend expert context length to 16K tokens on a high-quality mix of repo-level code data... and follow similar post-training modeling recipes to align the model with SFT/DPO data mixes primarily targeting code. This model is also used for [[Rejection Sampling]] of coding prompts.
- Synthetic data generation
	- During development, we identified key issues/problems in code generation... used LLaMA 3 and the code expert to generate 2.7M synthetic SFT dialogues.
	- Authors introduce ==execution feedback== as the primary source of truth.
	1. Process:
		1. Problem description generation: We generate a large collection of programming problem descriptions spanning diverse ranges of topics, sampling random code snippets from various sources and prompting the model to generate programming problems inspired by these examples.
			- This is the [[OSS-Instruct]] technique from the [[Magicoder]] paper
		2. Solution generation: We prompt LLaMA 3 to solve each problem, and add general rules of good programming to the prompt, and ask it to explain its thought process. ([[Chain of Thought|CoT]])
		3. Correctness analysis: After generating a solution, it's crucial to recognize its correctness isn't guaranteed -- we extract the source code from the solution and apply a combination of static and dynamic analysis techniques to approximate correctness (but not guarantee it):
			1. ==Static==: We run code through a linter and parser to ensure syntactic correctness, catching syntax errors, style issues, typing errors
			2. ==Dynamic==: We prompt the model to generate unit tests, executed in a containerized environment together with the solution, catching run-time execution errors.
			- ((Unclear if they do 1 before 2 or if they they do them both in parallel))
			- ==If solution fails at any step, we prompt the model to revise==
		- The finetuning process is conducted over multiple rounds, with each round building on the previous one. After each round, the model is improved, and able to generate higher quality synthetic data for the next round.
	2. Programming language translation
		- There's a performance gap between major programming languages (Python, C++) and less common ones (PHP, Typescript).
		- ==We translate data from common programming languages to less common languages, using LLaMA 3 and ensuring quality by syntax parsing, compilation, and execution.==
	3. To improve certain coding capabilities (documentation, explanations) where execution feedback is less informative, we employ a multistep approach:
		- Generate 1.2M synthetic dialogs related to code explanation/generation/documentation/debugging.
		1. Generate: Prompt LLaMA 3 to generate data that represents our target capability.
		2. [[Back-Translation]]: Prompt the model to backtranslate the synthetically generated data to the original code. (eg generate code from documentation, or generate code from documentation/explanation)
		3. Filter: Use the original code as a reference, and prompt L3 to determine the quality of the output (how faithful is the backtranslation)? We use the best in SFT.
- System prompt steering during rejection sampling: We use code-specific system prompts to improve code readability, documentation, thoroughness, and specificity.
- Filtering training data with execution and model-as-judge signals: We occasionally encounter quality issues in rejection-sampled data... it's hard to detect these, because the rejection-sampled responses contain a mix of natural language and code, and the code isn't always expected to be executable on its own. So we use a [[LLM-as-a-Judge]] approach, where L3 assign a binary score (0/1) based on code correctness and code style, keeping samples that achieve a perfect score of 2.
	- They realized this led to a regression in benchmark perf, since it disproportionately removed difficult prompts, so they revised responses until they met the Judge's quality.

Multilinguality
- Like code, we train an *==multilingual expert==* specialized on substantially more multilingual data, sourcing and generating high-quality multilingual instruction tuning data for German, French, Italian, Portuguese, Hindi, Spanish, and Thai.
	- We branch off the pretraining run and continue to pretrain on a data mix consisting of 90% multilingual tokens. This expert is used to collect higher-quality annotations on non-English languages until pretraining was fully complete.
- Mixture: 2.4% human annotations, 44.2% data from other NLP tasks, 18.8% rejection sampled data, and 34.6% translated reasoning data.
	- Human annotations: High-quality manually-annotated data from linguists and native speakers.
	- Other NLP tasks: We use multilingual training data from other tasks and rewrite it into dialog format. We also use parallel texts from GlobalVoices and Wikimedia.
	- [[Rejection Sampling]] data: We apply RS on our human-annotated prompts to generate high-quality samples for finetuning, with specialized system prompts and different levels of temperature. Prior to reward-model based selection, they do multilingual-specific checks to ensure a language match between prompt and response.
	- Translated data: We try to avoid using MT data to finetune the model, to prevent *==translationese==*... and we want to prevent the model from being exposed only to tasks rooted in an English cultural context... but we made ==one exception== to this for the synthetic quantitative ==reasoning data== to improve performance in qualitative reasoning in non-English languages.
		- Due to the simple nature of the language in these math problems, translated samples were found to have little to no quality issues.

### Math Reasoning
- We define the ability to perform multi-step computations to arrive at the correct final answer. There were several challenges that guide our approach:
	1. Lack of prompts (as complexity of question increases)
		- We source relevant pre-training data from math contexts and convert it into a question-answer format which can then then be used for SFT.
		- We identify math skills where the model underperforms and actively source prompts from humans to teach such skills.
	2. Lack of ground-truth CoT
		- We use LLAMA 3 to generate step-by-step solutions for a set of prompts... we filter to correct answer generations and do self-verification using L3.
	3. ==Incorrect intermediate steps when using model-generated CoT.==
		- We train [[Reward Model|Outcome Reward Model]]s and [[Process Reward Model]] ("stepwise") to filter training data where intermediate reasoning steps were incorrect.
		  - For more challenging prompts, we use [[Monte-Carlo Tree Search|MCTS]] with [[Process Reward Model|PRM]]s to generate valid to generate valid reasoning traces, further enhancing collection of high-quality reasoning data.
	4. Teaching models to use external tools (eg code interpreters), which can improve problem-solving abilities.
	5. ==Discrepancy between training and inference== (During inference, the finetuned model may interact with humans or other models, requiring it to improve its reasoning using feedback. Ensuring consistency between training and real-world usage is crucial for maintaining reasoning performance.)
		- We utilize incorrect generations (with incorrect reasoning traces) and perform error correction by prompting LLaMA 3 to yield correct generations.
		- The iterative process of using feedback from incorrect attempts and correcting them helps improve the model's ability to reason accurately and learn from its mistakes.

Long Context
- ==Naively applying our SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training, highlighting the need to incorporate long-context data in our SFT data mix.==
- We predominantly rely on synthetic data to fill the gap... We use earlier versions of LLaMA 3 to generate synthetic data based on the ==key long context use-cases: question answering, summarization for long documents, and reasoning over code repositories.==
	- QA: We curate a set of long documents from the pretraining mix, and split into 8K chunks. We then generate QA pairs conditioned on randomly selected chunk (pairs?) During training, we then use the whole doc.
	- Summarization: We applied hierarchical summarization of long-context documents by first summarizing 8K chunks, then summarizing the summaries. During training, we then use the whole document.
	- Long context code reasoning: We parse Python files to identify `import` statements and determine their dependencies. We select the most commonly depended-upon files, specifically those referenced from at least five other files. We remove one of these key files from a repository and prompt the model to identify which files depended on the missing file, and to generate the missing code.
- We then further categorize these examples based on sequence length (16K, 32K, 64K, 128K) to enable more fine-grained targeting of input lengths.
	- Authors mix ==only 0.1% of synthetically-generated long-context data with the original short-context data== to optimize performance across both short and long-context benchmarks.
		- ((Recall that this is just to maintain the long-context performance that was engendered during pretraining!))
- Authors note that only using ==short context training data in [[Direct Preference Optimization|DPO]] didn't seem to negatively impact long-context ability==, given that the SFT model was good at long-context tasks.


Tool use
- LLaMA 3 is trained to interact with the following tools:
	1. ==Search Engine==: L3 is trained to use the ==Brave Search== API to answer questions about recent events that go beyond its knowledge cutoff or that require retrieval.
	2. ==Python Interpreter==: L3 can generate and execute code to perform complex computations, read files uploaded by the user, and solve tasks like QA, summarization, data analysis, visualization.
	3. ==Math Computation Engine==: Can use the ==Wolfram Alpha== API to more accurately solve math, science problems, or retrieve accurate information from Wolfram's database.
- We also improve L3's ==zero-shot tool use capabilities== (given in-context, potentially unseen tool definitions and a user query), training the model to generate correct tool calls.
	- Tools can be implemented as Python functions with descriptions, demonstrations, and the models only needs the function signature and docstring as context to generate the appropriate call.
	- We also convert function definitions and calls to JSON format (eg for web API calls).
- Different from [[Toolformer]], we rely on human annotations and preferences to teach L3 to use tools.
	- For tools, dialogs often contain more than a single assistant message (eg call the tool and then reason about its output), so we annotate at the message level to collect granular feedback; Annotators provide a preference between two assistant messages with the same context, or, if both contain major problems, edit one of the messages.
		- We use ==human preference tuning== (with possible editing of agent's reasoning if both are bad) to help the assistant learn to both call tools and reason about tool outputs.
	- ==We accelerate the annotations process by bootstrapping basic tool use capabilities by finetuning on synthetically generated data from previous L3 checkpoints. ==
	- L3 improves gradually through iterative development; we progressively complexify our human annotation protocols, start with single turn before moving to tool use in dialogs, and finally multi-step tool use and data analysis.
- Tool datasets
	- Single-step tool use: ==We do synthetic generation of user prompts requiring a call to a core tool. Relying on few-shot generation, we generate appropriate tool calls, execute them, and add the output to model context. We then prompt the model again to generate a final answer.==
	- Multi-step tool use: ==We follow a similar protocol and prompt L3 to generate user prompts that require at least two tool calls that can be the same or different tools from our core set.==
		- ==We few-shot prompt L3 to generate solutions consisting of interleaved reasoning steps and tool calls, similar to [[ReAct]].==
	- File uploads: We annotate for .txt, .docx, .pdf, .pptx, .xlsx, .csv, .tsv, .pv, .json, .jsonl, .html, .xml. Our prompts are based on a provided file, and ask to summarize the contexts of the file, find and fix bugs, optimize a piece of code, perform data analysis and/or visualization.
- We also include ==challenging situations==:
	- Multi-turn interactions
	- More than three-step tool use
	- Instances where a tool call doesn't give a satisfying answer.
	- We also train the model to avoid calling tools for simple queries by adding queries from easy math or QA datasets and their responses without tools, ==with tools activated in the system prompt!==
- To improve L3 zero-shot tool use, we finetune on a large and diverse set of partly synthetic tuples (function definitions, user query, corresponding call).
	- We mine [[The Stack]] to ground our synthetic user queries in real functions, extracting function calls and their definitions, cleaning and filtering them, and using L3 to generate a natural language query corresponding to the function call.
	- For multi-turn function calling, we generate synthetic data for multi-turn dialogs with function calls. We use multiple agents to generate domains, APIs, user queries, API calls, and responses.

Factuality/[[Hallucination]] mitigation
- We follow the principle that post-training should "align the model to know what it knows," rather than add additional knowledge.
- We develop a knowledge probing technique that takes advantage of L3's in-context abilities:
	1. ==Extract a data snippet== from pretraining data
	2. ==Generate a factual question== about the snippet with L3
	3. ==Sample responses== (plural) from L3
	4. ==Score *correctness*== of generations using the original context and L3 as a judge
	5. ==Score *informativeness*== of the generations using L3 as a judge
	6. ==Generate a refusal== for responses which are consistently informative and *incorrect* across generations (these are ones that seem to be deceptive to users) using L3.


[[Steerability]]
- The ability to direct model's actions and outcomes to meet developer/user specifications -- important for generic foundation models.
- We use a system prompt with natural language instructions, especially about response length, format, tone, and character/persona.
	- We collect steerability preference samples within the general English category by asking annotators to design different system prompts for L3, and then engage in conversations with models to evaluate consistency in following instructions defined in system prompts over the course of the conversation.
	- After we collect preference data, we leverage it in reward modeling, rejection sampling, SFT, and DPO to enhance steerability.

# Results
- We investigate:
	1. Pre-trained LM
	2. Post-trained LM
	3. Safety characteristics of L3

## Pre-trained LM
- Evaluations cover 8 top-level categories (see Table 8):
	1. Commonsense reasoning (eg [[SQuAD v2]])
	2. Knowledge eg ([[HumanEval]], [[MBPP]])
	3. Reading comprehension (eg [[CommonSenseQA]], [[Winogrande]])
	4. Math, reasoning, problem solving (eg [[GSM8K]], [[MATH]])
	5. Long-context
	6. Code
	7. Adversarial evaluations
	8. Aggregate evaluations (eg [[MMLU]])
- LLaMA 3 8B outperforms competing models in virtually every category.
- LLaMA 3 70B outperforms [[Mixtral 8x22B]] consistently.
- LLaMA 3 405B performs competitively with other models in its class.
- Models are also evaluated the robustness of our pretrained models to design choices in multiple-choice question setups (eg answer ordering, prompt formats). Results show that models are "very robust" to changes in MCQ labels and to structure of few-shot prompt labels.
- We also evaluate on several adversarial benchmarks in QA, Math reasoning, and paraphrase detection... ours performed well against paraphrase detection, but performed substantially lower in the adversarial setting for math and question answering.
- We conduct a contamination analysis to estimate the extent to which benchmark scores might be influenced by contamination of evaluation data in the pre-training corpus.

## Post-trained LM
- We evaluate over:
	1. General (eg [[MMLU]], [[IFEval]])
	2. Math and reasoning (eg [[GSM8K]], [[MATH]], [[GPQA]])
	3. Code (eg [[HumanEval]], [[MBPP]], [[MBPP+]], [[HumanEval+]])
	4. Multilinguality (eg MGSM, internal multilingual MMLU)
	5. Tool-use (eg [[Berkeley Function-Calling Leaderboard|BFCL]]])
	6. Long context (eg [[Needle in a Haystack|NIAH]], InfiniteBench)
Also evaluate models on a wide variety of proficiency exams originally designed to test humans:
- GRE
- LSAT
- SAT
- AP
- GMAT

## Safety characteristics
- Blah blah blah
- False refusal rates, violation rates, CBRN testing, read teaming

Authors develop and release [[LLaMA Guard 3]] classifier, which is a L38B model finetuned for safety classification
- Used to detect whether input prompts and/or output responses violate safety policies on specific categories of harm.  It's designed to support LLaMA's growing capabilities, and can be used for English and multimodal text.
- ==Train on the 13 categories listed in the "AI Safety Taxonomy" (Vidgen et al, 2024)==
	1. Child Sexual Exploitation
	2. Defamation
	3. Elections
	4. Hate
	5. Indiscriminate Weapons
	6. Intellectual Property
	7. Non-Violent Crimes
	8. Privacy
	9. Sex-related Criems
	10. sExual Content
	11. Specialized Advice
	12. Suicide and Self-Harm
	13. Violent Crimes
- Authors also train to support tool-calls use cases and prevent code interpreter abuses.
- Training data starts with the English data from [[LLaMA Guard]] and then expands it to incorporate new capabilities (multilinguality, tool use).
	- We do extensive cleaning of collected samples using human and LLM annotation (using L3).

Authors also introduce and open-source two prompt-based filtering mechanisms:
- [[Prompt Guard]]: Model-based filter designed to detect prompt attacks, which are input strings designed to subvert intended behavior of an LLM.
	- Detects direct jailbreaks and indirect prompt injections.
	- Model finetuned from mDeBERTa-v3-base, a small (86M) model suitable for filtering inputs into an LLM.
- [[Code Shield]]: Provides inference-time filtering, focusing on detecting the generation of insecure code before it might enter a downstream use case like a production system.
	- Does so by leveraging a static analysis library (Insecure Code Detector, ICD) to identify insecure code.
==These guardrails are generally useful for developers, who can deploy multi-layered protections in various applications.==

# Inference
Authors investigate two main techniques to main LLaMA 3 405B model efficient:
1. Pipeline parallelism
2. FP8 quantization
	- Turns out that with some care and kernel engineering they can do FP8 quantization with negligible impact on the quality of model responses, according to our reward models.




# Vision Experiments


# Speech Experiments


# Related Work


# Conclusion





Abstract
> Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called ==Llama 3==. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with ==405B parameters== and a ==context window of up to 128K tokens==. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including ==pre-trained and post-trained versions== of the 405B parameter language model and our [[LLaMA Guard 3 ]]model for input and output safety. The paper also presents the results of ==experiments in which we integrate== ==image==, ==video==, and ==speech== capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development.

# Paper Figures
![[Pasted image 20240724202317.png|600]]
Comparison of the LLaMA 3.1 suite of models with competing frontier models, including [[Gemma 2]], [[Mistral 7B]], [[Mixtral 8x22B]], [[GPT-3.5]] Turbo, [[Nemotron-4]] 340B, [[GPT-4]], [[GPT-4o]], [[Claude 3.5]] Sonnet. See that the LLaMA 3.1 models are absolutely competitive in the 405B scale, and actually seem to ==*wipe the floor* with alternatives in the 8B and 70B categories==.
- Note that results are obtained with 5-shot prompting and no CoT on some of the usual  benchmarks.

![[Pasted image 20240725001928.png|500]]

![[Pasted image 20240725003251.png|500]]
Scaling laws that the Meta folks develop for their models.

![[Pasted image 20240725005849.png]]
Their 4D Parallelism, combining [[Tensor Parallelism]], [[Context Parallelism]], [[Data Parallelism]] ([[Fully Sharded Data Parallelism|FSDP]]), and [[Pipeline Parallelism]].

![[Pasted image 20240725105458.png|500]]
Pipeline parallelism

![[Pasted image 20240725123709.png|500]]
Despite the mixture of the pre-training data having like... 50% or so split between coding/reasoning/multilingual, it seems here that the majority (82%) of human preference data is just in the "general english" category (which might subsume).
- Interesting that the average number of turns per dialog is 4.1 (where a turn is a user-agent loop)

![[Pasted image 20240725133627.png]]
Ah, but the SFT mixture is pretty similar to the pretraining data mixture. Note how long the average turn length is, 6.3!

![[Pasted image 20240725151536.png|500]]
![[Pasted image 20240725153021.png]]

![[Pasted image 20240725193113.png|500]]
So that given a user prompt, the agent determines an initial plan of action, and then works through it, doing tool use and reasoning after each tool use (similar to [[ReAct]]).

![[Pasted image 20240725195701.png|600]]
Processing file uploads. Given a file path, assistant using code interpreted to load a pd.dataframe from the filepath, and plot some aspects of the data. Interesting that in this case there's no reflection/reasoning between the first and second tool use. 

![[Pasted image 20240725204308.png|700]]
The pre-training benchmarks, organized by category. See some classices like [[SQuAD v2]], [[MBPP]], [[HumanEval]], [[CommonSenseQA]], [[Winogrande]], [[GSM8K]], [[MATH]], [[ARC Challenge]], [[MMLU]], [[MMLU-Pro]], [[AGIEval]], [[BIG-Bench Hard]]

![[Pasted image 20240725210907.png|600]]

![[Pasted image 20240725210948.png]]
Evaluation of L38B and L370B on standard benchmarks

![[Pasted image 20240725212340.png]]
![[Pasted image 20240725212342.png]]
![[Pasted image 20240725212344.png]]
![[Pasted image 20240725212348.png]]
![[Pasted image 20240725212350.png]]
![[Pasted image 20240725212352.png]]

![[Pasted image 20240725212646.png]]
The evaluations used for post-training evaluation

![[Pasted image 20240725212804.png|500]]
Evaluation of L3 post-training on a variety of human proficiency exams

![[Pasted image 20240725213027.png]]
![[Pasted image 20240725213037.png]]
![[Pasted image 20240725213058.png|500]]
![[Pasted image 20240725213258.png]]
![[Pasted image 20240725213306.png]]
![[Pasted image 20240725213319.png]]
It's sort of interesting when you see something like this. Like... when the L3 team saw GPT4o outcompeting L3405B on file upload-related tasks, did they just say "ok"?
![[Pasted image 20240725213725.png]]
![[Pasted image 20240725213826.png]]

![[Pasted image 20240725215152.png]]
Impact of [[LLaMA Guard 3]] on violate rate and false refusal rate

![[Pasted image 20240725215753.png]]
![[Pasted image 20240725215800.png]]
See that [[LLaMA Guard 3]] can be quantized down to [[INT8]] without losing much performance.

![[Pasted image 20240725220555.png|500]]
On quantization


# Non-Paper Figures
![[Pasted image 20240723235006.png]]

