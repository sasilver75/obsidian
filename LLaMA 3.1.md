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
		- Self-supervised learning, obtaining large amounts of knowledge about the world. Teh 405B is pretrained on 15.6T tokens using a context window of 8K tokens, which is then followed by a [[Continued Pretraining]] stage that increases the supported context window to 128K tokens.
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
	- ((This doesn't seem to be their most granular taxonomy though, if they're downsampling arts and entertainment. And ))

## Model Architecture


## Infrastructure, Scaling, and Efficiency


## Training Recipe



# Post-Training


# Results


# Inference


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

# Non-Paper Figures
![[Pasted image 20240723235006.png]]