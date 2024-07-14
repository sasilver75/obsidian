---
aliases:
  - DCLM
---
June 17, 2024
59 authors from many universities and research groups, including [[Ludwig Schmidt]]
[DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794v1)
Website link: [DataComp-LM](https://www.datacomp.ai/dclm/index.html#home)
#zotero 
Takeaway: A benchmark/competition introduced in 2024 focused on data-centric AI. Whereas historically we treat datasets as constant and compete on model architectures (eg [[ImageNet|ILSVRC]]), DataComp is a competition in which architecture and training code is held constant, and competitors are encouraged to find ways of filtering and augmenting a dataset of 240T tokens from [[Common Crawl]] (or bringing their own data) called ==DCLM-Pool==. The benchmark consists of multiple scales, with various candidate pool sizes and associated compute budgets ranging from 412M to 7B parameters. As point of proof, authors produce a dataset called ==DCLM-Baseline==, which they use to train a model with a SoTA trade-off between compute and performance.

---

## Introduction
- Researchers are increasingly interesting in looking at language models from a data-centric view, asking how they can filter or improve datasets to optimize LM performance.
- Authors release ==DCLM-Pool==, a corpus of 240T trillion tokens derived from Common Crawl, with participants aiming to curate the bets possible training set out of DCLM-Pool.
- Authors do 416 baseline experiments with different training sets and compute scales. Model-based filtering is a key component of data curation -- Interestingly, a simple bigram classifier carefully trained on a set of positive and negative performs best among the classifiers they try.
	- Finally, we combine our results into ==DCLM-Baseline,== a new SoTA public training set for LMs. 

## Related Work
- Most data curation efforts focus on improving model performance, including filtering by language, heuristic-based filtering, quality filtering, data deduplication, and mixing. We conduct the largest public investigation of data curation, resulting in the strong DCLM-Baseline dataset.
- As the scale of LMs has increased over the last few ears, the community has curated larger datasets to match, with [[C4]] having 160B and [[The Pile]] having 300B, and [[RefinedWeb]] having 600B, [[Dolma]] having 3T, [[FineWeb]] 15T, and [[RedPajama v2]] at 30T tokens.
	- There are also large domain-specific datasets like The Stack V2, and high-quality filtered datasets like FineWeb-Edu with 1.3T tokens.
- With an initial 200T token pool and 7B models, DCLM is the first large-scale data-centric benchmark for language models.

## DataComp for language models (DCLM) Benchmark
- ==DCLM-Pool== is an unfiltered web-text corpus comprised of all Common Crawl data prior to 2023, composed by re-extracting text from HTML using `resiliparse` instead of using Common Crawl's pre-extracted text (which is known to be suboptimal).
	- Contains 200B documents, resulting in 240T GPT-NeoX tokens.
	- Instead of decontaminating DCLM-Pool directly, they release decontamination tooling to help participants examine their datasets for overlap with test sets.
- There are different competition scales to spanning three orders of magnitude, which each specifying a number of model parameters (eg 7B) and a *Chinchilla multiplier* (eg 1x), where the number of training tokens for each scale is 20 x \#params x Chinchilla multiplier so that a multiplier of 1x corresponds to a Chinchilla-optimal allocation.
- After choosing a scale, participants choose one of two tracks 
	- Filtering Track: Participants propose algorithms to select training data from a candidate pool.
	- Mixing Track: A submission combines documents from potentially many sources (eg from DCLM-Pool, a custom crawl, StackOverflow, or Wikipedia).
- At each scale, the training recipe is fixed.
- The full evaluation suite (based on Mosaic's LLM-Foundry) contains 53 downstream tasks suitable for base model evaluation. To evaluate data-curation algorithms, we focus on three main performance metrics:
	1. MMLU 5-shot accuracy
	2. *"Core"* centered accuracy (Computed over a subset of 22 tasks providing a low-variance signal)
	3. *"Extended"* centered accuracy (Computed over all 53 tasks)

## Building High-Quality Datasets with DCLM
- We begin by evaluating several well-known datasets: [[RefinedWeb]], [[RedPajama]], [[Dolma]]-V1, and [[C4]]. All four datasets use various heuristic filters and data cleaning steps, but we find that RefinedWeb performs the best on the Core and Extended metrics.
- For text extraction, they compare `resiliparse`, [[Trafilatura]] (used by [[RefinedWeb]]), and the Common Crawl-provided WET files that contain pre-extracted text... They find that both `resiliparse` and `trafilatura` improve "Core" by at least 2.5 points over the WET extraction. This is significant because most open source datasets use the WET extraction, which could partially explain their worse performance.
	- While Trafilatura and [[Resiliparse]]  both have similar downstream performance, `resiliparse` is 8x faster to run, and hence more practical for large-scale processing.
- Deduplication is important for reducing memorization and improving data diversity. They explore [[MinHash]] as part of a "suffix array pipeline," and near-duplicate [[Bloom Filter]]ing. Both approaches provide comparable downstream performance, but the bloom filtering one scales better.
- Model-based quality filtering
	- ==We compare many strategies:==
		1. PageRank score filtering: Retain documents highly linked to other documents
		2. Semantic Deduplication (SemDedup): Remove documents with similar informational content
		3. Linear classifiers: Fit on pre-trained BGE text embeddings
		4. AskLLM: Prompts an LM to see if a document is helpful
		5. Perplexity filtering: We retain low-perplexity sequences following CCNet
		6. Top-k average logits: We average the top-k model logits over all words to score how confident a model is that the correct words are within k reasonable choices.
		7. fastText: Binary classifiers to distinguish data quality.
	- It seems like the fastText one works the best.
- Dataset mixing
	- Researchers often combine [[Common Crawl]] with datasources considered high-quality (eg Stack exchange, Wikipedia). Since DCLM participants in the mixing track can bring their own data, we examined benefits of adding high-quality sources... It usually improves performance, but it's interesting that for the DCLM-Baseline model/dataset, it actually decreases performance, which suggests it might be counterproductive given performant filtering.
- Decontamination
	- Authors attempt to detect and remove questions from MMLU that exist in DCLM-Baseline. Our strategy is to flag training documents that that contain the last sentence of a question from MMLU along with one of the corresponding options.

## Conclusion and Limitations
- There are many variations of DCLM-Baseline that they didn't explore, like understanding the impact of shaded deduplication in more detail... and many more ways of training filtering models, both in terms of their architecture and training data.
- Most of their experiments were conducted with only one tokenizer (GPT-NeoX), and other tokenizers might perform better on multilingual tasks or math.
- Combining DCLM Baseline with domain-specific training sets and extending DCLM to cover code and math are interesting avenues for future work.
- They've only trained 7B maximum parameter models ... and SoTA models are much larger; we're optimistic that these gains will extend to larger model scales, but future work still needs to test this experimentally.

Lots of appendices in this paper about their specific process for creating DCLN-Baseline.

Abstract
> We introduce ==DataComp for Language Models (DCLM)==, a testbed for controlled dataset experiments with the goal of improving language models. As part of DCLM, we ==provide a standardized corpus of 240T tokens extracted from Common Crawl, effective pretraining recipes based on the OpenLM framework, and a broad suite of 53 downstream evaluations==. ==Participants in the DCLM benchmark can experiment with data curation strategies such as deduplication, filtering, and data mixing at model scales ranging from 412M to 7B parameters==. As a baseline for DCLM, we conduct extensive experiments and find that model-based filtering is key to assembling a high-quality training set. The resulting dataset, DCLM-Baseline enables training a 7B parameter language model from scratch to 64% 5-shot accuracy on MMLU with 2.6T training tokens. Compared to MAP-Neo, the previous state-of-the-art in open-data language models, DCLM-Baseline represents a 6.6 percentage point improvement on MMLU while being trained with 40% less compute. Our baseline model is also comparable to Mistral-7B-v0.3 and Llama 3 8B on MMLU (63% & 66%), and performs similarly on an average of 53 natural language understanding tasks while being trained with 6.6x less compute than Llama 3 8B. ==Our results highlight the importance of dataset design for training language models and offer a starting point for further research on data curation.==


# Paper Figures

![[Pasted image 20240713215504.png|400]]
From the large CommonCrawl dump, the authors develop a filtered high-quality dataset called ==DCLM-Baseline==(in orange), which they use to train models that make a SoTA playoff between compute and performance.

![[Pasted image 20240713220007.png|450]]
The DCLM process, if you want to compete.

![[Pasted image 20240713221545.png|400]]
Five competition scales for the DCLM challenge

![[Pasted image 20240713222911.png|400]]
Construction of the DCLM-Baseline dataset from DCLM-Pool.

![[Pasted image 20240713230411.png|400]]
See that [[Trafilatura]] and [[Resiliparse]] both seem to perform pretty similarly.

![[Pasted image 20240713230445.png|400]]
Comparisons between different methods for quality-based filtering of data! It seems that training a [[fastText]] classifier performs best. (From "Bag of tricks for efficient text classification" paper, 2017)

![[Pasted image 20240713231750.png|400]]
It's interesting that mixing in additional high quality datasets (eg Wikipedia, Stack Exchange, arXiv) usually improves datasets, but it doesn't in the case of the example DCLM-Baseline dataset (which is the one they do a lot of filtering on, from the DCLM-Poool dataset).