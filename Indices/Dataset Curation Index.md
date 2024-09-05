Resources:
- [BLOG: HuggingFace RefinedWeb Creation](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

1. It's common to train a model on a given corpus considered "clean" (eg Wikipedia) and use it to check the perplexity on the dataset that we're trying to curate.
	- This doesn't always correlate with improved performance on downstream tasks of interest.
2. Another method is to train small (cheap, fast) models on a representative subset of our dataset, and evaluate them on a set of evaluation tasks set of evaluation tasks.
	- It's important to choose diverse and representative evaluation tasks, and try not to overfit to any individual benchmark.
3. Another way to compare different datasets is to train a model on each dataset and have humans rate/compare the generations of the models, eg on [[ChatBot Arena]]. 
	- This might provide the most reliable results, but it's expensive and slow, and simple pretrained (not-instruction-tuned) models aren't yet prepared to be assistants and might be very sensitive to prompt details.

An example of the pipeline used to create a dataset ([[HuggingFace]]'s [[FineWeb]]):
![[Pasted image 20240605132951.png|300]]
- Text Extraction
	- If using [[Common Crawl]] data, some often use their WET format, which has the text pre-extracted, but if you want better extraction, you should do your own (eg using a library like [[Trafilatura]]) on their raw WARC format.
- Base Filtering
	- Filtering is an important part of the curation process, removing parts of the data that lowers the performance of the model, thus deemed to be "lower quality" in our eval-driven process of dataset crafting.
	- You might:
		- Apply URL filtering using a blocklist to remove adult content
		- Apply a [[fastText]] language classifier to keep (eg) English-only text with a score of >= 0.65.
		- Applied quality and repetition filters from [[MassiveText]] (using default thresholds).
- Deduplicating the data
	- Deduplication is one of the most important steps! The web has many aggregators/mirrors/templated pages, or otherwise just repeated content. Removing these duplicates has been correlated with improvements in model performance.
	- There are different ways to identify/define duplicate data. Common approaches rely on any of:
		- Hashing techniques (eg [[MinHash]]) to speed up the process
			- In FineWeb, they collect each document's 5-grams and compute MinHashes using 112 hash functions, split into 14 buckets of 8 hashes each, targeting documents that are at least 75% similar.
		- Building efficient datastructures (eg suffix arrays) to index the data
		- Fuzzy methods, using some similarity metric to mark documents as duplicates
		- "Exact" methods, checking for exact matches between two documents (or lines, paragraphs, or other granularity-level).
- Quality Filtering
	- Include things like dropping lines not ending in punctuation marks, those containing "lorem ipsum" or curly brackets, etc.
	- [[C4]] has some interesting quality filters.
	- FineWeb's blog post has some interesting ways of creating heuristic filters using statistical techniques, ending up with things like "Remove documents where the fraction of lines ending with punctuation <= 0.12".


