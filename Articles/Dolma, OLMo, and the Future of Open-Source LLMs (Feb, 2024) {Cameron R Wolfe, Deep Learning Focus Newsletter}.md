#article 
Link: https://cameronrwolfe.substack.com/p/dolma-olmo-and-the-future-of-open?utm_source=post-email-title&publication_id=1092659&post_id=141461162&utm_campaign=email-post-title&isFreemail=true&r=764e6&utm_medium=email

-------
![[Pasted image 20240220210052.png]]

The increasing popularity of open-source LLMs is widely considered to be beneficial, but there's one problem: ==the definition of open-source varies drastically between models.== 
- In many cases, open-source LLMs simply ==release their model parameter and inference code, but OBFUSCATE important details like the contents of the pretraining dataset, training hyperparameters, evaluation strategy, training logs, checkpoints, and more.==

Here, we will overview the recently-released [[Dolma]] dataset and [[OLMo]] suite of LLMs, from [[Allen Institute|AI2]]

The process of constructing the Dolma pretraining corpus, released under the AI2 impact licenses, is fully-documented -- authors even release code and tooling for re-building the dataset from scratch!

Similarly, the model itself is released under an Apache-2.0 license along with all training data, code, checkpoints, training logs, and more.

Dolma and OLMo take massive steps toward demystifying the LLM preparing process by releasing all relevant resources and analyzing any and all choices made in the process of training OLMo.


# Dolma 🍇🍃 : An Open Corpus of 3T Tokens for LLM Pretraining Research

> The most powerful LMs are built by a few organizations who withhold most model development details; The composition of LM pretraining data is often only vaguely stated, even in cases where the model itself is released for public use.

![[Pasted image 20240220211043.png]]
Above: The Dolma corpus
- [[CommonCrawl]]: A free, open repository of web crawl data, with over 250B pages... 3-5B pages added per month.
- [[The Stack]]: A 3TB dataset of permissively licensed source code in 30 programming languages from [[Eleuther]].
- Reddit: You know what it is-
- [[C4]]: A *Colossal, Cleaned Common Crawl* -- it's basically a subset of the good parts of Common Crawl.
- PeS2o: A collection of 40M creative open-access academic papers, derived from the Semantic Scholar Open Research Corpus, or S2ORC -- these are both from AI2. 
- [[Project Gutenberg]]: An online library of free eBooks.

The ==process of constructing this composite dataset, including design principles and the empirical reasoning behind each decision, is documented in the paper==.
- Additionally, the authors even ==released a full toolkit for efficiently reproducing and modifying resulting dataset!==

Dolma design principles
- The end goal of Dolma is to:
	1. Create the largest open (data and process) pretraining corpus
	2. Use this pretraining corpus to train an open language model (weights, code, checkmarks, process, logs), called [[OLMo]]

Authors aimed to pull from data sources that are commonly used in prior work, and to process the data that aligns with existing best practices. In areas where best practices are unknown, empirical practices are adopted, whereby smaller variants of OLMo are trained and evaluated to determine the downstream impact of changes to the pretraining dataset.

The impact of data size on model performance was investigated in [[Chinchilla]], where authors showed that larger models must be pretrained over sufficiently large pretraining datasets to truly optimize their performance.
- But even smaller models like LLaMA-2 models haven't yet fully converged after pretraining over 2T tokens!
- As a result, authors aimed to make Dolma sufficiently large (ie 3T tokens) to facilitate further studies on the pretraining dataset size.


# Prior Open Pretraining Datasets
- Dolma isn't the only effort that has tried to shed light on the process of constructing pretraining data for LLMs.
- [[C4]]: Colossal Clean Crawled Corpus
	- This pretraining dataset was originally constructed for the [[T5]] model; ==This data is sourced from Common Crawl==, and the full dataset contains about 750GB of text in total. Authors chose to create the C4 dataset from scratch due to the lack of high-quality, publicly-available pretraining datasets for language models.
	- ==Many models (Gopher, Chinchilla, MPT, and more) use C4 as a subset of their pretraining data, since the quality is quite high.==
- [[ROOTS]]: Responsible, Open-Science Open-Collaboration Text Sources
	- Developed as part of the BigScience research initiative for training [[BLOOM]]
	- A ==multilingual dataset== that spans over 46 natural languages and 13 programming languages. The ==English portion of this dataset is limited in scale -- too small for training English-only LLMs!==
	- Comprised of over 1.6TB of text in total.
- [[RefinedWeb]]
	- The pretraining corpus that was curated to train the [[Falcon]] LLMs.
	- Corpus is ==comprised of web data that undergoes a purely-heuristic-based filtering pipeline== (ie no *model-based* filtering), and ends up with a high-quality pretraining dataset. 
	- Large 5T token corpus, but ==only a small portion of the data (600B) was made available to the public.==
- [[RedPajama]]
	- An initiative led by [[TogetherAI]] to produce a leading open-source LLM and release it to the public with a more permissive license.
	- As a first step, the authors recreated the pretraining dataset for LLaMA resulting in the RedPajama v1 and v2 datsets
	- Similar to Dolma in nature, but RedPajama has a more limited scope; ==the creators of RedPajama are trying to reproduce LLaMA, whereas Dolma attempts ot provide a transparent resource that empowers others to study all notable aspects of LLM pretraining==, and pulls from a variety of additional sources (eg scientific papers, code, conversational forums, more).


# Creating the Dolma Corpus
- ==The process of constructing a pretraining dataset for an LLM consists of three primary components:==
	1. ==Acquisition==: obtaining content from a variety of sources
	2. ==Cleanup==: Using heuristics and model-based techniques to filter the data
	3. ==Mixing==: Deduplication and up/down-sampling of data sources

![[Pasted image 20240220233435.png]]
Above: Some of the data cleaning pipelines applied to various types of data in the Dolma corpus.

To process data for pretraining, we rely on ==four primary transformations of our data==:
1. ==Langauge filtering==
2. ==Quality filtering==
3. ==Content filtering==
4. ==Deduplication==

------
#### Aside: What is fastText?
- ==fastText== is a free and open-source library that can be used to learn text representations and train lightweight text classifiers.
- fastText classifiers have simple architectures that take an average of word representations in a sentence, and then pass the result to a linear classifier. They perform well -- similar to deep-learning based classifiers, but are cheaper. They can be improved my moving to an n-gram representation of our data, rather than the default unigram word representations.
- fastText is used in the cleaning pipelines in Dolma, and in other large-scale data processing applications.

Note: The ==CCNet== data pipeline is a popular reference architecture for creating high-quality, deduplicated, and monolingual text datasets from web data. It uses a fastText classifier for language identification.

-------








































