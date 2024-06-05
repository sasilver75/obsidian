From [[HuggingFace]]
Github: [Link](https://github.com/huggingface/datatrove)

A library to process, filter, and deduplicate text data at a very large scale., enabling the scaling of filtering and deduplication setups to thousands of CPU cores.

Developed (?) by HF as part of the project to create datasets like HF's  [[RefinedWeb]], which exclusively used Datatrove.


- Provides a set of prebuilt commonly-used processing ==blocks== with a framework to easily add custom functionality. Each pipeline block takes a generator of `Document` as input and returns another generator of `Document`. The supplied block types are:
	- ==readers== read data from different formats, yielding `Documents`. Pipelines usually start with this block, which takes a path to a folder containing data to be read.
	- ==writers== save `Document`s to disk/cloud in different formats. Writers require you specify an output folder, as well as optional cmopression methods and filenames.
	- ==extractors== extract text content from raw formats (like webpage html). The most commonly-used extractor is [[Trafilatura]], which uses the popular `trafilatura` library.
	- ==filters== filter/remove some `Document`s based on specific rules/criteria
	- ==stats== collect statistics on the dataset
	- ==tokens== tokenize data or count tokens
	- ==dedup== help you deduplicate `Document`s. Examples include [[MinHash]] deduplication, sentence deduplication, exact substring deduplication, and more.
- You can also easily create Custom blocks
- Runs out of the box *locally* or on a small cluster (eg using SLURM).


![[Pasted image 20240415163137.png]]
