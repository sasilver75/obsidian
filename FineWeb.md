ddApril 25, 2024
[[HuggingFace]]
HuggingFace Dataset Card: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md)
HuggingFace Full Blogpost: [FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)

![[Pasted image 20240605000114.png|200]]


A large, ==15T token, 44TB disk space English dataset meant for LLM pre-training==, consisting of cleaned and deduplicated web data from 96 [[Common Crawl]] dumps (Data from Summer of 2013 to March of 2024) using [[Datatrove]], HuggingFace's large-scale data processing library. It produces better-performing LLMs than other open pretraining datasets.
- Note: Because of the filtering that's applied, ==it's likely that code content is not prevalent in our dataset== -- they recommend enriching the FineWeb data with information from [[The Stack v2]], as well as generally complementing FineWeb with specialized curated datasources (eg Wikipedia), since they will likely have better formatting than the content included in FineWeb.

FineWeb was ***originally*** meant to be a full open replication of the [[RefinedWeb]] dataset from the [[Falcon]] paper (Notable that Guilherme was previously on the TII UAE Falcon 40B team, responsible for the FineWeb dataset), ==but by carefully adding additional filtering steps, we managed to push the performance of FineWeb *well above* that of the original RefinedWeb==, and models trained on our dataset also outperform models trained on other commonly-used high-quality web datasets (like C4, Dolma-v1.6, The Pile, SlimPajama).

The authors note that there's still room for additional filtering, and that they intend to continue exploring how to improve the dataset quality in following versions of FineWeb.

The authors release all of the code needed to fully reproduce our processing setup using the `datatrove` library, as well as small ablation models trained using the `nanotron` library.

### The ==data processing pipeline== consists of:
1. URL Filtering: Removing documents originating from malicious and NSFW websites, using both block-list as well as subwords detection.
2. `Trafilatura` text extraction on the raw HTML from CommonCrawl's warc files (Web ARChive format). 
	- A Python package and command-line tool designed to gather text from the web, including discovery, extraction, and text-processing components. Main uses are web crawling, downloads, scraping, and extraction of texts.
3. `FastText` language filter: Removing any document with an `en` language score lower than .65.
4. Quality filtering
	1. Repetition and Quality filters from [[Gopher]]
	2. [[C4]] Quality filters (except the `terminal_punct` rule)
	3. FineWeb custom filters, consisting of heuristics for removing list-like documents, documents with repeated lines, and documents with likely-wrong formatting.
5. [[MinHash]] deduplication, with each crawl deduplicated independently (5-grams, 14x8hash functions)
6. PII Formatting to anonymize email and public IP addresses
	- For emails, we apply a regex pattern and replace email addresses with email@example.com or first.lastname@example.org. 
	- We employ a regex pattern to further filter/anonymize IP addresses, replacing with randomly generated IP addresses (which at the time of dataset creation were not responding toping requests).
	- We decided *against applying regex patterns to phone numbers* for phone numbers due to the high false positive rate.
	- It's still likely that some PIPI exists; HF has a PII removal form if someone wants their information removed.


> FineWeb is very different from even larger datasets like [[RedPajama v2]], which is double its size! Surprisingly, the size of the 15T tokens isn't very important -- what's more important is WHY we spent 120k GPU hours on an H100 cluster to prepare and share a dataset!
> Where can you get data at scale for web-scale LLM pretraining? Common Crawl. But can we train directly on the petabytes of CommonCrawl corpus? The BigScience/[[BLOOM]] training answered: No! You actually want a dataset which is both large AND high-quality! But what is high-quality for a web-scale LLM pretraining dataset?
> Unintuitive behavior: Between 2022 and 2023, the "LLM quality" of CommonCrawl dropped significantly; training an LLM on the crawls between 2022-2023 will give you *lower* performance on a set of evals. It turns out that commonCrawl has been filtering more strongly domains with adult content -- with an unintuitive result.
> So how do you know you have good quality data? The circular answer is: You... train on it, using models that are not too big to be expensive, but big enough to give you signal about the quality of a larger model trained on the same data. These are what they call "ablation models" in FineWeb.
> Which ablation models did they use? We used both 1.8B models trained on 28B tokens, and 1.8B models trained on 350B tokens.
> This is the main difference between FineWeb and datasets like CommonCrawl/RedPajama-V2 -- with the latter cases, you still need to do the work of selecting how to filter the data yourself! THIS is the work we wanted to provide the community with, in FineWeb.
> From Thomas Wolf (CSO@HF): https://twitter.com/Thom_Wolf/status/1782691683517305226


# Paper Figures


# Non-Paper Figures
![[Pasted image 20240605132602.png]]
