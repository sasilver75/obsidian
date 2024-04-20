October 30, 2023  (six months after [[RedPajama]]) -- [[TogetherAI]]
[RedPajama-Data-v2: An open dataset with 30 trillion tokens for training large language models](https://www.together.ai/blog/redpajama-data-v2)

==30T== filtered and deduplicated tokens (from 100T raw tokens) from 84 CommonCrawl dumps covering 5 languages, along with 40+ pre-computed data quality annotations that can be used for further filtering and weighting.

This is, at the time, the ==largest public dataset== released specifically for LLM training. Even more excitingly, we include 40+ pre-computer quality annotations, allowing the community to *further* filter and weigh the data.

Many other projects like [[C4]], [[RedPajama]], [[RefinedWeb]], [[Dolma]] cover only a small portion of the CommonCrawl crawls, and represent a very specific way in which data are filtered.
- In contrast, ==our goal== with this is to lift this burden off of the community and ==provide a pool of web data serving as a base from which high-quality datasets for LLM training can be extracted==.
- More importantly, we provide ==40+ quality annotations== — the result of different ML classifiers on data quality, minhash results that can be used for fuzzy deduplication, or heuristics such as “the fraction of words that contain no alphabetical character”.
- These annotations provide a way for an LLM developer to easily slice and filter the data, combining these into a new data quality pipeline to create their own pre-training dataset.

The authors note that other data sources like Wikipedia are available in RedPajama-V1, and they encourage us to enrich our data mixtures with [[The Stack]] for code and [[S2ORC]] for scientific articles.

They use [[CCNet]] pipelines (due to light processing), and keep five languages in this release (English, French, Spanish, Italian, German).

==There's some really great information in this blog post, I encourage me to read it later!==