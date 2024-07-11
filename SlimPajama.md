---
aliases:
  - SlimPajama627B
---

June 9, 2023 (~2 months after [[RedPajama]]) -- [[Cerebras]]
Blog: [SlimPajama](https://www.cerebras.net/blog/slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama)

==SlimPajama627B== is a extensively deduplicated, multi-corpora, open-source dataset for training LLMs.

Created by cleaning and deduplicating the 1.2T token [[RedPajama]] dataset from [[Together AI]] -- by ==filtering out low quality data and duplicates==, they cut the dataset in half, from 1210B to 627B tokens.

> To produce SlimPajama, we first removed short, low quality documents from RedPajama. After removing punctuation, space symbols, newlines and tabs, we filtered out documents with less than 200 characters. These documents typically contain only meta data and no useful information. Low-length filter was applied to every corpora other than Books and GitHub where we found useful short documents. In total this removed 1.86% of documents from RedPajama.
> To perform deduplication we used MinHashLSH Leskovec et al. (2014) with a Jaccard similarity threshold of 0.8. We construct document signatures on top of pre-processed lower-cased 13-grams. Pre-processing includes removing punctuation, consecutive spaces, newlines and tabs. We also strip the documents to remove any leading or trailing escape characters. Deduplication was performed both within and between data sources.

In addition to the data, they release the tools they used to create SlimPajama
- Applying MinHashLSH deduplication to trillion-token datasets like RedPajama wasn't possible with off-the-shelf open-source doe, so they produced an infrastructure that could do that, which they're open-sourcing.

