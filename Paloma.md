---
aliases:
  - Perplexity Analysis for Language Model Assessment
---

December 16, 2023 -- [[Allen Institute]]
Paper: [Paloma: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523)


A perplexity evaluation dataset from a diverse set of 585 domains, developed by [[Allen Institute]] together with the [[Dolma]] pre-training dataset and [[OLMo]] family of models.

Compared to prior work, Paloma ==significantly improves the *diversity* of perplexity-based evaluation benchmarks==, allowing us to determine whether an LLM can accurately model text across a wide variety of domains.

Abstract
> Language models (LMs) commonly report perplexity on monolithic data held out from training. Implicitly or explicitly, this data is composed of domains–varying distributions of language. Rather than assuming perplexity on one distribution extrapolates to others, Perplexity Analysis for Language Model Assessment (Paloma), ==measures LM fit to 585 text domains==, ==ranging from The New York Times to r/depression on Reddit==. We invite submissions to our benchmark and organize results by comparability based on compliance with guidelines such as removal of benchmark contamination from pretraining. Submissions can also record parameter and training token count to make comparisons of Pareto efficiency for performance as a function of these measures of cost. We populate our benchmark with results from 6 baselines pretrained on popular corpora. In case studies, we demonstrate analyses that are possible with Paloma, such as finding that pretraining without data beyond Common Crawl leads to inconsistent fit to many domains.


![[Pasted image 20240420145948.png]]