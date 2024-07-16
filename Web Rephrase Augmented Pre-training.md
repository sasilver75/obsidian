---
aliases:
  - WRAP
  - Rephrasing the Web
---
January 29, 2024
[[Apple]]
Paper: [Rephrasing the Web: A Recipe for Compute and Data-Efficient Language Modeling](https://arxiv.org/abs/2401.16380)

Web Rephrase Augmented Pre-training (WRAP) is a technique aimed at ==enhancing LM training efficiency by rephrasing web documents into styles like Wikipedia, or question-answer formats== using Mistral-7B-instruct. This challenges the idea of learning from unstructured web data, which typically requires significant compute and data resources.

WRAP involves prompting a pre-trained LLM to generate paraphrases of web documents into ==four distinct styles==:
1. Easy - Understandable even by a toddler
2. Medium - Similar to a Wikipedia article
	- Prompt: *For the following paragraph, give me a paraphrase of the same in high-quality English language as in sentences on Wikipedia"*
3. Hard - In terse and abstruse languages
4. Q/A - In question-answering format

Application of WRAP on [[C4]] resulted in ~3x faster pre-training, and improved model perplexity by over 10%, across various subsets of [[The Pile]].

Authors list two challenges faced during synthetic data curation:
1. Generation cost
	- Addressed by using an open-source, smaller LLM (1.8b/7b) to perform rephrasals
2. Data bias
	- Thanks to the information-maintaining nature of rephrasing, we're able to leverage the natural diversity of the web, rather than relying on an LLM directly for information, which might be ((more)) prone to factual errors/data biases. 
	- Our work shows that "style" alone can result in gains in downstream performance!

Interesting notes:
- Authors set a maximum length of 300 for their rephrasings, which was decided based on their observation that if they asked an LLM to rephrase in more than 300 tokens, it often led to loss of information.

Abstract
> Large language models are trained on ==massive scrapes of the web==, which are often ==unstructured, noisy, and poorly phrased==. Current scaling laws show that learning from such data requires an abundance of both compute and data, which grows with the size of the model being trained. This is infeasible both because of the large compute costs and duration associated with pre-training, and the ==impending scarcity of high-quality data on the web==. In this work, we propose ==Web Rephrase Augmented Pre-training== (==WRAP==) that uses an ==off-the-shelf instruction-tuned model prompted to paraphrase documents on the web in specific styles such as "like Wikipedia" or in "question-answer format" to jointly pre-train LLMs on real and synthetic rephrases==. First, we show that using WRAP on the C4 dataset, which is naturally noisy, speeds up pre-training by ∼3x. At the same pre-training compute budget, it ==improves perplexity by more than 10% on average across different subsets of the Pile==, and improves zero-shot question answer accuracy across 13 tasks by more than 2%. Second, we investigate the impact of the re-phrasing style on the performance of the model, offering insights into how the composition of the training data can impact the performance of LLMs in OOD settings. Our gains are attributed to the fact that re-phrased synthetic data has higher utility than just real data because it (i) *incorporates style diversity that closely reflects downstream evaluation style*, and (ii) has higher 'quality' than web-scraped data.

Above:
- I assume "speeds up pre-training by ~3x" is analogous to "compressing the ((noisy)) datasets by a favor of 3"
- "Incorporates style diversity that closely reflects downstream evaluation style"; Isn't this pretty much saying that you're overfitting to the evaluations?

![[Pasted image 20240429023253.png]]