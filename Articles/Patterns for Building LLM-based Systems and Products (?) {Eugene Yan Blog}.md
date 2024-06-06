---

---
Link: https://eugeneyan.com/writing/llm-patterns/?utm_source=convertkit&utm_medium=email&utm_campaign=2023+Year+in+Review%20-%2012699108#guardrails-to-ensure-output-quality

Note: I think this is going to be pretty similar to [[What we learned from a Year of Building with LLMs (Part 1) (May 28, 2024) {Eugene Yan, Bryan Bischof, Charles Frye, Hamel Husain, Jason Liu, Shreya Shankar}]] , which also included Eugene as an author.

-----

This write-up is about practical patterns for integrating LLMs into systems and products; we'll build on:
- Academic Research
- Industry Resources
- Practitioner Know-How
and distill them into key ideas and practices


We're going to talk about ==Seven Key Patterns==:
1. ==Evals==: To measure performance
2. ==RAG==: To add recent, external knowledge
3. ==Fine-tuning==: To get better at specific tasks
4. ==Caching==: to reduce latency and cost
5. ==Guardrails==: To ensure output quality
6. ==Defensive UX:== To anticipate and manage errors gracefully
7. ==Collecting user feedback==: To build our data flywheel

![[Pasted image 20240605192030.png]]

## (1/7) Evaluations: To Measure Performance
- Evaluations are a set of measurements used to assess a model's performance on a task -- they include benchmark data and metrics.

> ==How important evals are to the team is a *major differentiator* between folks rushing out how garbage, and those seriously building products in the space.== 

Why do we need evaluations?
- Enable us to measure how well our system or product is doing, and to detect any regressions.
- Without evaluations, we're flying blind, or would have to manually/visually inspect LLM outputs with each change.

There are many benchmarks in the field of language modeling:
- [[Massive Multi-Task Language Understanding|MMLU]]: A set of 57 tasks that span elementary math, US history, CS, law, and more. To perform well, you need both extensive world knowledge and problem-solving ability.
- [[Eleuther LM Evaluation Harness]]: A unified framework to test models via zero/few-shot settings on 200 tests. Incorporates a large number of *other* evaluations, including [[BIG-Bench]], [[Massive Multi-Task Language Understanding|MMLU]], etc.
- [[HELM]]: Instead of specific tasks and metrics, HELM aims to offer a *holistic and comprehensive* assessment of LLMs by evaluating them across multiple domains. Metrics include accuracy, calibration, robustness, fairness, bias, toxicity, etc. Tasks include Q&A, information retrieval, summarization, text classification, etc.
- [[AlpacaEval]]: *Automated* evaluation framework which measures how often a strong LLM (e.g. GPT-4) prefers the output of one model over a reference model (eg TextDavinci003). Metrics include win rate, bias, latency, price, variance. Validated to have high agreement with 20k human annotations.

We can group metrics into two categories:
- ==Context-dependent==: These take context (domain/use-case) into account. They're often proposed for a specific task (eg MT). Repurposing them for other tasks will require some adjustment.
- ==Context-free==: Aren't tied to the context/use-case when evaluating generated output; They only compare the output with the provided gold references. Because they're task-agnostic, they're easier to apply across a variety of tasks.

To get a better sense of these metrics, we'll explore a few of the commonly-used ones:

### BLEU: Bilingual Evaluation Understudy
- [[BLEU]] is a precision-based metric often used for [[Machine Translation|MT]]; it counts the number of n-grams in the generated output that are found in the gold reference, and divides it by the total number of words in the output. It remains a popular metric due to its cost-effectiveness, but has some problems.
	- ((What percentage of the words in the output are found in the input?))
- ![[Pasted image 20240605203304.png]]
- ![[Pasted image 20240605203315.png]]
Above: I think p refers to the generated sentence, and r to the source sentence. $|p|$ is the *length* of the generated sentence.

### ROGUE: Recall-Oriented Understudy for Gisting Evaluation
- In contrast to BLEU, [[ROGUE]] is recall-oriented, counting the number of words in the *reference* that also occur in the *output.* It's typically used to assess automatic [[Summarization]] tasks.
- There are several ROGUE variants; ROGUE-N is most similar to BLEU in that it also counts the number of matching n-grams between the output and the reference.
![[Pasted image 20240605205119.png]]
Other variants include:
- ROGUE-L: Measures the longest common subsequence (LCS) between output and reference.
- ROGUE-S: Measures the skip-bigram between the output and reference.


### BERTScore
- [[BERTScore]] is an embedding-based metric that uses cosine similarity to compare each token or n-gram in the generated output with the reference sentence.
- There are three components:
	- Recall: Average cosine similarity between each token in the *reference* and its closest match in the *generated output*.
	- Precision: Average cosine similarity between each token in the *generated output* and its nearest match in the *reference*.
	- F1: Harmonic mean of precision and recall
![[Pasted image 20240605205343.png]]
Useful because it's able to account for synonyms and paraphrasing, which simpler metrics like BLEU and ROGUE can't due, due to their reliance on exact matches.

### MoverScore
- Uses contextualized embeddings to compute the distance between tokens in the generated output and reference. Unlike BERTScore, which is based on one-to-one matching (hard alignment) of tokens, MoverScore allows for many-to-one matching (soft alignment).
- ![[Pasted image 20240605205537.png]]


The G-Eval is a framework that applies LLMs with Chain of Thought (CoT) and a form-filling paradigm to evaluate LLM outputs. First, they provide a task introduction and evaluation criteria to an LLM and ask it to generate a CoT of evaluation steps. Then, to evaluate coherence in news summarization, they concatenate the prompt, CoT, news article, and summary and ask the LLM to output a score between 1 to 5. Finally, they use the prombabilities of the output tokens frmo the LLM to normalize the score and take their weighted summation as the final result.

![[Pasted image 20240605210102.png]]

...

----

How to apply evals?
- Building solid evals should be the starting point for any LLM-based system! 
- Classical metrics like BLEU and ROGUE don't make any sense for more complex tasks such as abstractive summarization or dialogue.
- We've seen that benchmarks like MMLU are sensitive to how they're implemented and measured -- and to be candid, ==unless your LLM system is studying for a school exam, MMLU probably isn't actually a good evaluation==.

So instead of using off-the-shelf benchmarks, we should start by collecting a set of ==task-specific evals==, including (prompt, context, expected outputs).
- These will guide prompt engineering, model selection, fine-tuning, and so on.
As we update our systems, we can run these evals to quickly measure improvements or regressions. Think of it as ==Eval Driven Development (EDD)==


We also need useful ==metrics==, in addition to our evaluation dataset!
- These help us distill performance changes into a single number that's comparable across eval runs.

The simplest task is probably Classification. If we're using an LLM for classification-like tasks (eg toxicity detection, document categorization), we can rely on standard classification metrics like [[Recall]], [[Precision]], [[PR-AUC]], etc.

If our task has no correct answer but we have references (eg Machine Translation, extractive summarization), we can use reference metrics based on term matching ([[BLEU]], [[ROGUE]]) or semantic similarity ([[BERTScore]], MoverScore)

But these metrics might not work well for open-ended tasks like abstractive summarization, dialogue, and others.
- Collecting human judgements in these situations is expensive, so we might lean to automatic evaluations via LLM-as-a-Judge. But be aware:
	- [[Positional Bias]]: LLMs favoring the response in the first position; to mitigate, evaluate the same pair of responses twice while swapping the order. Only if the same response is preferred in both orderings do we mark it as better.
	- [[Verbosity Bias]]: LLMs tend to favor longer, wordier responses over some more concise ones, even if the latter is clearer and of higher quality. A possible solution is to ensure that comparison responses are similar in length.
	- [[Self-Enhancement Bias]]: LLMs have a slight bias towards their own answers; to solve, don't use the same LLM for evaluation tasks!

Sometimes, the best evaluation is a human evaluation -- a ==vibe check==.

## (2/7) Retrieval Augmented Generation: To Add Knowledge
- [[Retrieval-Augmented Generation|RAG]] Fetches relevant data from outside the foundation model and enhances the input with this data, providing richer context.
	- Reduce hallucinations
	- Allow for the ability to cite responses
	- Allow for the ability to update knowledge; it's cheaper to keep retrieval indices up to date than continuously pre-train LLMs.

First, a little on text embeddings
- Text embeddings are compressed, abstract representations of text data where text of arbitrary length can be represented as fixed-size vectors of numbers. Similar items are closer to eachother, while dissimilar items are further apart.
	- HF's [[MTEB]] scores various models on diverse tasks like classification, clustering, retrieval, summarization, etc.

Quick note: While we mainly discuss text embeddings here, embeddings can take many modalities. For example, CLIP is multimodal and embeds images and text in the same space!

RAG has its roots in open-domain QA:
- An early Meta [paper](https://arxiv.org/abs/2005.04611) showed that retrieving relevant documents via [[TF-IDF]] and providing them as context to a LM ([[Bidirectional Encoder Representations from Transformers|BERT]]) improved performance on an open-domain task.

### Dense Passage Retrieval (DPR)
- Later, Meta's [[Dense Passage Retrieval]] (DPR) showed that dense embeddings (rather than sparse ones used in TF-IDF) for document retrieval can outperform strong baselines like [[BM25]]... and that higher retrieval precision translates to higher e2e QA accuracy.
	- They fine-tuned two independent BERT-based encoders on existing (question, answer) pairs, embedding each with one of the two encoders and doing a MIPS search to retrieve k relevant passages.
	- Encoders trained so that the dot-product similarity makes a good ranking function, optimizing the loss function as the negative los likelihood of the positive passage.
	- Passage embeddings were indexed in [[FAISS]] offline, then at query time, they compute question embeddings and retrieve the top k passages via ANN, providing them to a third language model (BERT) that outputs the answer to the question.

### Retrieval-Augmented Generation (RAG) Model
- [[Retrieval-Augmented Generation (Model)|RAG]] highlighted the downsides of pretrained LLMs, and introduced RAG (aka semi-parametric models) -- they reused DPR encoders to initialize the retriever, and used 400M param [[BART]] as an LLM.
	- ![[Pasted image 20240605213506.png]]
	- Above: Notably, the entire thing is fine-tuned end-to-end.
	- Proposed two sequences for how retrieved passages are used to generate output:
		- ==RAG-Sequence==: The same document is used to generate the complete sequence. The generator produces an output for each of the k retrieved documents, then the output is marginalized (we sum the probability of each output sequence from our k documents, and weigh them by the probability of each document being retrieved). Finally, the output sequence with the highest probability is selected.
		- ==RAG-Token==:  Each token can be generated based on a *different* document. Given k retrieved documents, the generator produces a distribution for the next output token for *each document*, before marginalizing (aggregating all of the individual token distributions). This process is repeated for the next token. This means that for each token generation, we can retrieve a set of k relevant documents based on both the original input *and* the previously-generated tokens. Thus, documents can have different retrieval probabilities, and contribute differently to the next generated token.

### Fusion in Decoder (FiD)
- [[Fusion-in-Decoder]] (FiD) also uses retrieval with generative models for open-domain QA.
- It supports two methods for retrieval:
	- BM25 (Lucene with default params)
	- DPR
- FiD is named for how it performs fusion on the retrieved documents in the decoder only:![[Pasted image 20240605214728.png]]
- For each retrieved passage, the title and passage are concatenated with the question, and then these pairs are processed independently in the encoder. The decoder attends over the concatenation of these retrieved passages.
- Because we process passages independently in the encoder, it can scale to a large number of passages as it only needs to do self-attention over one context at time. (so compute grows linearly, instead of quadratically) with the number of retrieved passages. Then, during decoding we process the encoded passages jointly, allowing the *decoder* to better aggregate context across multiple retrieved passages.

### RETRO: Retrieval-Enhanced Transformer
- Adopts a similar pattern as FiD above where it combines a frozen BERT retriever, a differentiable encoder, and a chunked cross-attention to generate output.
- ==What's different is that RETRO does retrieval throughout the entire pre-training stage, and not just during inference!==
- Furthermore, we fetch relevant documents based on *==chunks==* of the input -- this allows for finer-grained, repeated retrieval during generation, instead of only retrieving once per query.
	- The encoding of the retrieved chunks depends on the attended activations of the input chunk.
	- (There's more to this)

### Internet-Augmented LMs
- Proposes using a humble "off-the-shelf" search engine to augment LLMs
- The retrieve documents using google search, chunk these long documents into paragraphs of six sentences each, then embed the questions and paragraphs via TF-IDF and apply cosine similarity to rank the most relevant paragraphs for each query. 
- The retrieved passages condition the LLM (Gopher) via few-shot prompting. 
- For each question, they generate four candidate answers based on each of the 50 retrieved paragraphs, and then select the best answer by estimating the answer probability via several methods (direct inference, RAG, noisy channel inference, Product-of-Experts). PoE consistency performed the best.


### What if we don't have relevance judgements for query-passage pairs? Enter Hypothetical Document Embeddings (HyDE)
- Without these supervised pairs, we can't train the bi-encoders that learn to embed queries and documents in the same embedding space!
- Given a query, [[HyDE]] first prompts an LLM (InstructGPT) to generate a hypothetical document that would help us answer that query. 
- Then, an unsupervised encoder, like [[Contriever]], encodes the document into an embedding vector. 
- Finally, we compute the inner product between our *hypothetical document vector* and the *real* document vectors, and the most similar *real* documents are retrieved.
- The expectation is that the encoder's dense bottleneck serves as a lossy compressor, and the extraneous, non-factual details are excluded via the embedding.


### How to apply RAG
- Eugene has found that [[Hybrid Search]] works better than either alone.
	- Embedding-based search can fail when searching for a person/object's name, an acronym, or an ID.
	- Keyword search only models simple word frequencies and doesn't capture semantic or correlation information... but traditional search indices let us use metadata (custom ratings, date filters) to refine results.

How do we retrieve documents with low latency at scale?
- We use [[Approximate Nearest Neighbor Search|Approximate Nearest Neighber]] search, optimizing for retrieval speed and returns the approximate top-k most similar neighbors.
- ANN embedding indices are data structures that let us do ANN searches efficiently. Popular techniques include:
	- [[Locality Sensitive Hashing]]
	- [[FAISS]]
	- [[Hierarchical Navigable Small Worlds|HNSW]]
	- Scalable Nearest Neighbors (ScaNN)

When evaluating an ANN index, consider:
- Recall: How does it fare against exact nearest neighbors?
- Latency/throughput: How many queries can it handle per second?
- Memory footprint: How much RAM is required to serve an index?
- Ease of adding new items: Can new items be added without having to reindex all documents (LSH) or does the index need to be rebuilt (ScaNN)?


## (3/7) Fine-tuning: To get better at specific tasks
- The process of taking a pre-trained model and further refining it on a specific task. The intent is to harness the knowledge the model already has, and apply it to a specific task, usually involving a smaller, task-specific dataset.
- The term fine-tuning is used loosely, and can refer to:
	- ==Continued Pre-Training (CPT)==: With domain-specific data, apply the same pre-training regime (NTP, MLM) on the base model.
	- Instruction Tuning: Pretrained model is finetuned on examples of instruction-output pairs to follow instructions, answer questions, be waifu, etc.
	- Single-task fine-tuning: Pre-trained model is honed for a narrow and specific task like toxicity detection or summarization, similar to BERT and T5
	- RLHF: Combines instruction tuning with RL. The reward model is then used to further fine-tune the instructed LLM via RL techniques like PPO.

Let's focus mainly on single-task and instruction fine-tuning:

Why fine-tuning?
- Performance and control
- Modularization (Use an army of smaller models, each specializing on their own tasks, like content moderation, extraction, summarization, etc.)
- Reduced dependencies (regarding legal concerns about proprietary data being exposed to external APIs. Also gets around constraints with 3rd party LLMs like rate limiting, high costs, or overly restrictive safety filters)

More on finetuning
- Why do we finetune a ***base model***? 
	- Base models are primarily optimized for NTP on their training corpus.
- Fine-tuning isn't without its challenges, we need a significant volume of demonstration data! In InstructGTP, they used:
	- 13k instruction-output samples for SFT
	- 33k output comparisons for reward modeling
	- 31k prompts without human labels as input for RLHF
- Fine-tuning comes with an alignment task; the process can lead to lower performance on certain critical tasks.

There are some other fine-tuning techniques that don't involve updating all of the parameters of the model:
- Soft [[Prompt Tuning]] prepends a trainable tensor to the model's *input embeddings*. Unlike discrete text prompts, soft prompts can be learned via backpropagation, meaning they can be fine-tuned to incorporate signals from any number of labeled examples.
- In [[Prefix Tuning]], instead of adding a soft prompt to the model's input, we prepend trainable parameters to the hidden state of all transformer blocks! During fine-tuning, the LM's original parameters are kept frozen while the prefix parameters are updated.
	- Paper showed that this achieved performance comparable to full fine-tuning despite requiring updates on only 0.1% of parameters!

In the ==adapter== technique, we add fully-connected network layers twice to each transformer block; after the attendion alyer, 
