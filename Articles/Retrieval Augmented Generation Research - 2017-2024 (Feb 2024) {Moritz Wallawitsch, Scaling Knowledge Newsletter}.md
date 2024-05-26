#article #elite
Link: https://scalingknowledge.substack.com/p/rag

-----

This article draws inspiration from the excellent "Stanford CS25: V3, Retrieval Augmented Language Models" by Douwe Kiela, who, along with others, invented RAG in May 2020.

![[Pasted image 20240413231441.png]]

![[Pasted image 20240413231455.png]]

# Retrieval

## Sparse vs. Dense Retrieval
- Sparse vectors are called *sparse* because they're sparsely populated with information (a lot of values are zeroes, because most words don't appear frequently).
	- These sparse vectors require less computational resources to process, and are often used to find information about a specific Brand or Object ((embedding lookup?)), but they don't handle semantic meaning in the same way that dense vectors do.

Popular embedding examples are [[BM25]] and [[TF-IDF]] 
- ((Not sure what he means by embedding examples here))

Dense retrieval enabled searching for semantic similarity. Unlike sparse vectors, the numbers in a dense vector represent learned relationships between words, capturing their meaning in a compact way. This means that semantically similar words like "doctor" and "physician" will have similar embeddings.


### ==ORQA==: Latent Retrieval for Weakly-Suprised Open Domain Question Answering (Lee et al, 2019)

- One of the first Q&A systems built on dense embeddings. It's ==trained end-to-end to jointly learn evidence retrieval and answering==, using only question-answer pairs!
- It treats retrieval as an unsupervised, latent variable initialized by pre-training on an Inverse Cloze Task (predicting a sentence's surrounding context).

![[Pasted image 20240414150030.png]]


# Vector DBs and Sparse-Dense Hybrids
- ==Maximumum Inner-Product Search (MIPS)== involves finding the vector in a given set that maximizes the inner product with a query vector.
	- ((An inner product is also known as a [[Dot Product]]))

- [[FAISS]] is a library for efficient similarity search (2019)
	- It implements approximate nearest neighbor search ([[Approximate Nearest Neighbor Search|ANN]]) to solve MIPS search  problems -- Faiss laid the foundation for many of today's popular vector DBs.

- [[ColBERT]]: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT (Khattab, 2020)
	- A SoTA neural search model enabling efficient semantic search by independently encoding queries and documents, before comparing their fine grained similarity via [[Late Interaction]] (delaying interaction until after separate encodings are created).
	- ==It finds maximum similarity matches between each query token and document tokens, aggregating these to efficiently estimate overall relevance (up to 170x faster, compared with prior BERT-based retrieval models).==
		- ((It seems notable in that it's comparing token-level embeddings, rather than document-level embeddings. I don't see how this would be *faster* though -- if anything, wouldn't more comparisons take longer?))
![[Pasted image 20240414150941.png]]

- [[SPLADE]]: Sparse Lexical and Expansion Model for First Stage Ranking (Formal et al, 2021)
	- An interesting hybrid between Sparse and Dense retrievers.
	- ==A sparse retriever that uses *query expansion, identifying synonyms, and related terms for the query*, enhancing its ability to capture semantic meaning even when not contained in the query.==
![[Pasted image 20240414152041.png]]

- [[DRAGON]]:  Diverse Augmentation Towards Generalizable Dense Retrieval (Lin et al, 2023)
	- A generalized dense retriever, ==undergoes training with progressive data augmentation, gradually introducing more challenging supervisions and diverse relevance labels over multiple training iterations==, enabling the model to learn complex relevance patterns effectively.
		- ((Sort of sounds like SPLADE, but also with a curriculum approach?))

![[Pasted image 20240414152642.png]]

- [[SANTA]]: Structure-Aware Language Model Pretraining (Li et al 2023)
	- SANTA (Structure Aware DeNse ReTrievAl) ==addresses the challenge of *aligning (natural, unstructured) queries with structured external documents*==, especially when addressing the incongruity between structured data (like code or product specifications) and text data (such as text descriptions).
	- It enhances the retriever's sensitivity to structured information through two pre-training strategies:
		1. Leveraging the intrinsic alignment between structured and unstructured data to inform *contrastive learning* in a structure-aware pre-training scheme
		2. Implementing ==Masked Entity Prediction== (utilizing an entity-centric masking strategy) that encourages LMs to predict and fill in the masked entities, fostering a deep understanding of structured data.

![[Pasted image 20240414154047.png|350]]


# 🧊 Frozen vs Dynamic RAG 🔥

The industry has mostly viewed the components of the RAG architecture as separate components that work in isolation -- we can call this ==Frozen RAG==.

In contrast, some research has focused on iteratively improving the individual components (we call this "Advanced RAG").

Ideally, in a "Fully Dynamic" model, the gradients from the loss function would flow back into the *entire system*, training the entire system end-to-end (==document encoder, retriever, generator==).
But this is computationally challenging and hasn't been done successfully!
## Dynamic Retriever but Fixed Generator

#### In-Context Retrieval-Augmented Language Models (Ram et al, 2023)
- The authors of this paper introduce a ==re-ranker==, which ranks the retrieved results (using a simple, sparse BM25) before passing them into the LLMs context.
- This component is *==dynamic==*, meaning the training signal of the entire model is backpropagated into the re-ranker.
	- They show that this optimization results in performance gains allowing a 345M parameter GPT-2 model to exceed the performance of a 1.5B GPT-2 model!
#### REPLUG: Retrieval-Augmented Black-Box Language Models (Shi et al, 2023)
- In this framework, the ==*language model* is treated as a frozen, black box, but is augmented with a tunable retriever model==!
- This name stems from the idea that you can *plug* any LM into the system.
- The retrieved documents/elements are presented to the LM separately, and we compute the perplexity of the model given the query and the retrieved item (LM likelihood).
	- This information is used to train the retriever to select:
		1. The highest Retriever Likelihood (using a similarity score)
		2. The lowest perplexity documents
- This framework does not work for any model that doesn't provide a perplexity score.
![[Pasted image 20240414160818.png]]

#### DREditor: A time-efficient approach for building a Domain-specificDense Retrieval Model (Huang et al, 2024)
- The authors propose DREditor, a time-efficient approach to ==customizing off-the-shelf dense retrieval models to *new domains* by directly editing their semantic matching rules (i.e. how the model compares vectors in the embedding)==.
- Motivated by needs in enterprise search for scalable and fast search engine specialization across corpora, DREditor calibrates model output embeddings using an efficient closed-form linear mapping (calculating the adjustment) instead of the usual long adaptation fine-tuning (similar to what REPLUG is doing).

Experiments on domain transfer and zero-shot setups show 100x-300x faster run times than finetuning, while maintaining or improving accuracy.

((I don't care about this one.))

![[Pasted image 20240414161218.png|300]]

![[Pasted image 20240414161232.png|500]]

# Dynamic Generator but a Fixed Retriever

#### FiD: [[Fusion-in-Decoder]] (Izacard and Trave, 2020)
- This paper addresses a core limitation of many RAG systems, which is that we have to cram all the documents into the LM context, which is limited to the model's context size.
- In this framework, we combine (concatenate) the query vector and the retrieved passage vectors, before *decoding* them together into an answer ((which is then passed into the context?))
![[Pasted image 20240414161433.png]]

#### [[KG-FiD]]: Infusing *Knowledge Graph* in Fusion-in-Decoder for Open-Domain Question Answering (Yue et al, 2021)
- This paper adds *another* [[Graph Neural Network]] (GNN) re-ranking/filtering step to the FiD pipeline explained above.
- The authors point out that FiD and ==other RAG frameworks wrongly assume that the content of the retrieved passages are *independent* from eachother -- but the entities references in the retrieved passages are likely related to eachother, and their relationship can be modeled==!

The steps of the framework can be summarized as:
1. Retrieve relevant passages and embeddings via [[Dense Passage Retrieval]] (DPR)
2. Construct a knowledge graph from WikiData and neighboring context passages.
3. Utilizing Graph Neural Network (GNN) for iterative re-ranking of passages based on semantic relationships.
4. Update passage embeddings to eliminate less relevant passages and enhance top-ranked selections for answer generations.

![[Pasted image 20240414162539.png]]


#### SURGE: Knowledge Graph-Augmented Language Models for Knowledge Grounded Dialogue Generation (Kang et al, 2023)
- Subgraph Retrieval-Augmented Generation (SURGE) addresses the problem with prior graph-based retrieval techniques where the LM can get confused by irrelevant content.
	- ==Their framework aims to retrieve only a context-relevant subgraph, and is end-to-end trainable along with a generative model.==
- The GNN-based context-relevant subgraph retriever extracts relevant pieces of knowledge from a Knowledge Graph (no vector DB), and extracts candidate triplets (3 nodes).
	- For each triplet, we generate a Retrieval Distribution by calculating the inner product between the Context Embedding (based on Dialog History) and our candidate triplet.
	- This process involves exponentiating the inner product of the triplet embedding and the context embedding, resulting in a score that determines the relevance of the triplet to the dialogue history.

The authors further leverage contrastive learning to train the model to distinguish between knowledge-grounded responses (using the retrieved subgraph) and irrelevant alternatives, mitigating the [[Exposure Bias]] that arises from only showing input and a single "correct" output during training.

![[Pasted image 20240414163102.png]]

#### KNN-LM: Generalization through Memorization: Nearest Neighbor Language Models (Khandelwal et al, 2019)
- This is another interesting paper in which the authors attempt to make the LM outputs more grounded.
- This is done by comparing the vector distance of the model's initial prediction to similar/neighboring passages from a data store.
- The database in this case is a collection of key-value pairs comprised of a token and its proceeding tokens (its context).
- In the end, the normalized KNN model outputs, ranked by their distances, and the LM output distribution are merged (interpolated) to converge on the final output.

![[Pasted image 20240414163422.png]]


#### [[Retrieval-Augmented Generation|RAG]]: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al, 2020)
- This paper is the ==origin of the idea of a *dynamic end-to-end trained RAG system backpropagating into both the retriever and the generator==*
	- (However, the document encoder step in this and the next paper is still static/frozen)
![[Pasted image 20240414171247.png]]

#### [[RETRO]]: Improving language models by retrieving from trillions of tokens (Borgead et al, Deepmind, 2022)
- This paper showed that ==using RAG as you pretrain an LM from scratch can outperform a 25x bigger model in terms of perplexity==!
	- It's ==unclear if this paper is reproducible==; Deepmind never published the paper's code, and the author heard that other tier-1 AI research companies failed to reproduce it.
- In RETRO, the retrieved chunks are selected similarly to the (eg) KNN process above, then added to the query, and ==processed by the transformer *encoder*== (using chunked cross-attention)...
	- In contrast, in RAG and related architectures, the retrieved passages are used as additional context for the transformer *decoder*!

# Fully-Dynamic RAG

#### [[REALM]]: Retrieval-Augmented Language Model Pre-Training (Guu et al, Deepmind, 2020)
- This paper represents ==the first fully-dynamic RAG model==, in which all of the *encoder, retriever, and generator* are trained jointly.
- It's main limitation is that it's ==not truly generate, just BERT-based==, limiting its ability to produce completely novel/free-form text.
- Updating the document encoder is costly, so REALM introduces asynchronous updates, where the knowledge base is re-embedded in batches
	- ((When you update your embedding model, you have to re-embed all of the documents! So we batch updates to our embedder so that we don't have to re-embed all of our documents after every batch of data))
![[Pasted image 20240414172613.png|400]]


# Other Retrieval Research

### [[FLARE]]: Forward-looking active retrieval augmentation (Jiang, Xu, Gao, Sun et al 2023)
- A limitation of some of the above-explained techniques is that sequentially retrieve ***then*** generate. This paper proposes a system that iteratively predicts the next sentence to retrieve relevant context if it contains low-confidence tokens.
	- ((This sort of reminds me of something like speculative decoding, a bit?))

![[Pasted image 20240414174433.png]]

#### [[HyDE]]: Hypothetical Document Embeddings (Gao, Ma et al 2022)
- A core problem of retrieval is that ==the user's query might not capture their actual intent -- there's a different between what someone *thinks* they want to know and what they *actually* want to know.==
- This paper aims to address that through an intermediate ==query-rephrasing step==.
- This leads to a process where the main weakness to vanilla LLMs is dampened by their main weakness: Hallucination against hallucination.

![[Pasted image 20240414182137.png]]

#### MuGI: Enhancing Information Retrieval through Multi-Text Generation Integration with Large Language Models (Zhang et al, 2024)
- This paper builds on the above Query Re-writing idea. It introduces a framework named Multi-Text Generation Integration (MuGI).
- This framework involves ==prompting an LLM to generate multiple pseudo-references==, which are then dynamically integrated with the original query for retrieval -- this modle is used both for re-ranking and retrieval.

![[Pasted image 20240414193253.png]] 

#### Query Rewriting for Retrieval-Augmented Large Language Models (Ma et al, 2023)
- This paper introduces a trainable rewrite-retrieve-read framework (reversing the traditional retrieval and reading order, focusing on query rewriting) that utilizes the LLM performance as a reinforcement learning incentive for a rewriting module.
![[Pasted image 20240414193430.png]]

#### Lost in the Middle: How Language Models Use Long Contexts (Liu et al, 2023)
- This paper points out a core problem with passing a long list of context items into the model, sorted by relevance, since it ==attends more to documents at the beginning and end==.
![[Pasted image 20240414194315.png|400]]

#### SILO Language Models: Isolating Legal Risk in Nonparametric Datastore (Min, Gururangan et al 2023)
- Suggests a solution to recent copywright infringement lawsuits where companies like the NYT are suing companies for training on their paywalled data.
- The authors suggest only using public domain data during training, but then augmenting the model with "higher-risk data" during test/inference time.
	- ((I'm not sure why this is obviously better...I can imagine examples.))
![[Pasted image 20240414194534.png|400]]


#### [[CRAG]]: Corrective Retrieval Augmented Generation (Yan et al, 2024)
- This paper ==proposes a method to improve the robustness of RAG models when retrieval fails to return relevant documents.==
- CRAG tackles this by:
	1. Assessing retrieved document quality with a confidence score
	2. Launching web searches for inaccurate retrievals
	3. Refining knowledge with a decompose-then-recompose algorithm (segmenting the document into fine-grained strips, and concatenating the relevant ones).
- CRAG improves RAG performance on short- and long-form generation tasks across diverse datasets, showcasing its generalizability and robustness.
![[Pasted image 20240414195229.png|450]]


#### [[WebGPT]]: Browser-assisted Question-Answering with human feedback (Nakono et al, 2021)
- The system presented here could also be termed *==Web Search Augmented Generation==*. The information retrieval model receives the query and can output browser commands like *click*, *scroll* to extract relevant paragraphs from web pages it determines as informative.
- It's trained on human demonstrations, using [[Imitation Learning]] (behavior cloning). 
- In the second step, a Text Synthesis Model synthesizes the answers.
- Finally, a reward model predicts the system output score.
- The entire system is then fine-tuned using human feedback, i.e. the reward model (RLHF). 

![[Pasted image 20240414223655.png]]


#### [[Toolformer]]: Large Language Models can Teach themselves to use tools (Shick et al, 2021)
- This paper is the generalization of the idea of Augmented Generation; it presents a solution taht ==allows LLMs to use external tools via simple APIs, including a calculator, search engines, a Q&A system, a translation system, and a calendar.==
- The steps can summarized as follows:
	1. Authors annotate a large text dataset and sample potential locations in the text where tools API calls could be useful.
	2. At each location, they generate possible API calls to different tools
	3. They execute the API calls and insert both the call+response back into the original text
		- eg "\[QA(Who founded Apple?) -> Steve Jobs\]"
	4. They check if adding the app call reduced the perplexity loss of the LM for predicting the following token and keep the API call if it did, and 
	5. The resulting training data is used to fine-tune the original LM.

the system has many ==limitations==, such as:
1. The ability to use tools in a chain ⛓
2. The ability to use tools interactively 🤝
3. The ability to take into account the *cost* of using a tool 💸

![[Pasted image 20240415001134.png|400]]


#### 🦍 [[Gorilla]]: Large Language Model Connected with Massive APIs (Patil et al, 2023)

- One limitation of the Toolformer paper was that its tool use is limited to a relatively small set of tools. In contrast, the authors of this paper ==develop a retrieval-based finetuning strategy to train an LLM, called Gorilla, to use over 1,6000 different deep learning model APIs== (e.g. from HuggingFace or TensorFlow Hub) for problem-solving.
- Procedure:
	1. First, it downloads the API documentation of various tools.
	2. It then uses this data to create question-answer pair dataset (using [[Self-Instruct]])
	3. Finally, the 7B model is finetuned over this dataset in a retrieval-aware manner.

![[Pasted image 20240415002005.png]]


#### [[Self-RAG]]: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al, 2023)
- The authors point out the problem with most RAG systems, which is that they retrieve passages indiscriminately, regardless of whether the factual grounding is helpful.
- The Self-RAG algorithm uses a special type of token called a "*reflection token*" to communicate between the different parts of the algorithm:
	- Retrieve
	- IsRel (relevance)
	- IsSup (fully or not supporting)
	- IsUse (useful response)

![[Pasted image 20240415002228.png|500]]
See above also: Conditional tool use

#### GRIT: Generative Representational Instruction Tuning (Muennighoff et al, 2024)
- Addresses a similar problem to the above-mentioned paper (that some documents that are retrieved are just not helpful to the prompt?) while being very performant.
- The authors train a single LLM to perform both text generation and embeddings tasks via "Generative Instruction Tuning" -- in other words, the model architecture of GRITLM lets it process input text, create embeddings, and generate output text...
- Performance is enhanced (beyond the conditional tool use ability) by:
	1. The query's vector representations for retrieval and generation.
	2. Reusing the document key-value store (basically the raw retrieved vector DB data) for generation
((??))

Outperforms all generative models up to its size of 7 billion parameters, excelling in both generative and embedding tasks as demonstrated on the Massive Text Embedding Benchmark (MTEB) and various other evaluation benchmarks.

![[Pasted image 20240415003438.png]]

![[Pasted image 20240415003522.png]]













