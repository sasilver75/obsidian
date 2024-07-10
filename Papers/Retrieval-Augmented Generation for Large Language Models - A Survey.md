*Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al.; March 27, 2024)* [Link](https://arxiv.org/abs/2312.10997v5) #zotero 

Abstract
> Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like ==hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes==. Retrieval-Augmented Generation (==RAG==) has emerged as a promising solution by incorporating knowledge from external databases. This ==enhances the accuracy and credibility of the generation==, particularly for knowledge-intensive tasks, and ==allows for continuous knowledge updates and integration of domain-specific information==. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the ==Naive RAG==, the ==Advanced RAG==, and the ==Modular RAG==. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the ==retrieval==, the ==generation== and the ==augmentation== techniques. The paper highlights the state-of-the-art technologies embedded in each of these critical components, providing a profound understanding of the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development.


Notes:
- The limitations of LMs include hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. RAG aims to solve each of them.
- The authors include a "technology tree" of RAG, which is kind of sick for the gamers out there.
- Authors note that this paper summarizes three main research paradigms from over 100 RAG studies, and analyzes techniques across the three core stages of "Retrieval," "Generation," and "Augmentation."
- Authors note that there are three paradigms of "naive RAG," "advanced RAG," and "modular RAG".

## Overview of RAG
- 1) ==Naive RAG==
	- Represents the earliest methodology that gained widespread adoption shortly after ChatGPT; it follows a traditional process that includes indexing, retrieval, and generation, which is also characterized as a "==Retrieve-Read" framework==
	- *Indexing*: Starts with the cleaning and extractions of raw data in diverse formats like PDF, HTML, Word, and Markdown, which is then converted into a uniform plain-text format. To accommodate the context limitations of language models, ==text is segmented into smaller, digestible chunks==. ==Chunks are then encoded into vector representations using an embedding model== and stored in a vector database.
	- *Retrieval*: Upon receipt of a user query, the RAG system employs the *same encoding model* utilized during the indexing phase to ==transform the query into a vector representation==. It then computes similarity scores between the query vector and the vector of chunks within the indexed corpus. The system prioritizes and ==retrieves the top K chunks that demonstrate the greatest similarity to the query==. These chunks are subsequently used as the expanded context in prompts.
	- *Generation*: The ==posed query and selected documents are synthesized into a coherent prompt== to which a large language model is tasked with formulating a response. The model's approach to answering may vary depending on task-specific criteria, allowing it to ==either draw upon its inherent parametric knowledge or restrict its responses to the information contained within the provided documents==.
	- *Drawbacks*
		- *Retrieval Challenges*: The retrieval phase struggles with precision and recall, leading tot he selection of misaligned or irrelevant chunks, missing crucial information.
		- *Generation Difficulties*: In generating responses, the model may face the issue of hallucination, where it produces content not supported by retrieved context.
		- *Augmentation Hurdles*: Integrating retrieved information with the different tasks can be challenging, sometimes resulting in incoherent outputs.... The process may also encounter redundancy when similar information is retrieved from multiple sources, leading to repetitive responses. Determining the significance and relevance of various passages and ensuring stylistic and tonal consistency add further complexity! When we face complex tasks, a *single retrieval* might not be sufficient to acquire adequate context information!
- 2) ==Advanced RAG==
	- Introduces specific improvements to overcome the limitations of naive RAG. Focusing on enhancing retrieval quality, it employs ==pre-retrieval== and ==post-retrieval== strategies.
	- To tackle indexing issues, AdvanceD RAG refines its indexing techniques through the use of a sliding window approach, fine-tuned segmentation, incorporation of metadata, and several optimization methods.
	- *Pre-retrieval Process*: In this stage, the primary focus on optimizing the indexing structure and the original query. 
		- The goal of ***optimizing indexing*** is to ==enhance the quality of the context being indexed==. This involves:
			- Enhancing data granularity
			- Optimizing index structures
			- Adding metadata
			- Alignment optimization
			- Mixed retrieval
		- The goal of **query optimization*** is to ==make the user's original question clearer and more suitable== for the retrieval task. This involves:
			- Query rewriting
			- Query transformation
			- Query expansion
			- Other techniques
	- *Post-Retrieval Process*:
		- Once relevant context is retrieved, it's crucial to integrate it effectively with the query.
		- The main methods in post-retrieval process include:
			- Reranking chunks
				- Reranking the retrieved information to relocate the most relevant content to the edges of the prompt is a key strategy.
			- Context compression
				- Feeding too many documents directly into LLMs can lead to information overload, diluting focus on key details with irrelevant content. To mitigate this, we might focus on selecting the most essential information, emphasizing critical sections, and shortening the context to be processed.
- 3) ==Modular RAG==
	- The modular RAG architecture advances beyond the former two RAG paradigms by offering enhanced adaptability and versatility. 
	- Might add a search module for similarity searches, refine the retriever through fine-tuning. 
	- New Modules
		- Introduces additional specialized components to enhance retrieval and processing capabilities. 
		- The ==Search== module adapts to specific scenarios, enabling direct searches across various data sources like search engines, databases, and knowledge graphs, using LLM-generated code and query languages.
			- ==RAG-Fusion== addresses traditional search limitations by employing a multi-query strategy that expands user queries into diverse perspectives, utilizing parallel vector searches and intelligent re-ranking to uncover both explicit and transformative knowledge.
		- The ==Memory== module leverages the LLM's memory to guide retrieval, creating an unbounded memory pool that aligns the text more closely with data distribution through iterative self-enhancement.
		- ==Routing== in the RAG system navigates through diverse data sources, selecting the optimal path for a query, whether it involves summarization, specific database searches, or merging different information streams.
		- The ==Predict== module aims to reduce redundancy and noise by generating context directly through the LLM.
		- The ==Task Adapter== module tailors RAG to various downstream tasks, automating prompt retrieval for zero-shot inputs, and creating task-specific retrievers through few-shot query generations.
	- New Patterns
		- Allows module substitution or reconfiguration to address specific challenges. 
		- Innovations like the ==Rewrite-Retrieve-Read== model leverage LLM's ability to refine retrieval queries through a rewriting modules, and an LM-feedback mechanism to update this rewriting model, improving task performance.
		- Approaches like ==Generate-Read== replace traditional retrieval with LLM-generated content, while ==Recite-Read== emphasizes retrieval from model weights, enhancing the model's ability to handle knowledge-intensive tasks.
		- ==Hybrid retrieval== strategies integrate keyword, semantic, and vector searches to cater to diverse queries. 
		- Additionally, employing sub-queries and hypothetical document embeddings ([[HyDE]]) seek to improve retrieval relevance by focusing on embedding similarities between generated answers and real documents.
		- Adjustments in module arrangement and interaction, like the Demonstrate-Search-Predict ([[DSPy]]) framework, and the iterative Retrieve-Read-Retrieve-Read flow of [[Iter-RetGen]] showcase the dynamic use of module outputs to bolster another module's functionality, illustrating a sophisticated understanding of enhancing module synergy.
		- Adaptive retrieval benefits are shown through techniques like [[FLARE]] and [[Self-RAG]], which evaluate the necessity of retrieval, based on different scenarios.
- RAG vs Finetuning
	- RAG is often compared with fine-tuning and prompt engineering; each method has distinct characteristics, but can be compared along the dimensions of *external knowledge requirements* and *model adaptation requirements*.
		- Prompt engineering leverages a model's inherent capabilities with minimum necessity for external knowledge and model adaptation.
		- RAG can be likened to providing a model with tailored information for IR. Excels in dynamic environments by offering realtime knowledge, but comes with higher latency. *Consistently outperforms fine-tuning for both existing knowledge encountered during training and entirely new knowledge.*
		- FT is comparable to a student internalizing knowledge over time, suitable for scenarios requiring replication of specific structures, styles, or formats. Requires retraining for updates, but enables deep customization of model's behavior and style.

## Retrieval
- In the context of RAG, it's crucial to ***efficiently*** retrieve ***relevant*** documents from the data source. Key issues include the retrieval data source, retrieval granularity, pre-processing of the retrieval, and selection of corresponding embedding model.
- Retrieval Source
	- The type of the retrieval source and the granularity of retrieval units both affect the final generation results.
	- Data Structure
		- Retrieval sources can be expanded to include semi-structured data (PDFs) and structured data (eg Knowledge Graphs) for enhancement.
		- In addition to retrieving from original text sources, there's also a growing trend of using content generated by LLMs themselves for retrieval and enhancement purposes.
		- Unstructured data like text is the most widely-used retrieval source. Can range from encyclopedic data, to cross-lingual texts, legal documents, etc.
		- Semi-structured data typically refers to data containing a combination of text and table information, such as PDFs. Poses a challenge to RAG systems, because inadvertently separating tables leads to data corruption, and incorporating tables can corrupt semantic similarity searches.
		- Structured data, like knowledge graphs, are typically verified can provide more precise information.
		- LLM-generated Content: Some research has focused on exploiting LLM's internal knowledge:
			- [SKR](https://arxiv.org/abs/2310.05002) classifies questions as known or unknown, applying retrieval enhancement selectively.
			- [GenRead](https://arxiv.org/abs/2310.05002) replaces the retriever with an LLM generator, finding that LLM-generated contexts often contain *more accurate answers*.
			- [Selfmem](https://arxiv.org/abs/2305.02437) iteratively creates an an unbounded memory pool with a retrieval-enhanced generator, using a memory selector to choose outputs that serve as dual problems to the original question, thus self-enhancing the generative model.
	- Retrieval Granularity
		- Coarse-grained retrieval units theoretically provide more relevant information for the problem, but may also contain redundant information that can distract both the retriever and the language model.
		- Fine-grained retrieval units increases the burden of retrieval, and doesn't guarantee semantic integrity and meeting the required knowledge.
		- Possible granularities include: Token, Phrase, Sentence, Proposition, Chunks, Document.
			- Propositions are defined as atomic expressions within text, encapsulating a unique factual segment.
		- On the KG side, retrieval granularity includes Entity, Triplet, and sub-Graph.
- Indexing Optimization
	- The quality of index construction determines whether the correct context can be obtain in the retrieval phase!
	- Chunking Strategy
		- The most common method is to split documents into chunks based on some fixed number of tokens. This leads to truncation within sentences, prompting the optimization of a recursive split and sliding window method, enabling layered retrieval by merging globally related information across multiple retrieval processes. (See Langchain: Recursively split by character)
		- Still, these approaches don't strike a balance between semantic completeness and context length; therefore, methods like Small2Big have been proposed, where sentences are used as the retrieval unit, and the preceding and following sentences are provided as (big) context to LLMs.
	- Metadata Attachments
		- Chunks can be enriched with metadata information like page number, file name, author, category timestamp. Subsequently, retrieval can be filtered based on this metadata, limiting the scope of the retrieval.
		- Assigning different weights to document timestamps during retrieval can achieve time-aware RAG, ensuring the freshness of knowledge and avoiding outdated information.
		- Metadata can also be artificially constructed; adding summaries of a paragraph, and introducing hypothetical questions. This method is known as ==Reverse HyDE==. Using LLM to generate questions that can be answered by the document, and then calculating the similarity between the original question and the hypothetical question(s) during retrieval to reduce the semantic gap between the question and the answer.
	- Structural Index
		- One effective method of enhancing information retrieval is to establish hierarchical structure for the documents. This can expedite the retrieval and processing of pertinent data.
		- In a hierarchical index structures, files might be arranged in parent-child relationships with chunks linked to them; data summaries are stored at each node, aiding in the swift traversal of data and assisting the RAG system in determining which chunks to extract.
- Query Optimization
	- One of the primary challenges with Naive RAG is its direct reliance on the user's original query as the basis for retrieval -- formulate precise and clear questions is difficult, and bad queries often result in bad retrieval. Furthermore, language itself is ambiguous -- even in an LLM context, does LLM refer to *large language model* or a *Master of Laws*?
	- ==Query Expansion==
		- Expanding a single query into multiple queries enriches the content of the query, providing further context to address any lack of specific nuances.
		- *Multi-Query*: Expand to multiple queries that can be executed in parallel. The expansion of queries isn't random, but rather meticulously designed.
		- *Sub-Query*: The process of sub-question planning represents the generation of the necessary sub-questions to contextualize and fully answer the original question, when combined.
		- *==Chain of Verification (CoVe)==*: The expanded queries undergo validation by LLM to achieve the affect of reducing hallucination. 
	- ==Query Transformations==
		- The goal is to retrieve chunks based on a transformed query, instead of a user's original query.
		- We can prompt an LLM to rewrite queries using specialized smaller language models like ==RRR== (Rewrite-Retrieve-Read). A query-rewrite method known as ==BEQUE== has notably enhanced recall effectiveness for long-tailed queries, resulting in a rise in GMV.
		- In the ==Step-back== prompting method, the original query is abstracted to generate a high-level concept question (a step-back question); both the step-back question and original query are used for retrieval, with both being utilized as the basis for LM generation.
	- Query Routing
		- Based on varying queries, routing to a distinct RAG pipeline. This is suitable for versatile RAG systems that need to accommodate diverse scenarios.
		- Metadata Router/Filters involve extracting keywords (entities) from the query, followed by filtering based on the keywords and metadata within the chunks, to narrow down search scope.
		- Semantic Router is another method of routing that involves leveraging the semantic information of the query. 
		- Hybrid routing approaches can be employed by combining both semantic and metadata-based methods for enhanced query routing.
- Embedding
	- In RAG, retrieval is achieved by calculating similarity (eg [[Cosine Similarity]]) between the embeddings of the question and document chunks. 
	- This mainly includes a sparse encoder ([[BM25]]) and a dense retriever ([[BERT|BERT]]). 
	- Mixed/Hybrid Retrieval
		- Sparse and dense embedding approaches capture different relevance features, and can benefit from eachother by leveraging complementary relevance information.
			- eg sparse retrieval models can be used to provide initial search results for training dense retrieval models.
	- Finetuning Embedding Models
		- In instances where context significantly deviate from pre-training corpus, finetuning the embedding model on your own domain-specific dataset becomes essential to mitigate such discrepancies.
		- In addition to supplementing domain knowledge, another purpose of fine-tuning is to align the retriever and generator, using the results of the LLM as supervision signal for fine-tuning ((of the retriever?)).
		- [[Promptagator]] utilizes the LLM as a few-shot query generator to create task-specific retrievers, addressing challenges in supervised fine-tuning -- particularly in data-scarce domains.
		- LLM-Embedder exploits LLMs to generate reward signals across multiple downstream tasks -- the retriever is fine-tuned with two types of supervised signals: hard labels for the dataset and soft rewards from the LLMs. 
		- REPLUG utilizes a retriever and an LLM to calculate the probability distributions of the retrieved documents, then performs supervised  training by computing the KL divergence.
- Adapter
	- To optimize the multi-task capabilities of LLMs, UPRISE trained a lightweight prompt retriever that can automatically retrieve prompts from a pre-built prompt pool that are suitable for a given zero-shot task input.
	- AAR (Augmentation-Adapted Retriever) introduces a universal adapter designed to accommodate multiple downstream tasks.
	- PRCA adds a pluggable reward-driven contextual adapter to enhance performance on specific tasks.
	- ==BGM== keeps the retriever and LLM fixed, and trains a bridge Seq2Seq model in between, which aims to transform the retrieved information into a format that the LLM can work with effectively.
	- PKG introduces an innovating method for integrating knowledge into white-box models via directive fine-tuning. In this approach, the retriever module is directly substituted to generate relevant documents according to a query.
- Generation
	- Context Curation
		- Redundant information can interfere with the final generation of LLM, and overly-long contexts can also lead LLMs to "lost in the middle" problems; LLMS tend to focus on mostly the beginning and end of long texts, so in RAG systems, we need to further process retrieved content.
		- ==*Reranking*==
			- Reranking fundamentally reorders document chunks (at the chunk-level) to highlight the most pertinent results first. It acts as both an enhancer and filter, delivering refined inputs for more precising processing. Reranking can be performed using rule-based methods or model-based approaches, which include specialized reranking models like Cohere rerank, in addition to the usual BERT suspects (Eg SpanBERT), or even frontier LLMs like GPT.
		- ==*Content Selection/[[Context Compression]]*==
			- A ==common misconception== is the belief that retrieving as many relevant documents as possible and concatenating them to form a lengthy retrieval prompt is beneficial -- excessive context can introduce more noise, diminishing the LLM's perception of key information!
			- LLMLingua uses small LMs to detect and remove unimportant tokens, transforming ((chunks?)) into a format challenging for humans to comprehend, but well-understood by LLMs.
			- PRCA and RECOMP both tackled this issue of balancing langauge integrity and compression ratio by training information extractors/condensers.
			- In addition to compressing the context, reducing the number of documents also helps improve the accuracy of the model's answers; Ma eta l propose the "Filter Reranker" paradigm, which combines the strengths of SLMs(for filtering) *and* LLMs(for reordering). Instructing LLMs to rearrange challenging samples identified by SLMs leads to significant improvement in various IE tasks.
	- LLM Fine-tuning
		- Fine-tuning is a good way to adjust model's intput and output; to adapt to specific data formats and to generate responses in a particualr style as instructed.
			- The [[SANTA]] framework is apparently a good example of this in the context of interacting with structured data?
		- Aligning LLM outputs with human/retriever preferences through RL is a potential approach... rather than aligning with human preferences.

## Augmentation Process in RAG
- Standard practice often involves a singular (once) retrieval step followed by generation, which is sometimes inefficient for complex queries.
- ==[[Iterative Retrieval]]==
	- A process where the knowledge base is repeatedly searched based on the initial query and the text generated so far, providing a more comprehensive knowledge base for LLMs.
	- However, ti may be affected by semantic discontinuity, and the accumulation of irrelevant information.
		- [[Iter-RetGen]] has a synergistic approach that leverages "retrieval-enhanced generation" alongside "generation-enhanced retrieval" for tasks that necessitate the reproduction of specific information.
- ==[[Recursive Retrieval]]==
	- Involves iteratively refining search queries based on the results obtained from previous searches. 
	- Aims to enhance the search experienced by gradually converging on the most pertinent information through a feedback loop.
	- [[IRCoT]] uses CoT to guide the retrieval process, and refines CoT with the obtained retrieval results.
	- [[Tree of Clarification]] (ToC) creates a clarification tree that systematically optimizes the ambiguous parts in the query. Can be useful in complex search scenarios where a user's needs aren't entirely clear from the outset.
	- To address specific data scenarios, recursive retrieval and multi-hop retrieval techniques are utilized together; Recursive retrieval involves a structured index to process and retrieve data in a hierarchical manner - this might include summarizing sections of a document or lengthy PDF  before performing a retrieval based on the summary. Subsequently, a secondary retrieval within the document refines the search, embodying the recursive nature of the process.
- ==[[Adaptive Retrieval]]==
	- Exemplified by [[FLARE]] and [[Self-RAG]]; Adaptive retrieval refines the RAG framework by enabling *LLMs* to actively determine the optimal moments and content for retrieval, thus enhancing the efficiency and relevance of the information sourced.
		- These methods are vaguely related to other methods in a trend wherein LLMs employ *active judgement* in their operations (eg in [[Toolformer]]).
	- Self-RAG introduces "reflection tokens" that allow the model to introspect its outputs. These tokens come in two varieties: "retrieve" and "critic." The model autonomously decides when the activate retrieval -- during retrieval, the generator conducts a fragment-level beam search across multiple paragraphs to derive the most coherent sequence. Critic scores are used to update the subdivision scores, with the flexibility to adjust these weights during inference ((Honestly these authors do not fuckign speak english)).

## Task and Evaluation
- Let's introduce the main downstream tasks of RAG, datasets, and how to evaluate the RAG system
- Downstream task
	- The core task of RAG remains [[Question Answering]], including both single and [[Multi-Hop]] QA. 
	- RAG has also been expanded into multiple downstream tasks, like information extraction (IE), dialogue generation, code search, etc.
- Evaluation Target
	- Assessment of RAG models historically have centered on their execution in specific downstream tasks.
		- QA evaluations might rely on EM and F1 Scores, whereas fact-checking might hinge on Accuracy. BLEU/ROUGE metrics are commonly used to evaluate answer quality.
	- Distinct characteristics of RAG models:
		- Retrieval Quality: Evaluating the retrieval quality is crucial in determining the effectiveness of the context sourced by the retriever component.  Standard metrics of search engines like Hit Rate, MRR, and NDCG are commonly utilized.
		- Generation Quality: Evaluation depends on the content's objectives: For unlabeled content, the evaluation encompasses faithfulness, relevance, and non-harmfulness.  For labeled content, the focus is on the focus is on the accuracy of the information produced by the model.
- Evaluation Aspects
	- ==Contemporary evaluation practices of RAG models emphasize three primary quality scores and four essential abilities.==, which collectively inform the evaluation of the two principal targets of the RAG model: retrieval and generation.
	1. Quality Scores
		- ==Context Relevance==: Evaluates the precision/specificity of retrieved context, ensuring relevance and minimizing processing cost associated with extraneous content.
		- ==Answer Faithfulness==: Ensures generated answers remain true to the retrieved context, maintaining consistency and avoiding contradictions.
		- ==Answer Relevance==: Requires that the generated answers are directly pertinent to the posed questions, effectively addressing the core inquiry.
	2. Required Abilities
		- ==Noise Robustness==: Appraises the model's capabilities to manage noisy documents that are question-related but lack substantive information. *How well can you ignore shitty documents?*
		- ==Negative Rejection==: Assesses the model's discernment in refraining from responding when retrieved documents don't contain necessary information to answer a question. *Can you not respond if you don't have the information to?*
		- ==Information Integration==: Evaluates the model's proficiency in synthesizing information from multiple documents to address complex questions. *Can you synthesize an answer from disparate documents?*
		- ==Counterfactual Robustness==: Tests the model's ability to recognize and disregard known inaccuracies within documents, even when instructed about potential misinformation. *Can you ignore known, incorrect information in the documents you're given?*
	- 

... Skipped the rest

# Paper Figures
![[Pasted image 20240515211757.png]]

![[Pasted image 20240515213042.png]]
Above: Given a user query, we index the query and compare it to indexed chunk vectors of document chunks. We retrieve some relevant document chunks, combine them with our user query and system prompt, and produce an answer using our LM.

![[Pasted image 20240515214835.png]]
Above: Comparing Naive RAG, Advanced RAG, and Modular RAG. 
- Naive RAG is mostly ==indexing, retrieval, and generation==
- Advanced RAG proposes multiple optimizations around *==pre-retrieval==* and *==post-retrieval==*, but still follows a chain-like structure similar to Naive RAG.
- Modular RAG showcases greater flexibility overall, evident in the introduction of *multiple* specific functional modules, and the replacement of existing modules. Not limited to sequential retrieval and generation; includes methods like *==iterative retrieval==* and *==adaptive retrieval==*.

![[Pasted image 20240516000350.png]]
Above: Showcases Iterative retrieval, Recursive retrieval, and Adaptive retrieval
- Iterative: Involves alternating between retrieval and generation
- Recursive: Involves gradually refining the user query and breaking the problem into subproblems, then continuously solving complex problems through retrieval and generation.
- Adaptive: Focuses on enabling the RAG system to autonomously determine determine whether external knowledge retrieval is necessary and when to stop retrieval and generation, often utilizing LLM-generated special tokens for control.

![[Pasted image 20240516011314.png]]
