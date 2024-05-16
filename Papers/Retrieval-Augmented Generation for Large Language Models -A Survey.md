*Retrieval-Augmented Generation for Large Language Models: A Survey (Gao et al.; March 27, 2024)* [Link](https://arxiv.org/abs/2312.10997v5) #zotero 

Abstract
> Large Language Models (LLMs) showcase impressive capabilities but encounter challenges like ==hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes==. Retrieval-Augmented Generation (==RAG==) has emerged as a promising solution by incorporating knowledge from external databases. This ==enhances the accuracy and credibility of the generation==, particularly for knowledge-intensive tasks, and ==allows for continuous knowledge updates and integration of domain-specific information==. RAG synergistically merges LLMs' intrinsic knowledge with the vast, dynamic repositories of external databases. This comprehensive review paper offers a detailed examination of the progression of RAG paradigms, encompassing the ==Naive RAG==, the ==Advanced RAG==, and the ==Modular RAG==. It meticulously scrutinizes the tripartite foundation of RAG frameworks, which includes the ==retrieval==, the ==generation== and the ==augmentation== techniques. The paper highlights the state-of-the-art technologies embedded in each of these critical components, providing a profound understanding of the advancements in RAG systems. Furthermore, this paper introduces up-to-date evaluation framework and benchmark. At the end, this article delineates the challenges currently faced and points out prospective avenues for research and development.


Notes:
- The limitations of LMs include hallucination, outdated knowledge, and non-transparent, untraceable reasoning processes. RAG aims to solve each of them.
- The authors include a "technology tree" of RAG, which is kind of sick for the gamers out there.
- Authors note that this paper summarizes three main research paradigms from over 100 RAG studies, and analyzes techniques across the three core stages of "Retrieval," "Generation," and "Augmentation."
- Authors note that there are three paradigms of "naive RAG," "advanced RAG," and "modular RAG".
- ==Naive RAG==
	- Represents the earliest methodology that gained widespread adoption shortly after ChatGPT; it follows a traditional process that includes indexing, retrieval, and generation, which is also characterized as a "==Retrieve-Read" framework==
	- *Indexing*: Starts with the cleaning and extractions of raw data in diverse formats like PDF, HTML, Word, and Markdown, which is then converted into a uniform plain-text format. To accommodate the context limitations of language models, text is segmented into smaller, digestible chunks. Chunks are then encoded into vector representations using an embedding model and stored in a vector database.
	- *Retrieval*: Upon receipt of a user query, the RAG system employs the *same encoding model* utilized during the indexing phase to transform the query into a vector representation. It then computes similarity scores between the query vector and the vector of chunks within the indexed corpus. The system prioritizes and retrieves the top K chunks that demonstrate the greatest similarity to the query. These chunks are subsequently used as the expanded context in prompts.
	- *Generation*: The posed query and selected documents are synthesized into a coherent prompt to which a large language model is tasked with formulating a response.


# Paper Figures
![[Pasted image 20240515211757.png]]

![[Pasted image 20240515213042.png]]
