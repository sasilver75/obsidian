---
aliases:
  - QA
---
A task in NLP focused on building systems that can answer questions posed in natural language. A key component in many systems, with examples include virtual assistants, research assistants, medical diagnosis support, etc. 

Components of a QA system:
- Question Analysis: Understanding the type and intent of the question
- Information Retrieval: Finding relevant information sources
- Answer Extraction: Identifying and formulating the answer
- Answer Generation: Creating a natural language response

Variants:
- [[Open-Domain]] QA: Systems aiming to be generalist in their domain knowledge
- [[Closed-Domain]] QA: Systems specialized for specific areas (eg medical diagnosis, legal advice)
- [[Multi-Hop]] QA: A more complex form of question answering requiring reasoning over *multiple pieces of information* to arrive at a correct answer (perhaps querying a knowledge base multiple times in series).
- "Open Book" QA: System has access to external information sources
- "Closed Book" QA: System has no access to external information sources
- [[Multimodal]] QA: Passages/Documents include non-text elements

Relevant datasets and benchmarks include:
- [[SQuAD]]
- [[Natural Questions]]
- TriviaQA
- [[HotpotQA]]


![[Pasted image 20240425204626.png]]
![[Pasted image 20240501215717.png]]
Above: From the ORQA paper

![[Pasted image 20240516011211.png]]
Above: From [[Retrieval-Augmented Generation for Large Language Models - A Survey]]