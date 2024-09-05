April 24, 2024
[[Microsoft Research]] (*Edge et al.*)
[From Local to Global: A Graph RAG Approach to Query-Focused Summarization](https://arxiv.org/abs/2404.16130#)
#zotero 
Takeaway: ...


---

## Introduction
- Regular [[Retrieval-Augmented Generation|RAG]] often makes the assumption that answers are contained locally within regions of text whose retrieval provides sufficient grounding for the generation task.
- Instead, a more appropriate task is ==query-focused summarization== (QFS), in particular query-focused *abstractive* summarization, that generates natural language summaries and not just concatenated excerpts.
- While LLMs have made abstractive summarization "easy," there's still a challenge for query-focused abstractive summarization *over an entire corpus,* which will exceed the limits of even the longest LLM context windows (not to mention the *lost in the middle* problem).
- Our Graph RAG approach is based on ==global summarization of an LLM-derived knowledge graph==
	- In contrast to other papers that exploit the structured retrieval/traversal affordances of graph indexes, we focus instead of their inherent modularity, and the ability of ==community detection algorithms== to partition graphs into modular communities of closely-related nodes. ((Okay, so they're just doing more clustering than traversing?))
	- They use LLM-generated summaries as community descriptions. 
	- Query-focused summarization of an entire corpus is the made possible using a map-reduce approach:
		1. First, using each community summary to answer the query independently, and in parallel.
		2. Then, summarizing all relevant partial answers into a final global answer.
- "We show that all global approaches outperform naïve RAG on comprehensiveness and diversity, and that Graph RAG with intermediate- and low-level community summaries shows favorable performance over source text summarization on these same metrics, at lower token costs."

## Graph RAG Approach and Pipeline

==Source Documents -> Text Chunks==
- A fundamental design choice is the granularity at which we chunk input documents.
	- Longer text chunks require fewer LLM calls for extraction but suffer from recall degredation of longer LLM context windows.
	- Any extraction process needs to balance recall and precision for target activity.

==Text Chunks -> Element Instances==
- Identify and extract instances of graph nodes and edges from each chunk of source text, using a multi-part LLM prompt (with few-shot examples) that:
	1. First identifies all *entities* in the text, including name/type/description.
	2. Defines all *relationships* between clearly-related entities, including source/target entities and a brief description of the relationship.
- We also support a secondary extraction prompt for any additional covariates we would like to associate with extracted node instances... Our default covariate prompt aims to extract *claims* linked to detected entities, including subject/object/type/description/source text span/start and end dates.
- We use multiple rounds of "gleanings" to encourage the LLM to detect any additional entities it may have missed on prior extraction rounds.
	- If, after self-critique, the LLM responds that entities were missed, then a continuation prompts the LLM to glean these missing entities. 
		- ((It's a form of self-correction, and makes me worry, with respect to the [[Large Language Models Cannot Self-Correct Reasoning Yet]] paper's claims.))

==Element Instances -> Element Summaries==
- To convert instance-level summaries into single blocks of descriptive text for each graph element (node, relationship edge, claim covariate) requires a further round of LLM summarization over matching groups of instances.
- A potential concern is that the LLM might not consistently extract references to the same entity, resulting in duplicate entity elements.
	- Since all closely-related "communities" of entities will be detected and summarized in the following step, this approach is pretty resilient to this.
(This step might result in a homogenous undirected weighted graph in which each entity nodes are connected by relationship edges, with edge weights representing the normalized count of detected relationship instances)

==Element Summaries -> Graph Communities==
- Given the undirected weighted graph from the previous step, use a community detection algorithm to partition the graph into communities of nodes.
- In our pipeline, we use ==Leiden== (2019) to recover a hierarchical community structure, ==where each level of the hierarch provides a community partition that covers the nodes of the graph in a mutually-exclusive, collectively-exhaustive way, enabling divide-and-conquer global summarization.==

==Graph Communities -> Community Summaries==
- Next, we create report-like summaries of each community in the Leiden hierarchy, using a method designed to scale to very large datasets.
	- A user may scan through community summaries at one level, looking for general themes of interest, then follow links to reports at the lower level that provide more details for each of the subtopics.
	- Here, we focus on ==their utility as part of a graph-based index used for answering global queries==.
- Community summaries can be generated in the following ways:
	1. *Leaf-level Communities*: Element summaries of leaf-level community (nodes, edges, covariates) are prioritized and iteratively added to LLM context window until token limit is reached... Then for each community edge in decreasing order of combined source and target node degree (ie overall prominence), add descriptions of the the source node, target node, linked covariates, and the edge itself.
		- ((This isn't clear to me at all why they have prioritization, or why they're stuffing the context window rather than making multiple calls))
	2. *High-level Communities*: If all element summaries fit within the token limit of the context window, proceed as for leaf-level communities and summarize all element summaries within the community. Otherwise, rank sub-communities in decreasing order of element summary tokens, and iteratively substitute sub-community summaries (shorter) for their associated elements summaries (longer) until they fit within the context window.

==Community Summaries -> Community Answers -> Global Answer==
- Given a user query... it can be answered using the community summaries generated in the previous steps
	- The hierarchical nature of community structure also means that questions can be answered using community summaries from different levels, raising the question of whether a particular level in the hierarchical community structure offers the best balance of summary detail and scope.
- For a given community level, global answer to any user query is generated as follows:
	1. ==Prepare community summaries==: Community summaries are randomly divided and ***shuffled*** into chunks of pre-specified token size.
	2. ==Map community answers==: Generate intermediate answers in parallel, one for each chunk. 
		- LLM is also asked to generate a score between 0-100 indicating how helpful generated answer is in generating the target question. Answers with score 0 are filtered out.
		- ((The score bit seems weak))
	4. ==Reduce to global answer==: Intermediate community answers are sorted in descending order to helpfulness score and iteratively added into a new context window until the token limit is reached. The final context is used to generate the global answer returned to the user.


## Evaluation
- Authors used two datasets in the 1M token range, each equivalent to about 10 novels of text:
	- Podcast Transcripts (Podcast conversations of Kevin Scott, MSFT CTO)
	- News Articles (Benchmark dataset of news articles from Sep 2013-  Dec 2023 in a range of categories)
- To generate questions, we used an "activity-centered approach" to automated the generation of questions:
	- We asked the LLM to identify N potential users and N tasks per user, and then for each (user, task) combination, we asked the LLM to generate N questions that require understanding of the entire corpus. Using N=5, we got 125 test questions per dataset. (See figure for example of (user, task, questions)).
- We compare six different conditions in our analysis, including Graph RAG using four levels of graph communities (C0, C1, C2, C3), a text summarization method applying our map-reduce approach directly to source texts (TS), and a na ̈ıve “semantic search” RAG approach (SS):
	- CO. Uses root-level community summaries (fewest in number) to answer user queries.
	- C1. Uses high-level community summaries to answer queries. These are sub-communities of C0, if present, otherwise C0 communities projected down.
	- C2. Uses intermediate-level community summaries to answer queries. These are subcommunities of C1, if present, otherwise C1 communities projected down.
	- C3. Uses low-level community summaries (greatest in number) to answer queries. These are sub-communities of C2, if present, otherwise C2 communities projected down.
	- TS. The same method as in subsection 2.6, except source texts (rather than community summaries) are shuffled and chunked for the map-reduce summarization stages.
	- SS. An implementation of na ̈ıve RAG in which text chunks are retrieved and added to the available context window until the specified token limit is reached.
- Used head-to-head measures computed using an LLM evaluator:
	- ==Comprehensiveness==. How much detail does the answer provide to cover all aspects and details of the question?
	- ==Diversity==. How varied and rich is the answer in providing different perspectives and insights on the question
	- ==Empowerment==. How well does the answer help the reader understand and make informed judgements about the topic?
	- ==Directness==. How specifically and clearly does the answer address the question?
Global approaches consistently outperformed the naïve RAG (SS) approach in both comprehensiveness and diversity metrics across datasets

## Related Work
- Advanced RAG systems includes ==pre-retrieval==, ==retrieval==, and ==post-retrieval== strategies designed to overcome drawbacks of Naive RAG.
- Modular RAG systems include patterns for iterative and dynamic cycles of interleaved retrieval and generation.
- Other systems have combined concepts of multi-document summarization and multi-hop question answering... 
- Use of Graphs in RAG is a developing research area with multiple directions already established...
	- LLM for knowledge graph creation
	- LLM for knowledge graph completion
	- Forms of advanced RAG where index is a knowledge graph
	- A variety of graph databases are supported by both the LangChain and LlamaIndex libraries...


## Discussion and Conclusion
- Limitations of evaluation approach: 
- Trade-offs of building a graph index: In the real world, the decision about whether to invest in building a graph index depends on multiple factors, including compute budget, expected # of lifetime queries per dataset, and value obtained from other aspects of the graph index.
- Future work: Hybrid RAG schemes that combine embedding-based matching against community reports, become employing our map-reduce summarization mechanisms?


Abstract
> The use of retrieval-augmented generation (RAG) to retrieve relevant information from an external knowledge source enables large language models (LLMs) to answer questions over private and/or previously unseen document collections. However, ==RAG fails on global questions directed at an entire text corpus==, ==such as "What are the main themes in the dataset?",== since this is inherently a ==query-focused summarization (QFS) task==, rather than an explicit retrieval task. Prior QFS methods, meanwhile, fail to scale to the quantities of text indexed by typical RAG systems. To combine the strengths of these contrasting methods, we propose a Graph RAG approach to question answering over private text corpora that scales with both the generality of user questions and the quantity of source text to be indexed. Our approach ==uses an LLM to build a graph-based text index== in two stages: ==first to derive an entity knowledge graph from the source documents==, ==then to pregenerate community summaries for all groups of closely-related entities.== Given a question, ==each community summary is used to generate a partial response==, ==before all partial responses are again summarized in a final response to the user==. For a class of global sensemaking questions over datasets in the 1 million token range, we show that Graph RAG leads to substantial improvements over a naïve RAG baseline for both the comprehensiveness and diversity of generated answers. An open-source, Python-based implementation of both global and local Graph RAG approaches is forthcoming at [this https URL](https://aka.ms/graphrag).


# Paper Figures
![[Pasted image 20240808143625.png|500]]

![[Pasted image 20240808155056.png|500]]

![[Pasted image 20240808162457.png|500]]
Questions that target a global understanding from diverse documents, rather than any specific detail: This is the ==query-focused summarization (QFS)== task

![[Pasted image 20240808165722.png|500]]

![[Pasted image 20240808165732.png|500]]

![[Pasted image 20240808165743.png|500]]

