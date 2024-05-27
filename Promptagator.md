September 23, 2022
[[Google Research]]
Paper: [Promptagator 🐊: Few-Shot Dense Retrieval from 8 Examples](https://arxiv.org/pdf/2209.11755v1)
#zotero 
Takeaway: This paper is really about adding query domain/task-specific *intent* via LLM generation of query-document pairs, given ~8 exemplars per task. Argues that different retrieval tasks have very different search intents (retrieving entities, finding evidence, etc.), so they propose a few-shot setting for dense retrievers where each task comes with a short description and a few annotated examples to illustrate search intents. They use these examples to create a large set of synthetic examples, using FLAN, with the quality of these generated queries improved by ensuring "round-trip consistency"/consistency filtering, i.e. that the query should retrieve its source passage.

Consistency Filtering: They first train their retrieval model using augmented data, then feeding this augmented data back into the model, and, given a query, checking if the relevant document appears in the top-k. If it doesn't then, they throw out the query/document pair.

The goal is to not have to bring a lot of task/domain-specific annotation of query-document pairs to every new retrieval challenge. The goal is for synthetic data generation to create annotated pairs as well as a human could!

Related paper on creating synthetic data for information retrieval: [[InPars]]

----

Notes:
- The recently proposed [[BEIR]] heterogenous retrieval benchmark showed that it's still difficult for neural retrievers to perform well on a wide variety of tasks without having dedicated training data. To best transfer from QA datasets, expressive retrievers are developed that allow fine-grained token-level interaction such as [[ColBERT]]. and [[SPLADE]], but with higher inference cost, or training models on enormous large multi-domain QA retrieval datasets (that will never cover all bases).
	- ((Interesting that they say higher inference cost here; my understanding what that the inference cost/latency on ColBERT was much lower than bi-encoder alternatives?))
- Problem: Different retrieval tasks have very different *search intents*, with different definitions of what "relevance" means, even if two retrieval tasks share the same domain, and different tasks have different distributions of queries, even when search intents are similar (eg long compositional questions vs short financial questions)
	1. Ddpedia-Entity (Entity search) is a task to retrieve entities that are mentioned in the query
	2. FEVER (Claim verification) is a task to find evidence that either supports or refutes a given statement
	3. ArguAna (Counter-argument retrieval) is a task to retrieve counter-arguments
	- ((Given this description, it's obvious that a transferrable multi-task retrieval system *must* also be conditioned on a prompt that describes what relevance looks like for that retrieval task, right? But even then, are human-written prompts *enough* to really describe the granular nature of what "good retrieval" looks like for a task?))
- This paper works on the setting of ==Few-shot Retrieval== for diverse retrieval, where each task comes with a short description and a few annotated examples to clearly illustrate the search intents. 
	- We resolve the data scarcity issue while retaining the efficiency of a small dual encoder by harnessing the power of a LLM like FLAN as a query generator without fin-tuning, relying solely on a few supervised examples from the target task rather than a large number of annotated query-document pairs.
	- The key insight of Promptagator is to amplify the power of few-shot examples by creating ==task-specific prompting==, which in turn enables generation of a large set of synthetic queries for training retrievers suited to the task. We combine this with a filtering technique to ensure round-trip consistency using generated data only, which removes ambiguous, generic, and low-quality questions, significantly improving retrieval performance.
- Authors define a retrieval task as $T = {\{D,Q,I\}}$ , where the task is defined by the document corpus ($D$), query distribution ($Q$), and underlying search intent for the task ($I$). Importantly, for the same pair of ($Q,D$), document relevance can be completely different under different search intents (eg some retrieval tasks look for supporting arguments, while other tasks need to retrieve *counter* arguments).
- The key idea of Promptagator is to transform the few examples into many more examples by prompting an LLM, instead of using them to train a retriever directly.
- Promptagator consists of three components:
	1. ==Prompt-based query generation==: A task-specific prompt will be combined with an LLM to produce queries for the target task.
	2. ==Consistency filtering==: Cleans the generated data based on round-trip consistency (surprisingly a retriever trained only on the synthetic data can be used to filter the synthetic data)
	3. ==Retriever training==: A retrieval (here, dual encoders) and a cross-attention reranker will be trained based on the generated data.
- ==Prompt-based query generation==
	- We create a large set of synthetic ($q,d$) examples, amplifying information from a few examples into a large synthetic dataset whose query distribution is similar to true task distribution $Q_T$ and query-document pairs that convey the true search intent $I_T$.
	- We use [[FLAN]] 137B as the LLM for query generation, using at most 8 examples (less, if thy exceed the input length limit of FLAN). ![[Pasted image 20240526111014.png]]
- ==Consistency filtering==
	- The filtering step improves the quality of generated queries by ensuring the round-trip consistency: a query should be answered by the passage from which the query was generated.
	- Consistency filtering (Albert et al., 2019 & Lewis et al., 2021) has been shown crucial for synthetic question generation, but these techniques typically rely on an *external* question-answering model as the filter, trained on existing supervised QA data. Surprisingly, we find that consistency filtering based on generated data alone can work well over the different search intents observed in BEIR.
		- We use the *generated* query-document pairs to train an initial retriever. Given a synthetic ($q,d$) pair, we use the initial retriever to predict the most relevant passages for $q$; we keep $q$ only when $d$ occurs among the Top-K passages returned by the retriever.
- ==Retriever training==
	- Our synthetically generated data allows training task-specific retrievers for tasks where supervised in-domain fine-tuning is challenging due to data scarcity.
	- We use the standard dual-encoder retrieval architecture and we propose a simple pretrain/fine-tune recipe. We use use the Transformer encoder from a [[T5]] checkpoint. We pretrain our retriever on [[C4]] with the independent cropping task from [[Contriever]], where we treat two random crops from the same document as positive retrieval pairs, and train with a cross-entropy loss over in-batch random negatives. Next, we finetune the dual-encoder on the query-document pairs generated from our prompt-base QGen, again with cross-entropy loss over in-batch random negatives. After training for a set number of epochs, we apply round-trip filtering on our synthetic data using this initial dual encoder, and continue to fine-tune the dual encoder on the filtered data.
	- We also propose ==Promptagator++==, a reranker trained on the same synthetic data generated from our prompt-base QGen, which refines the retrieved candidates using a slower but more accurate cross-attention model. We train the reranker using a cross-entropy loss with 31 sampled negatives from top 200 passages retrieved by the Promptagator retriever, which approximates the inference time distribution (reranking top 200 from the retriever).
	- The prompt-based query generation can also run in a *zero-shot* manner, where we universally apply the following prompt, irrespective of the target task (`f'{d} Read the passage and generate a query'`), where ${d}$ denotes the document text. We train retrievers and rerankers on the zero-shot prompt generated data, leading to zero-shot Promptagator and zero-shot Promptagator++. Still, few-shot Promptagator markedly improves on the zero-shot version.
- Results
	- Few-shot Promptagator outperforms strong baselines like GenQ and GPL, which also use query generation to augment training data, as well as both [[ColBERTv2]] and [[SPLADEv2]], which rely on token-level interaction architectures and distillation recipes.
- Related Work
	- Neural retrieval models
		- ==Representation-based models== encode a query and passage independently into a common dense space, and scores their relevance based on vector dot-product or cosine similarity.
			- Recent research has focused on developing better pre-training tasks or pre-training architectures, improving expressiveness using multi-vector representations, improving negative contrast, and improving generalization across different domains. Different techniques have been explored to improve generalization, such as using query generation for data augmentation, using contrastive learning for better pre-training, using knowledge distillation, and scaling model size.
		- Although encoding the query/document into a single vector enables fast retrieval via ANN search, it also constrains the representation power; ==Interaction-based models== in contrast explicitly model the interaction between query and document terms. These models are typically more expensive, and thus are used for reranking or rescoring. ==Distilling interaction-based models into representation based models has been shown effective in closing the gap between the two==. Another attempt is by *postponing* the interaction until the last layer of the model, blurring the boundary between representation and interaction models.
	- Few-shot learning
		- Two approaches are commonly used
			- One approach is to provide the LLM an instruction of the task in natural language with a few examples, and don't update any parameter of LLM (Promptagator uses this)
			- The other approach provides the LLM with the instruction, a few examples, and also performs model fine-tuning.
	- Prompt-based Query Generation
		- We showed that the quality of generated data can be improved by task-specific prompts and consistency filtering.
	- Retrievers with late interaction
		- While dual-encoder models are very efficient at retrieval due to the MIPS algorithms, their expressivity is limited due to the fact that their score is just a dot-product between query and document vector.
		- [[ColBERT]] and [[SPLADE]] increase the interactions between the query and document by allowing token-level interactions. Because these models are not just dot products between queries and documents, MIPS algorithms can't be used directly; hence, these models usually have much higher serving cost compared to dual encoders.


Abstract
> Much recent research on information retrieval has focused on ==how to transfer from one task (typically with abundant supervised data) to various other tasks where supervision is limited==, with the implicit assumption that it is possible to generalize from one task to all the rest. However, this overlooks the fact that there are many diverse and unique retrieval tasks, each targeting different search intents, queries, and search domains. In this paper, we suggest to work on Few-shot Dense Retrieval, a setting where each task comes with a short description and a few examples. To amplify the power of a few examples, we propose ==Prompt-base Query Generation for Retriever (Promptagator),== which ==leverages large language models (LLM) as a few-shot query generator==, and ==creates task-specific retrievers based on the generated data.== Powered by LLM's generalization ability, Promptagator ==makes it possible to create task-specific end-to-end retrievers solely based on a few examples== {without} using Natural Questions or MS MARCO to train %question generators or dual encoders. Surprisingly, LLM prompting with no more than 8 examples allows dual encoders to outperform heavily engineered models trained on MS MARCO like ColBERT v2 by more than 1.2 nDCG on average on 11 retrieval sets. Further training standard-size re-rankers using the same generated data yields another 5.0 point nDCG improvement. Our studies determine that query generation can be far more effective than previously observed, especially when a small amount of task-specific knowledge is given.


# Paper Figures
![[Pasted image 20240526092621.png]]

![[Pasted image 20240526101425.png]]

![[Pasted image 20240526103257.png]]

![[Pasted image 20240526103335.png]]


# Non-Paper Figures