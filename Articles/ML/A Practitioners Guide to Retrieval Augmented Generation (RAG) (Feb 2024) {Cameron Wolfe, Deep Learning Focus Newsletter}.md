#article 
Link: https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval

-----
![[Pasted image 20240224150211.png]]


LLMs have a notoriously poor ability to retrieve and manipulate the knowledge that they possess, which leads to issues like *==hallucination==*, *==knowledge cutoffs==*, and *poor ==understanding of specialized domains==*.

Is there a way that we can improve an LLM's ability to access and utilize high-quality information?
- We'll answer this by exploring *[[Retrieval-Augmented Generation]](RAG)*, a way to integrate LLMs with external data sources and supplement the model's knowledge with helpful+relevant information.


# What is Retrieval Augmented Generation?
- Before diving into the technical component of this overview, let's build a basic understanding of RAG, how it works, and why it's useful.
- LLMs contain a lot of knowledge in their pretrained weights ("==parametric knowledge==") that can be surfaced by prompting the model and generating output... but this information is only "fuzzily memorized" -- the model has a tendency to hallucinate and generate false information -- so we can say that ==specific parametric knowledge possessed by an LLM can be unreliable!== 
	- With RAG, we *augment the knowledge base* of an LLM by inserting relevant context into the prompt, and relying on the [[In-Context Learning]] abilities of the LLMs to produce better output by taking advantage of this context.

# The Structure of a RAG Pipeline
- The usual mode of interacting with an LLM involves us inserting a query (perhaps as part of a prompt template) and generating a response by the LLM.
- RAG modifies this approach by first using the input query to *search for relevant information within an external dataset!* Then, we add the info that we find to the model's prompt when generating output, allowing the LLM to use this context to generate a better and more factual response.
- ==By combining the LLM's parametric knowledge with incontext learning over relevant documents, we have the ability to generate a better and more factual response!==

![[Pasted image 20240224153401.png]]
- Above:
	- Ingest/chunk/index documents
	- Given prompt
	- Retrieve and rank relevant chunks
	- Inject query and relevant chunks into prompt

### Cleaning and Chunking
- RAG requires access to a dataset of correct and useful information to augment an LLM's knowledge base. 
- We must construct a pipeline that allows us to *search* for relevant data within this knowledge base!
- The external data sources that we're using might contain data in a variety of formats (pdf, markdown, more). As such, we must first ==clean== the data and extract the raw textual information from these heterogenous data sources.
- Then we can ==chunk== the data, splitting it into sets of shorter sequences (typically 100-500 tokens).

![[Pasted image 20240224153627.png]]
- The goal of chunking is to split the data into ==units of retrieval== (pieces of text that we can retrieve as search results).
	- An entire document could itself be a unit of retrieval, or it could be too large, and must be split.
	- The most common chunking strategy is a *==fixed-size chunking approach==*, which breaks longer texts into shorter sequences that contain a fixed number of tokens!
		- ((This seems obviously to not be the optimal strategy, since you could be splitting a semantic chunk across units of retrieval))
	- An alternative is to used ==variable-sized chunks -- for example if our data is naturally divided into chunks (eg social media posts, or product descriptions)==.


### Searching over chunks
- Once we've cleaned our data and split the data into units of retrieval, we must build a search engine for matching input queries to chunks!
	- We covered this prior in [[The Basics of AI-Powered Vector Search (Jan, 2024) {Cameron Wolfe Deep Learning Focus newsletter}]]

![[Pasted image 20240224154101.png]]
- First, let's build a RAG system by:
	1. Using an embedding model (e.g. a [[Bidirectional Encoder Representations from Transformers|BERT]]-based [[Bi-Encoder]]) to produce corresponding vector representations for each of our chunks
	2. Index all of these vector representations within a vector database. 
	3. Then, embed the input query using the same embedding model and ...
	4. Perform an efficient vector search ([[Dense Retrieval]]) to retrieve semantically related chunks
		- These vector searches can be augmented with a [[Sparse Retrieval|Lexical Retrieval]] component to create [[Hybrid Search]], which is going to make the pipeline much better.
	5. With the retrieved chunks, we can then perform ranking using a [[Cross-Encoder]] model
		- Alternatively, you could use a less-expensive component like [[ColBERT]] to sort candidate chunks based on relevance.
- 
![[Pasted image 20240224154712.png]]


### More data wrangling
- *After retrieval*, we might perform additional data cleaning on each textual chunk to compress the data or emphasize key information.
- ==Some practitioners add an extra processing step after retrieval that passes textual chunks through an LLM for summarization or reformatting prior== to feeding them to the final LLM.
- Practitioners have recently explored ==connecting LLMs to graph databases==, for instance -- forming a RAG system that can search for relevant information via queries to a graph database!
	- Other researchers have ==directly connected LLMs to search APIs like Google== or Serper to access up-to-date information!

![[Pasted image 20240224155528.png]]
Above: ((We can see that, given a prompt, we fetch relevant chunk(s) from our external data source, in this case, a search engine. It's possible that these chunks could then be further "upgraded" via another LLM, midway through the blue line, to "format" them such that the downstream LLM that consumes them in the prompt can make maximal use of them during in-context learning.))

# The benefits of RAG
- Implementing RAG allows us to specialize an LLM over a knowledge base of our choosing.
	- Compared to other knowledge-injection techniques (with fine-tuning being the primary alternative), RAG is both simpler to implement and computationally cheaper, while also producing better results!

#### (1/4) Reducing hallucinations
- The primary reason that RAG is so commonly used in practice is its ability to reduce hallucinations that occur when LLMs rely on their parametric knowledge.
- Reducing hallucinations is key to building trust among users.
- RAG provides us direct references to data that is used to generate information in the model's output.
#### (2/4) Access to up-to-date information
- When relying on parametric knowledge, LLMs typically have a knowledge cutoff date after which training data wasn't available.
- Using RAG, we can pull in documents with relevant semantic information that were posted *15 minutes ago*, if we care to!
- ((There's worries about data-poisoning/manchurian candidate LLMs because of data that was created on the internet after about 2020 or so.))
#### (3/4) Data Security
- When we add data into an LLM's training set, there's always a chance that the LLM will leak this data within its output. LLMs are vulnerable to *==data extraction attacks==* that can discover the contents of an LLM's pretraining dataset via prompting techniques!
	- But we can still specialize an LLM to such data using RAG, which mitigates the security risk by never actually training the model over proprietary data.
		- ((Okay... you've moved the information from the parametric weights into documents that are injected into the context -- but are there still other data extraction techniques that might be effective against RAG?))
#### (4/4) Each of implementation
- Finally, one of the biggest reasons to use RAG is the simple fact that the implementation is quite simple compared to alternatives like finetuning -- the core ideas from the original RAG paper rare like 5 lines of code, and there's no need to train the LLM itself.


# From the Origins of RAG to Modern Usage
- Many of the ideas used by RAG are derived from prior research on the topic of question answering.
- RAG was inspired by a "compelling vision of a trained system that had a retrieval index in the middle of it, so it could learn to generate any text output you wanted."

### Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (RAG Paper)
- RAG was first proposed in 2021, when LLMs were less-explored and Seq2Seq models were extremely popular.
- As we know, pretrained language models possess a lot of information within their parameters, but they have a notoriously poor ability to access and manipulate this knowledge base -- for this reason, the performance of language-model-based systems was far behind those of specialized extraction methods at the time.
#### How can RAG Help?
- The basic idea is to improve a prettrained LM's ability to access and use knowledge by connecting it with a non-parametric memory store --typically a set of documents or textual data over which we can perform retrieval.
- This approach provides extra (factual_ content) to the model and also allows us (the people using/training the model) to examine the results of retrieval and gain more insights into the LLM's problem-solving process.
- Setup details:
	- Formally, RAG considers an input sequence `x` (the prompt) and uses this input to retrieve documents `Z` (the text chunks), which are used as context when generating a target sequence `y`.
	- For retrieval, the authors used a pretrained bi-encoder that uses separate BERT mdoels to encode queries (query encoder) and documents (document encoder)
		- ((Interesting that they used separate encoders for queries and documents... is that how we expected this to work? Probably yes, I think.))
- Both the retriever and the generator are based on pretrained models, which makes finetuning optional -- the RAG setup already possesses the ability to retrieve and leverage knowledge via its pretrained components.

The data used for RAG is a Wikipedia dunk that's chunked into sequence of 100 tokens. ==The chunk size used for RAG is a hyperparameter that must be tuned depending on the application==. Each chunk is converted to a vector embedding using DPR's pretrained document encoder.

Training with Rag
- The dataset used to train the RAG model consists of pairs of input queries and desired responses.
- When training the model, we first embed the input query using the query encoder of DPR and perform a nearest-neighbor search within the document to provide the K most similar textual chunks.
	- ((Performing approximate/fuzzy kNN, using the embedded query vs embedded chunks))
- From here, we can concatenate a textual chunk with the input query and pass this concatenated input to BART to generate an output.

![[Pasted image 20240224162555.png]]
The model in their paper takes only a single document as input when generating output with BART, so we must ==marginalize== over the top K documents when generating text, meaning that we predict a distribution over generated text using each individual document... In other words, we run a forward pass of BART with each of the different documents used as input.
- Then we take a *weighted sum* over the model's outputs (each output is a probability distributed over generated text)
- This document probability is derived from the retrieval score (eg cosine similarity) of the document.

((So recapping... given that  their model for some reason could only take in one document, they did the usual think of retrieving (say) k=5 documents. Each of those documents have some *retrieval score* of some sort that was used to retrieve them... maybe it's the cosine similarity. For each of the five documents, we inject them into the prompt and do a forward pass prediction of the LLM to generate a probability distribution over tokens. We take retrieval-score-weighted average of each of these 5 forward passes, and use that probability distribution.

The paper proposes ==two methods of marginalizing over documents==:
1. ==Rag-Sequence==: The same document is used to predict each target token ((? Is this saying choose one document and stick with that for the entire sequence generation? This feels like the "normal"/intuitive way of doing it.))
2. ==RAG-Token==: RAG-Token: Each target token is predicted with a different document ((? Is this saying at every token in the generation, you make a choice over which document you want to inject into the prompt? Seems noisy.))

At inference time, we can generate an output sequence using either of these approaches, together with a modified form of [[Beam Search]]!
- To train the model, we use a standard language modeling objective that maximizes the log probability of the target output sequence.


How does it perform?
- Well! Sets new SoTA performance on open domain question answering, and near-SoTA on abstractive question answering tests.


# Using RAG in the age of LLMs
![[Pasted image 20240224164158.png]]
- RAG is still heavily used today to improve the factuality of modern LLMs.
- The structure of RAG used for LLMs is shown within the figure above...
	- ((We have a document corpus, and we chunk it using a variety of strategies. We embed these chunks. Later, we have a query. We embed the query and perform some similarity search to find relevant chunks. We add the retrieved chunks to the prompt! (Optionally, we could have done additional reranking or "upgrading" of the chunks before putting them into the prompt.)))

The main differences between this approach above and those as introduced in the original paper (previous section) are:
1. ==Finetuning is optional and oftentimes not used. Instead, we rely on the in-context learning abilities== of the LLM to leverage the retrieved data.
2. Due to the large context windows present in most LLMs, ==we can pass in several documents to the model's input== at once when generating a response.
3. Additionally, ==we often use a [[Hybrid Search]] algorithm== rather than pure vector search ([[Dense Retrieval]])! We use everything we know about AI-powered search to craft the best RAG pipeline.

Let's go over more recent research that builds on the past and applies this RAG framework to modern, generative (decoder-only) LLMs!


#### How Context Affects LLMs' Factual Predictions
- Pretrained LLMs have factual information fuzzily encoded in their parameters, but there are limitations with leveraging this knowledge base.
	- Pretrained LLMs tend to struggle with storing and extracting (or manipulating) knowledge in a reliable faction -- that ==hallucinate!==
	- Using RAG, we can mitigate these issues by injecting reliable and relevant knowledge directly into the model's input.


#### [[Gorilla]]: Large Language Models connected with Massive APIs
- Combining language models with external tools is a popular topic in AI research -- these techniques usually teach the underlying LLM to leverage a small, fixed set of potential tools (e.g. a calculator or search engine) to solve problems.
- In contrast, authors develop a retrieval-based finetuning strategy to train an LLM, called Gorilla, to use over 1,600 different deep learning model APIs (eg from HuggingFace or TensorFlow hub).
![[Pasted image 20240225120223.png]]
- First, the documentation of all these different deep learning models is downloaded.
- Then, a [[Self-Instruct]] approach is used to generate a finetuning dataset that pairs questions with an associated response that *leverages a call to one of the relevant APIs*. From here, the model is finetuned over this dataset in a *retrieval-aware manner*, in which a pretrained information retrieval system is used to retrieve the documentation of the most relevant APIs for solving each question... This documentation is then passed into the model's prompt when generating output, thus teaching the model to leverage documentation of retrieved APIs when solving and generating API calls.
![[Pasted image 20240225120454.png]]
Unlike most RAG applications, Gorilla is actually finetuned to better leverage its retrieval mechanism!


## Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs
- In this study, authors study the concept of *==knowledge injection==* which refers to methods of incorporating information from an external dataset into an LLM's knowledge base.
- Two basic ways of injecting knowledge into a pretrained model are:
	1. Finetuning (i.e. continued pretraining)
	2. RAG

![[Pasted image 20240225121734.png]]
![[Pasted image 20240225121747.png]]
- Above: Interestingly, we see that RAG far outperforms finetuning, but combining finetuning with RAG doesn't consistently outperform RAG alone, thus revealing the impact of RAG on the LLM's factuality and response quality
	- ((It depend on your test, and how much "reasoning" is required, given the document chunks. If you're recalling documents that just have the answer in them, you could basically use any LLM to retrieve the answer, and the two RAG options above would be identical in accuracy. If it still requires some multi-hop reasoning, etc. then perhaps finetuning could help.))

# ==RAGAS==: Automated Evaluation of Retrieval Augmented Generation
- RAG is an effective tool for LLM applications, but the approach is difficult to evaluate, since there are many dimensions of "performance" to consider:
	1. Ability to identify relevant documents
	2. Properly exploiting data in the retrieved documents via in-context learning
	3. Generating a high-quality, grounded output 
RAG isn't just a retrieval system, but rather a multi-step process of finding useful information and leveraging it to get better outputs..

The authors propose ==Retrieval Augmented Generation Assessment (RAGAS)== to evaluate these complex RAG pipelines without any human-annotated datasets or reference answers.

Three classes of metrics are used to evaluate RAG pipelines:
1. ==Faithfulness==: Is the answer *grounded* in the given context?
	- Can be evaluated by prompting an LLM to extract a set of factual statements from the generated answer, and then prompting an LLM again to determine if each of the statements can be inferred from the provided context.
2. ==Answer Relevance==: Does the answer address the provided question?
	- (The evaluation of this can similarly be automated, as above)
3. ==Context Relevance==: The context is focused and contains as little irrelevant information as possible.
	- (The evaluation of this can similarly be automated, as above)
((==In other words, do you (1) use provided documents to (2) answer the question with (3) as little extra information as possible?==))

Together, these metrics holistically characterize the performance of any RAG pipeline.

![[Pasted image 20240225154758.png]]
((Above: The example of how we could automatically calculate some stuff on faithfulness, which is about whether the answers are grounded in the provided documents. We can do this by using an LM to extract a collection of factual statements from the response, and then again use an LM to determine if these factual statements can be inferred from the used documents.))

These tools from the RAGAS paper are now quite popular among LLM practitioners.

# Practical tip of RAG applications
- Practitioners, rather than academics, generally have the best takeaways for how to successfully use RAG, so most of the best tips for how to do RAG effectively are hidden within blog posts, discussion forums, and other non-academic publications.

### RAG is a Search Engine!
- When applying RAG in practical applications, we should realize that the retrieval pipeline used for RAG is just a *search engine!*
- Namely, the same retrieval and ranking techniques that have been used by search engines for years can be applied by RAG to find more relevant textual chunks.

#### Don't just use Vector Search
- Many RAG systems *purely* leverage [[Dense Retrieval]] methods to find relevant textual chunks. This approach is quite simple, as we can just:
	1. Generate an embedding for the input prompt
	2. Search for related embedding-indexed chunks in our vector database.
- ==However, semantic search has the tendency to yield false positives, and may have noisy results!==
- To solve this, we perform [[Hybrid Search]] using a *combination* of vector search and [[Sparse Retrieval|Lexical Retrieval]] -- just like a normal search engine does!
- The approach to vector search doesn't change, but we can perform a parallel lexical search by
	1. Extracting *keywords* from the input prompt.
	2. Performing a *lexical search* with these keywords (looking for overlap)
	3. Taking a ==weighted combination of results from lexical/vector search==.

By performing hybrid search, we can make our RAG pipeline more robust and reduce the frequency of irrelevant chunks in the model's context...
- ==Plus, adopting keyword-based search lets us perform clever tricks like *promoting* documents with important keywords, or *excluding* documents with negative keywords... or even augmenting documents with *synthetically generated data* ((tags?)) for better matching==! ((Though you could do this for either Dense or Sparse retrieval... though it's probably more controllable for the Sparse/Lexical retrieval version)).

#### Optimizing the RAG pipeline
- To improve our retrieval system, we need to collect metrics that allow us to evaluate its results similarly to any normal search engine.
	- We could expose the actual textual chunks used for certain generations to the user, as part of the response! As part of this, we could then prompt the user to provide binary feedback (+/-) as to whether the information was absolutely relevant! ((Human-in-the-loop-rating of retrieval))
	- We can also evaluate the results of our retrieval system using traditional search metrics like [[Discounted Cumulative Gain|DCG]]/nDCG, test changes to our system via AB tests, and iteratively improve our results!

Evaluations for RAG must go beyond simply verifying the results of *retrieval* -- even if we retrieve the right documents, the generated outcome may not be correct. To evaluate the *generation* component of a RAG-included system, the AI community relies heavily on automated metrics like RAGAS and LLM-as-a-Judge, which perform evaluations by prompting LLMs like GPT-4 to score the results.


#### Improving over time
- Once we've built a proper retrieval pipeline and can evaluate the end-to-end RAG system... the last step of applying RAG is to perform iterative improvements using a combination of better models and data!
	1. ==Add ranking to the retrieval pipeline using either a [[Cross-Encoder]] or a hybrid model== that performs both retrieval *and* ranking (eg [[ColBERT]]).
	2. ==Finetune the embedding model== for dense retrieval over human-collected relevance data (ie pairs of input prompts with relevant/irrelevant passages... ((Like siamese/triplet loss?)))
	3. ==Finetune the LLM generator over examples of high-quality outputs== so that it learns to better follow instructions and leverage useful context ((Though I thought we saw above that this didn't give a large boost if you were already using a capable base model))
	4. Using LLMs to ==augment either the input prompt or the textual chunks with extra synthetic data== to improve retrieval.

For each of these changes, we have to measure their impact over historical data in an offline manner ((backtest it))

### Optimizing the Context Window
- Successfully applying RAG isn't just a matter of retrieving the correct context -- prompt engineering also plays a massive role! Once we have the relevant data, we need to craft a prompt that:
	1. Includes this context
	2. Formats it in a way that elicits a grounded output from the LL
- Within this section, we'll investigate a few strategies for crafting effective prompts with RAG:

#### RAG needs a larger context window
- The choice of sequence length during pretraining determines the model's context length! Recently we've seen LLMs with longer and longer context lengths, with MPT-StoryWriter-65k, Claude-2.1, and GPT-4-Turbo having lengths of 65k, 200k, and 128k tokens, respectively ((and Gemeni 1.5 supporting > 1 million!))
	- For reference, ==The Great Gatsby contains ~ 70k tokens==
- Though not all LLMs have a large context window, RAG requires a model with a large context window so that we can include a sufficient number of textual chunks in the model's prompt.

#### Maximizing Diversity
- Once we've been sure to select an LLM with a large enough context length, next we need to determine how to select the best context to include in the prompt.
- We can optimize the prompt by adding a specialized ==selection component== that sub-selects the results of retrieval!
	- Selection doesn't change the *retrieval* process of RAG -- rather, the selection is added to the *end*  of the retrieval pipeline, ==after relevant chunks of text have been identified and ranked== to determine how documents can best be *sub-selected and ordered* within the resulting prompt.

One popular selection is a ==diversity ranker==, which can be used to maximize the diversity of textual chunks included in the model's prompt by performing the following steps:
1. Use the retrieval pipeline to ==generate a large set of documents== that could be included in the model's prompt
2. ==Select the document that is *most similar==* to the input (or query) as determined by embedding cosine similarity.
3. For each remaining document, ==select the document that is *least similar* to the documents that are already selected==.


#### Optimizing Context Layout, with the "lost in the middle" problem
- Despite increases in context length, recent research has indicated that LLMs struggle to capture information in the middle of a large context window.
	- ==Information at the beginning and end of the context window is captured most accurately, causing certain data to be "lost in the middle".==
		- As a result, we have to be mindful about where in the context we place our information.
![[Pasted image 20240225163331.png]]
==Above==: With the "lost in the middle" theory in mind...that information at the *beginning and end* of the context window is captured most accurately... ==we adopt a selection strategy that is mindful of where context is placed in the prompt -- we iteratively place the most relevant textual chunks at the beginning and end of the context window==!
- ((You can see that #1 goes beginning, then #2 end, then #3 is just after beginning, then #4 is just before end, etc. Building inwards from the outside, basically!))

# Data Cleaning and Formatting
- In most RAG apps, our model will be retrieving textual information from many sources. An assist that is built to discuss the details from a codebase may pull information from:
	1. Code itself
	2. Documentation/manual pages
	3. Blog posts
	4. User discussion threads
	5. More
- So the data that's being *used* for RAG has a variety of different formats that might lead to artifacts (eg logos, icons, symbols) in the text that might confuse the LLM. ==As a result, we need to extract, clean, and format the text from each of these heterogenous sources.==

There's a lot more to preprocessing data for RAG than just splitting textual data into chunks!

#### Performance Impact
- If text isn't extracted properly from each knowledge source, the performance will deteriorate!
- In some blog post, preprocessing/cleaning data offered the following benefits:
	1. 20% boost in correctness
	2. 64% reduction in the number of tokens passed to the model
	3. Noticeable improvement in overall LLM behavior

#### Data Cleaning Pipeline
- The details of any data cleaning pipeline for RAG will depend heavily on the specific application
- We should:
	1. Observe manually large amounts of data in our knowledge base
	2. Visually inspect whether unwanted artifacts are present
	3. Amend issues that we find by adding changes to the data cleaning pipeline
- This isn't a flashy or cool approach, but any AI/ML practitioner knows that 90% of the time building an application is spent observing and working with data, not build sexy models.

- If you aren't interested in manually inspecting data, you can automate the process of creating a functional data processing pipeline by using LLM-as-a-Judge to iteratively construct the code for cleaning up and properly formatting data.
	- ((Unclear if he's saying just using an LLM to perform the data cleaning, or using the LLM to JUDGE the data cleaning process.))


## Further Practical Resources for RAG
((Added to my ML Articles bookmark folder))


# Closing Thoughts
- At this point, we have a comprehensive grasp of RAG, its inner workings, and how we can best approach building a high-performing LLM application using RAG.

Successfully applying RAG in practice requires more than a minimal functioning pipeline with pretrained components. We must refine our RAG approach by:
1. ==Creating a high-performing hybrid retrieval algorithm== (potentially with a reranking like this)
2. Constructing a ==data preprocessing pipeline== that properly formats data and removes harmful artifacts before the data is used for RAG
3. ==Finding correct prompting strategy== that allows the LLM to reliably incorporate useful context when generating output
4. ==Putting detailed evaluations in place== for both the retrieval pipeline (eg using ==traditional search metrics==) and the generation component (eg using an ==LLM-as-a-Judge==)
5. Collecting data over time that can be used to improve the RAG pipeline's ability to discover relevant context and generate



















