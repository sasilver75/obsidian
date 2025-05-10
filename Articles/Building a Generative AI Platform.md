[[Chip Huyen]]
July 25, 2024
#review

-------

There are many similarities in generative AI applications -- this post outlines common components of generative AI platforms, what they do, and how they're implemented.

![[Pasted image 20250109102011.png|600]]
This is the complex system that we'll end up with -- we'll start with a simple architecture and progressively add more components.

Let's start with a simple form, where our application receives a query and sends it to the model:
![[Pasted image 20250109102058.png|600]]
There are no guardrails, no augmented context, and no optimization. The Model API box refers to either some third-party API (e.g. OpenAI, Anthropic) or some self-hosted API.

From this, we can add more components as needs arise:
1. ==Enhance context== input into a model by giving the model access to external data sources and tools for information gathering.
2. Put in ==guardrails== to protect your system and your users.
3. Add ==model router and gateways== to support complex pipelines and add more security.
4. Optimize for latency and costs with ==caches==.
5. Add ==complex logic== and write ==actions== to maximize your system capabilities.

- ==Observability== allows you to gain visibility into your system and monitoring and debugging, and orchestration.
- ==Orchestration== involves chaining all the components together, are two essential components of the platform. We will discuss them at the end of the post.

----------

# Step 1: Enhance Context

The initial expansion of a platform usually involves adding mechanisms to allow the system to augment each query with the necessary information.

Gathering the relevant information is called ==context construction==.

Many queries require context to answer. The more relevant information there is in the context, the less the model has to rely on its internal knowledge, which can be unreliable due to its training data and training methodology.

Example:
> "Will Acme's fancy-printer-A300 print 100pps?"

- The model will be able to respond better if it's given the specifications of fancy-printer-A300.
- Context construction for foundation models is equivalent to feature engineering for classical ML models.

In-context learning is a form of continual learning. It enables a model to incorporate new information continually to make decisions, preventing it from becoming outdated.

### RAGs
- The most well-known patterns for context construction is RAG, [[Retrieval-Augmented Generation]].
	- RAG consists of two components: a ==generator== (e.g. a language language) and a ==retriever==, which retrieves relevant information from external sources..

![[Pasted image 20250109103904.png|600]]
Retrieval isn't unique to RAG; it's the backbone of search engines, recommender systems, log analytics, etc. 
- Many retrieval algorithms developed for traditional retrieval systems can be used for RAG.

External memory sources typically contain unstructured data called ==documents==. 
- A document can be 10 tokens or 1 million tokens -- naively retrieving whole documents can cause your context to be arbitrarily long, so RAG typically requires documents to be split into manageable ==chunks==, which can be determined from the model's maximum context length and your application's latency requirements.

1. Term-based retrieval (==lexical retrieval==):Can be as simple as keyword search, but more sophisticated algorithms include [[BM25]] (which leverages [[TF-IDF]]) and ElasticSearch (which leverages inverted indexes). Usually used for text data, but also works for searching for images and videos if they have text metadata like tags, captions, comments, etc.
2. Embedding-based (==vector search==): Convert chunks of data into embedding vectors using an embedding model like [[BERT]], [[Sentence Transformers]], and proprietary embedding models provided by OpenAI or Google. Given a query, the data whose vectors are closest to the query embedding, as determined by the vector search algorithm, is retrieved.

Vector search is usually framed as ==nearest-neighbor search==, using [[Approximate Nearest Neighbor Search|Approximate Nearest Neighbor]] (ANN) search algorithms like [[FAISS]], Google's ScaNN, Spotify's ANNOY, and [[Hierarchical Navigable Small Worlds|HNSW]].

For these ANN algorithms, there are four main metrics to consider, taking into account the ==tradeoffs between indexing and querying==:
- ==Recall==: The fraction of the nearest neighbors found by the algorithm.
- ==Query per Second== (QPS): Number of queries that the algorithm can handle per second. This is crucial for high-traffic applications.
- ==Build Time==: The time required to build the index. This metric is important especially if you need to frequently update your index (e.g. because your data changes)
- ==Index Size==: The size of the index created by the algorithm, which is crucial for assessing its scalability and storage requirements.

This works not just with text documents, but with images, video, audio, and code -- anything that can be embedded! (Some teams even try to summarize SQL tables and dataframes).

==Term-based retrieval (lexical retrieval)== is much *faster* and *cheaper* than embedding-based retrieval. It can work well out of the box, making it an attractive option to start. Both BM25 and ElasticSearch are widely used and serve as formidable baselines.

==Embedding-based (dense vector search)== is more computationally expensive, but can outperform term-based retrieval with better semantic understanding.

A production system typically combines several approaches, a technique called [[Hybrid Search]] or [[Hybrid Search|Hybrid Retrieval]].
- A common pattern is ==sequential==:
	- First a cheap, less-precise lexical retrieval fetches candidate
	- Then a more precise but expensive mechanism (e.g. [[K-Nearest Neighbors|kNN]]) finds the bets of these candidates. This step is also called [[Reranking]].

For example, given the term "transformer," you can fetch all documents that contain the word "transformer," regardless of whether it's about the electric device, the neural architecture, or the movie. Then we use vector search to find among these documents those that are actually related to our query.

==Context reranking== (meaning ordering documents to be fed into a language model context) differs from traditional search reranking in that the *exact* position of items is less critical -- though models may better understand documents at the beginning and end of the context, as suggested by the Lost in the Middle (Liu et al. 2023) paper.

- Another common pattern is ==ensemble==: 
	- Remember that a retriever works by ranking documents by their relevance scores to the query. You can use *multiple retrievers* to fetch candidates at the same time, then combine these different rankings together to get a final ranking.

### RAG with tabular data
If you have structured external data sources, like dataframes or SQL tables, retrieval might look different: 
![[Pasted image 20250109111354.png|500]]
1. Text-to-SQL: Based on the user query and the table schema, determine what SQL query is needed
2. SQL execution: Execute the SQL query
3. Generation: Generate a response based on the SQL result and the original user query.

### Agentic RAG

Another important source of data is the Internet. A web search tool like Google or Bing API can give the model access to rich, up-to-date resources.

Term-based retrieval, embedding-based retrieval, SQL execution, and web search are all actions that a model can take to augment its context.
- You can think of each action as a ==function== that the model can call -- a workflow that can incorporate external actions is also called *agentic.*
![[Pasted image 20250109111628.png|600]]

Action vs. Tool:
- ==A tool allows one or more actions==
	- A search tool might allow for:
		- search by email
		- search by username
But this difference is minimal, so many people use action and tool interchangeably.

Read-only actions vs. Write actions
- Actions that retrieve information from external sources but don't change their states are read-only actions.
- Giving a model write actions enables the model to perform more tasks, but also poses more risk.

### Query rewriting

Often, a user query needs to be rewritten to increase the likelihood of fetching the right information. Consider the following conversation:

```
User: What was the last time John Doe bought something from us?
AI: John bought a Fruity Fedora hat from us two weeks ago, on January 3.
User: How about Emily Doe?
```

The "How about Emily Doe" is ==ambiguous==; if you use this query verbatim to retrieve documents, you'd likely get irrelevant results.
You need to rewrite this query to reflect what the user is actually asking -- ==the new query should make sense on its own== , and should be rewritten to:

```
User: When was the last time Emily Doe bought something from us?
```

This query-rewriting is typically done using other AI models, using a prompt similar to "Given the following conversation, rewrite the last user input to reflect what the user is actually asking."

Query rewriting can get complicated, especially if you need to do identity resolution or incorporate other knowledge. If the user asks "How about his wife?" you will first need to query your database to find out who his wife is. ==If you don't have this information, the rewriting model should acknowledge that this query isn't solvable.==

# Step 2. Put in Guardrails

Guardrails help reduce AI risks and protect not just your users but also you, the developers!

### Input Guardrails
- ==Input guardrails== typically protect against two types of risks: ==leaking private information== to external APIs and ==executing malicious prompts== that compromise your system.

Leaking private information to external APIs
- This risk is specific to using external model APIs when you need to send data outside your organization.
	- If a user accidentally sends their private information or password into the prompt, you want to not propagate that to other parties.
- Common sensitive data classes include:
	- Personal information (ID numbers, phone numbers, bank accounts)
	- Human faces
	- Specific keywords and phrases associated with the company's intellectual properties or privileged information.

Many sensitive data detection tools use AI to identify potentially sensitive information, such as determining if a string resembles a home address.

If a query is found to contain sensitive information, you have two options:
- Block the entire query
- Remove the sensitive information from it, e.g. masking a user's phone number with \[PHONE NUMBER\].
	- You can also re-map these in model responses to the actual content, e.g. mapping from \[ACCESS_TOKEN\] to "secret_token_that_shouldnt_be_leaked" using a PII reversible dictionary.

![[Pasted image 20250109120346.png|500]]

#### Model Jailbreaking

It's become an online sport to try to Jailbreak AI models to get them to say or do bad things. This is especially dangerous for AI systems that have access to tools -- imagine a user finding a way to get y our system to execute SQL queries that corrupt your data.

To combat this, you need ==guardrails== on your system so that no harmful actions can automatically be executed:
- e.g. No SQL queries that can insert, delete, or update data can be executed without human approval

To prevent your application from making outrageous statements that it shouldn't be making, you can define out-of-scope topics for your application - if it's a customer support chatbot, it shouldn't answer political or social questions.
- A simple way is to just filter inputs that ==contain predefined phrases== typically associated with controversial topics, such as "immigration" or "antivax." 
- More sophisticated algorithms use ==ML classifiers== to determine if an input is about one of the pre-defined restricted topics.

### Output guardrails

- Guardrails can improve your application's reliability, and have two main functionalities:
	1. ==Evaluate the quality of each generation==
	2. ==Specify the policy to deal with different failure modes.==

To catch outputs that fail to meet your standards, you need to understand what failures look like for your application - look at your data!

Examples
1. Empty responses 
2. Malformatted responses that don't follow expected output format (e.g. JSON. Regex): There are available as well as tools for [[Constrained Decoding|Constrained Sampling]] such as `guidance`, `outlines`, and `instructor`.
3. Toxic responses, such as those that are racist or sexist: Can be caught using one of many toxicity detection tools.
4. Factually inconsistent responses hallucinated by the model: Can mitigate with sufficient: Hallucination detection is an active area of research (see SelfCheckGPT, SAFE) -- Mitigate by providing models with sufficient context and using appropriate prompting techniques.
5. Responses that contain sensitive information (If your model leaks sensitive information it was trained on, or leaks some of the sensitive information it retrieved to enrich its response)
6. Brand-risk responses (Those that mischaracterize your company or competitors): Use keyword monitoring (your brand, competitors' brand) -- Block those outputs or pass them on to human reviewers, or use other models to detect the sentiment of these outputs to ensure that only the right sentiments are returned.
7. Generally bad responses (e.g. low-quality essay, unhealthy or gross recipes): Use AI judges to evaluate the quality of models' responses. These AI judges can be general-purpose models (e.g. ChatGPT) or specialized scorers trained to output a concrete score for a response given a query.

Failure management:
- Many failures (e.g. empty response) can be mitigated using a basic retry logic. For example, if the response is empty or if JSON is malformed, try again X times until you get a non-empty response
- The retry policy can incur extra latency and cost though.
	- For this reason, you might ==consider making calls in parallel==, and pick the better (or first) response.
- It's common to fallback to humans to handle tricky queries. You can transfer a query to human operators if it contains specific key phrases. You might use a ==specialized model trained in-house to decide when to transfer a conversation== to human, or you might transfer a ==conversation after a certain number of turns== to prevent users from getting stuck in an infinite loop.

Guardrail tradeoffs
- ==Reliability vs. Latency tradeoff==
	- While acknowledging the importance of guardrails, some teams told me that latency is more important -- They decided not to implement guardrails because they can significantly increase their applications' latency. But this is a minority of teams.
- ==Output guardrails might not work well in stream completion mode==. On the flip side, if the whole response isn't shown to users until it's complete, that can take a long time.
- Self-hosted vs. Third-Party tradeoff:
	- If you're self-hosting models, you don't have to send data to a third party, which reduces the need for certain input guardrails -- but it means that you also need to implement all the guardrails yourself, rather than relying on those provided by inference providers.

![[Pasted image 20250109123239.png|600]]
Above:
- User query undergoes ==context construction==, in which the query is rewritten and enhanced with appropriate context.
- Input guardrails are applied (e.g. PII redaction) 
- Full conversation is sent to Model API (either internally or externally)
- Response is scored by a LM Judge
- Output guardrails are applied, looking for keywords, adherence to structured outputs, etc.
- I'm guessing reverse PII is applied for any templated/masked content.


## Step 3: Add Model Router and Gateway

- As applications grow in complexity and involve more models, tow types of tools emerged to help you work with multiple models: routers and gateways:

==Router==:
- An application can use different models to response to different types of queries.
- First, this allows you to have specialized solutions:
	- One model trained in technical troubleshooting
	- Another trained specialized in subscriptions.

Specialized models can potentially perform better than a general-purpose model. This can also help you save costs! Instead of routing all queries to an expensive model, you can route simpler queries to cheaper models.

A router typically consists of:
- An ==intent classifier== that predicts what the user is trying to do
- Based on the predicted intent, the query is routed to the appropriate solution.

For customer service, if the intent is:
- To reset a password: Route this user to a page about password resetting.
- Correct a billing mistake: Route this user to a human operator.
- Troubleshoot a technical issue: Route this user to a model finetuned for troubleshooting.

An intent classifier can also help your system avoid out-of-scope conversations.
- If the query is deemed inappropriate (e.g. the user asks who you would vote for in the upcoming election), the chatbot can politely decline to engage using one of the stock responses ("As a chatbot, I don't have the ability to vote. If you have questions about our products, I'd be happy to help.") without wasting an API call.

If your system has access to multiple actions, a router can involve a ==next-action predictor== to help the system decide what action to take next.
One valid action is to ask for clarification if the query is ambiguous.
For example, in response to the query "Freezing," the system might say: "Do you want to freeze your account or are you talking about the weather?"

When routing queries to models with varying context limits, the query's context might need to be adjusted accordingly.
- Consider a query of 1,000 tokens slated for a model with a 4k context limit; the system then takes an action, e.g. web search, that brings back 8,000-token context.
	- That system might do a web search that brings back an 8,000-token context; you can either truncate the query's context to fit the originally-intended model or route the query to a model with a larger context limit..


## Gateway
- A ==model gateway== is an intermediate layer that allows your organization to interface with different models in a unified and secure manner.
- The most basic functionality of a model gateway is to enable developers to access different models
	- Whether they're self-hosted models or models behind commercial APIs such as OpenAI or Google -- they can be accessed the same way.

A model gateway makes it easier to maintain your code. If a model API changes, you only need to update the model gateway instead of having to update all applications that use this model API.

![[Pasted image 20250109130358.png|500]]

In its simplest form, a model gateway is a unified wrapper that looks like the following code example. This example is to give you an idea of how a model gateway might be implemented.

This is a very simple example (without error checking, etc.):

```
import google.generativeai as genai
import openai

def openai_model(input_data, model_name, max_tokens):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    response = openai.Completion.create(
        engine=model_name,
        prompt=input_data,
        max_tokens=max_tokens
    )
    return {"response": response.choices[0].text.strip()}

def gemini_model(input_data, model_name, max_tokens):
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name=model_name)
    response = model.generate_content(input_data, max_tokens=max_tokens)
    return {"response": response["choices"][0]["message"]["content"]}

@app.route('/model', methods=['POST'])
def model_gateway():
    data = request.get_json()
    model_type = data.get("model_type")
    model_name = data.get("model_name")
    input_data = data.get("input_data")
    max_tokens = data.get("max_tokens")

    if model_type == "openai":
        result = openai_model(input_data, model_name, max_tokens)
    elif model_type == "gemini":
        result = gemini_model(input_data, model_name, max_tokens)
    return jsonify(result)
```

A model gateway is ==access control and cost management==
- Instead of giving everyone who wants access to the OpenAI API your organizational tokens, which can be easily leaked, you only give people access to the model gateway, creating a centralized and controlled point of access.
- The gateway can also implement ==fine-grained access controls==, specifying which users or application should have access to which model.
- The gateway can ==monitor and limit the usage of API calls==, preventing abuse and managing costs effectively.
- Can be used to implement ==fallback policies== to ==overcome rate limits or API failures== (the latter is unfortunately common).

Given that gateways are relatively straightforward to implement, there are many off-the-shelf gateways:
1. Portkey's Gateway
2. MLflow AI Gateway
3. WealthSimple's LLM-Gateway
4. TrueFoundry
5. Kong
6. Cloudflare

With the added gateway and routers, our platform gets more exciting!
Like scoring, routing is also in the model gateway. Like models used for scoring, models used for routing are typically smaller than models used for generation.

![[Pasted image 20250109130816.png|600]]
Above:
- Model context construction (e.g. enhancement with external information, query rewriting)
- Input guardrails are applied, removing PIP information and malicious content
- Requests are sent to model gateway, rather than directly to an internal or external API. The model gateway takes on routing, generation, scoring, certain error handling, monitoring, etc.
- Output guardrails regarding safety/verification, structured outputs are applied.

## Step 4. Reduce Latency with Cache

- [[Eugene Yan]] said that ==Caches== are perhaps the most underrated components of AI platforms, which can significantly reduce your application's latency and cost.

Let's focus on caches for inference, where common techniques include:
1. ==Prompt cache== (typically implemented by the inference APIs that you use)
2. ==Exact cache==
3. ==Semantic cache==

([[KV Cache]]s for attention mechanisms are out-of-scope for this discussion)

### [[Prompt Caching|Prompt Cache]] (Context Cache)
- Many prompts in an application have overlapping text segments
	- For example, all queries share the same system prompt.
	- A prompt cache stores these overlapping segments for reuse, so you only need to process them once. Without prompt caching, your model needs to process the system cache with every query -- with it, it only needs to process the system prompt once for the first query.
- Prompt caching can ==significantly reduce latency and cost with long system prompts==.
- ==This isn't free though, like KV cache, prompt cache size can be quite large and require significant engineering effort.==
- Prompt caching is ==also useful for queries that involve long documents==, especially multi-turn -- If many of your user queries are related to the same long document (such as a book or codebase), this long document can be cached for reuse across queries.

### Exact Cache
- A more general and straightforward cache: If a user asks a model to summarize a product, the system checks the cache to see if a summary of this model is cached.
	- If yes, fetch the summary
	- If not, summarize the product and cache the summary

- An exact cache is also used for embedding-based retrieval to avoid redundant vector search. If an incoming query is already in the vector search cache, fetch the cached search result; otherwise, perform a vector search and cache the result.
- An exact cache can be implemented using ==in-memory storage== for fast retrieval, but since in-memory storage is limited, a cache can also be implemented using databases like PostgreSQL, Redis, or tiered storage to balance speed and storage capacity.

Having an ==eviction policy== is crucial to manage the cache size and maintain performance:
1. Common eviction policies include Least Recently Used (LRU), Least Frequently Used (LFU), and First In, First Out (FIFO).
2. How long to cache a query depends on how likely this query is to be called again. User-specific queries such as "What's the status of my recent order" are less likely to be reused by other users, and therefore shouldn't be cached. Similarly, it makes less sense to cache time-sensitive queries such as "How's the weather?" ==Some teams train a small classifier to predict whether a query should be cached.==

### Semantic Cache (c.f. exact cache)
- Unlike exact cache, semantic caches don't require the incoming query to be identical to any of the cached queries. 
- ==Semantic cache allows the reuse of similar queries==.
- Imagine one user asks "What's the capital of Vietnam," and the model generates the answer "Hanoi"
	- Then later, "What's the capital *city* of Vietnam" -- this is a different query, but asks the same semantic question. The idea of a semantic cache is that we can reuse the answer "Hanoi" rather than computing the new answer from scratch.

Semantic cache only works if you have a reliable way to determine if two queries are semantically similar.
One approach is *==embedding-based similarity==,* which works as follows:
1. For each query, generate its embedding using an embedding model.
2. Use vector search to find the cached embedding closest to the current query embedding. Let's say this similarity score is X.
3. If X is more than the ==similarity threshold== you set, the cached query is considered to be the same as the current query, and the cached results are returned. If not, process the current query and cache it together with its embedding and results.

==This approach requires a vector database to store embeddings of cached queries.==

A semantic cache's value is more dubious because of many components that are prone to failure...
- ==Setting the right similarity threshold can be tricky and require a lot of trial and error== ((I would think it's also harder for open-ended systems))

Semantic caches can also be ==time-consuming and compute-intensive==, as it involves a vector search. The speed and cost of this depends on the size/structure of your database and cached embeddings.

Semantic cache might still be worth if it the cache rate is high, but make sure to evaluate efficiency, cost, and performance risks associated with it.

![[Pasted image 20250109133644.png]]
Above:
- Now we've added cache components both to the user query and to the context retrieval query. These could be exact or semantic caches.
	- It's interesting to me that the caching of the user query happens before context construction (which includes query-rewriting). I don't know if this is required.
- Input guardrails are still applied, removing PII and similar features.
- Model gateway applies routing to appropriate models, generation, and scoring of generations.
- Output guardrails perform safety verification, structured output confirmation, etc.

## Step 5: Add Complex Logic and Write Actions

The applications we've discussed so far have fairly simple flows.

The outputs generated by the foundation models are mostly returned to users (unless they don't pass the guardrails).

However, an application flow can be more complex, with ==loops and conditional branching.== A model's outputs can be used to invoke ==write actions,== such as composing an email or placing an order.

### Complex logic
- Outputs from a model can be conditionally passed onto another model or fed back to the same model as part of the input to the next step.
	- This goes on until a model in the system decides that the task has been completed and that a final response should be returned to the user.
	- An example of a use case that might involve this logic is: "Plan a weekend itinerary for Paris!"
		- We might initially generate a list of potential activities, and then each of these activities can be fed back into the model to generate more detailed plans.

"Visiting the Eiffel Tower" can prompt the model to generate sub-tasks like checking the opening hours, buying tickets, and finding nearby restaurants. This iterative process continues until a comprehensive and detailed itinerary is created.x

![[Pasted image 20250109134543.png|500]]
- See now that the response from the model becomes input to the model (or another model).

#### Write Actions
- Actions used for context construction are *read-only actions*. They allow a model to read from its data sources to gather context.
- But a system can also "write actions," making changes from the data sources and the world. For example, if the model outputs "Send an email to X with the message Y," the system will invoke the action `send_email(recipient=X, message=Y)`

Write actions make a system vastly more capable.
They enable you to automate the whole customer outreach flow:
- Researching potential customers
- Finding their contacts
- Drafting emails
- Sending first emails
- Reading responses
- Following up
- Extracting orders
- Updating your databases with new orders, etc.

==Just as you shouldn't give an intern the authority to delete your production database, you shouldn't allow an unreliable AI to initiate bank transfers!==

==Prompt Injection== happens when an attacker manipulates input prompts into a model to get it to express undesirable behaviors. You can think of prompt injection as social engineering done on AI instead of humans.

A scenario that many companies fear is that they give an AI system access to their internal databases and attackers trick this system into revealing private information from these databases. 

![[Pasted image 20250109135432.png|600]]


## Observability
- While I've placed observability in its own section, it should be integrated into the platform from the beginning, rather than added later as an afterthought.

This section has the least information compared to the others; it's hard to cover all nuances of observability in a blog post?

This section provides the least information compared to the others. It's impossible to cover all the nuances of observability in a blog post.

The three pillars of monitoring:
- ==Logs==
- ==Traces==
- ==Metrics==

We won't go into specifics or cover user ==feedback==, ==drift detection==, and ==debugging==.

Metrics
- Most people think of metrics when discussing monitoring.
- In general, there are two types of metrics you want to track:
	- ==Model Metrics==: Assess your model performance, such as Accuracy, Toxicity, Hallucination rate. Different steps in an application pipeline also have their own metrics. In a RAG application, the retrieval quality is often evaluated using context relevance and context precision.
	- ==System Metrics==: Tell you the state of your overall system: Throughput, memory usage, hardware utilization, service availability/uptime.

There are various ways that a model's output can fail:
- ==It's crucial to identify these issues and develop metrics to monitor them.==
- For example, you might want to track how often your model times out, returns empty responses, or produces malformed responses. If you're worried about your model revealing sensitive information, find a way to track that too!

==Length-related metrics== such as query, context, and response length are helpful for understanding your model's behaviors.
- Is one model more verbose than another? Are certain type of queries more likely to result in lengthy answers?
- If average query length suddenly decreases, it could indicate an underlying issue that needs investigation.

Tracking ==latency== is essential for understanding the user experience.
Common latency metrics include:
- ==Time to First Token (TTFT)==: The time it takes for the first token to be generated
- ==Time Between Tokens (TBT)==: The interval between each token generation
- ==Tokens per Second (TPS)==: The rate at which tokens are generated
- ==Time per Output Token (TPOT)==: The time ti takes to generate each output token
- ==Total Latency==: The total time required to complete a response

You'll also want to track costs. 
- Cost-related metrics:
	- ==Number of queries==, ==Request per Second==
	- ==Volume of input and output tokens==

When calculating metrics, you can choose between ==spot checks== and ==exhaustive checks==.
Spot checks involve sampling a subset of data to quickly identify issues, while exhaustive checks evaluate every request for performance review.

## Logs
- The philosophy for logging is simple: ==log everything!==
	- Log the system configuration
	- Log the query
	- Log the output
	- Log the intermediate outputs
	- Log when a component starts, ends, and when something crashes, etc.
- When recording a piece of log, make sure to give tit ==tags and IDS that help you know where int eh system this log comes from!==
- Logging everything means that the amount of logs that you can have grow very quickly. Many tools for automated log detection are powered by AI.
- It's useful to manually inspect your production data ==DAILY== to get a sense of how users are using your application -- developers' perceptions of what constitutes good and bad outputs changes as they interact with more data, allowing them to both rewrite their prompts to increase the chance of good responses, or update their evaluation pipeline to catch bad responses.


## Trace
- Trace refers to the detailed recording of a request's execution path through various system components and services. In an AI application, tracing reveals the entire process from when a user sends a query to when a final response is returend.
- ![[Pasted image 20250109142018.png|400]]
Langsmith Trace

Ideally, you should be able to trace each query's transformation through the system step-by-step. ==If a query fails, you should be able to pinpoint the exact step where it went wrong!==


## AI Pipeline Orchestration

- At a high level, an ==orchestrator== (which helps you specify how these different components are combined) works in two steps:
	- ==Components definition==
		- You need to tell the orchestrator what components your system uses, such as models databases form which you can retrieve, actions your system can take.
	- ==Chaining (Pipelining)==
		- You tell the orchestrator the short sequence of steps your system takes from receiving the user query until completing the task. In short, chaining is just function composition -- here's an example of what a pipeline looks like:
			1. Process the raw query
			2. Retrieve the relevant data based on the processed query
			3. The original query and retrieved data are combined to create a prompt in the format expected by the model
			4. The model generates a response based on the prompt
			5. Evaluate the response
			6. If the response is considered good, return it to the user. If not, route the query to a human operator.

There are many AI orchestration tools, including LangChain, LlamaIndex, Flowise, LangFlow, and Haystack -- each have their own APIs.

==Start building your application without an orchestration tool first!== 
- Any external tooling brings added complexity, and orchestrators can abstract away critical details of how your system works, making it hard to understand and debug your system.

Three aspects to keep in mind when evaluating orchestrators:
1. Integration and extensibility
2. Support for complex pipelines
3. Ease of use, performance, and scalability

Does you orchestrator support the components you need? Does it support advanced features like branching, parallel processing, and error handling? Does it initiate hidden API calls that introduce latency to your application? Can the orchestrator scale with you? How is the community?












































