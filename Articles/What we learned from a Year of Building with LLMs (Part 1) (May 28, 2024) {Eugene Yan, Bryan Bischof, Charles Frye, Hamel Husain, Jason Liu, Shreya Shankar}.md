Link: https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-i/

This article was eye-catching to me because it got a positive reception and many of the speakers were hosts or guest lecturers on the LLM bootcamp course that I purchased.

---

The six authors come from a variety of backgrounds, but have all experienced firsthand the challenges that come with using this new technology.
- Two are independent consultants
- One is a researcher studying how ML/AI teams work
- Two are leaders on applied AI teams
- One has taught deep learning to thousands

But they were all surprised by the consistent themes in the lessons they've learned. The goal is for this to be a practical guide regarding that -- lessons learned the hard way.

Three sections:
- Tactical
- Operational
- Strategic

# (1/3) Tactical
- Some best practices for the core components of the emerging LLM Stack:
	1. Prompting tips to improve quality and reliability
	2. Evaluation strategies to assess output
	3. Retrieval-Augmented Generation ideas to improve grounding
	4. (More)

## Prompting
- Authors recommend *==starting with prompting when developing new applications!==* The right techniques can get you very far.

#### N-Shot Prompting
- The idea is to provide a few examples that demonstrate the task, and align outputs to our expectations.
- As a rule of thumb, ==aim for n >= 5 demonstrations, and don't be afraid to go as high as a few dozen==.
- ==Examples distribution (eg across classes) should be representative of the expected input distribution at inference time!==
- You ==don't always need to provide full I/O pairs; In many cases, examples of desired outputs are sufficient==!
- If you're using an LLM that supports tool use, your n-shot examples should also use the tools you want the agent to use!

#### Chain of Thought Prompting
- In [[Chain of Thought|CoT]], we encourage the LLM to explain its thought process *before* returning the final answer! Think of it as providing the LLM with a sketchpad so it doesn't have to do it at all in memory.
- We might do something like (example: Meeting transcript summarization):
> First, list the key decisions, follow-up items, and associated owners in a sketchpad.
> Then, check that the details in the sketchpad are factually consistent with the transcript.
> Finally, synthesize the key points into a concise summary.

#### Retrieval Augmented Generation (RAG)
- [[Retrieval-Augmented Generation|RAG]] is a powerful technique to expand the model's knowledge base, reduce hallucination, and increase user trust through citations.
- When providing relevant resources, ==it's not enough to merely include them; don't forget to tell the model to prioritize their use, refer/quote/cite to them directly, and even mention when none of the resources are sufficient.==

#### Structure your inputs and outputs!
- Adding structured I/O helps models better understand the input, as well as return output that can reliably integrate with downstream systems.
	- For example, many questions about writing SQL begin by specifying the SQL schema you're interacting with.
- ==Structured outputs simplify integration with downstream systems==; tools like [[Instructor]] or Outlines work well for structured output.
- ==When using structured input, be aware that each LLM family has their own preferences==; Claude prefers `xml`, while GPT favors Markdown and JSON.


#### Have small prompts that do one thing, and only one thing, well
- Prompts typically start simple, but as we try to improve performance and handle more edge cases, complexity creeps in, and we're suddenly got a 2,000 token Frankenstein that's weirdly gotten *worse* on the more-common and straightforward inputs!
- Instead of having a single, catch-all prompt for our meeting transcript summarizer, we can break it into steps:
	1. Extract key decisions, action items, and owners into a structured format
	2. Check extracted details against the original transcription for consistency
	3. Generate a concise summary from the structured details
- ==Splitting our complex single prompt into *multiple* prompts that are each simple, focused and easy to understand lets us iterate and evaluate each prompt individually.==
- Think about how you structure your context -- a bag-of-docs representation isn't helpful for humans nor agents. Structure it to underscore relationships between parts of it, and make extractions as simple as possible -- root out redundancy, self-contradictory language, and poor formatting.

## Information Retrieval/RAG
- Providing knowledge as part of the prompt grounds the LLM on the provided context that is then used for in-context learning.
- ==The quality of your RAG output is dependent on the quality of your retrieved documents, which in turn can be considered along a few factors.==
- The first among these is ==Relevance==, which is typically calculated via metrics like [[Mean Reciprocal Rank]] (MRR) or [[Normalized Discounted Cumulative Gain]] (NDCG). These measure how good the system is at ranking relevant documents higher and irrelevant documents lower.
	- MRR evaluate how well a system places the *first relevant result* in a ranked list.
	- NDCG considers the relevance of *all* the results and their positions.
- Next, we want to consider ==Information Density==. If two documents are equally relevant, we should prefer one that's more concise and has fewer extraneous details.
- Finally, we consider the ==level of detail== provided in the document.
	- If we're building a RAG system to generate SQL queries from natural language, we could simply provide table schemas with column names as context, but what if we include column descriptions with some representative values?
- ==Don't forget keyword search such as [[BM25]]==! Use it as a baseline and in hybrid search!
	- While embeddings are powerful and are good at capturing high-level semantic similarity, they might struggle when users have more specific, keyword-based queries, acronyms, or IDs -- which are what keyword-based search algorithms are designed for!
	- They're also more interpretable; we can look at the keyword that match the query.
	- They're often more battle-tested! Systems like Lucene and OpenSearch have been optimized and battle-tested for decades.

---
> *"Vector embeddings _do not_ magically solve search. In fact, the heavy lifting is in the step before you re-rank with semantic similarity search. Making a genuine improvement over BM25 or full-text search is hard."*
> - Aravind Srinivas, CEO Perplexity.ai

> *"We’ve been communicating this to our customers and partners for months now. Nearest Neighbor Search with naive embeddings yields very noisy results and you’re likely better off starting with a keyword-based approach."*
> -Beyand Liu, CTO Sourcegraph

---

- ==In most situations a [[Hybrid Search]] option will work best==!
	- Keyword-matching for the obvious matches, and embeddings for synonyms, hypernyms, and spelling errors, as well as multimodality (eg images and text).
- Note that ==query rewriting== is also a useful technique.

- ==Prefer RAG over fine-tuning for new knowledge.==
	- It's better, easier, and cheaper to keep retrieval indices up-to-date, and we can easily purge toxic/biased content from them, as well as do things like access control if we're serving multiple users.

- Long-context models won't make RAG obsolete.
	- Even with a context window of Gemeni 1.5's 10M tokens, we still need a way to select information to feed into the model.
	- ==Beyond the narrow needle-in-a-haystack (NIAH) evaluation, we don't yet have convincing results that models can effectively *reason* over such large contexts!==
	- There's also inference cost, which scales quadratically (or linearly in both space and time) with context length -- just because you can read your entire organization's Google Drive contents before answering every question doesn't mean you should!

## Tuning and Optimizing Workflows

- Step-by-step multi-turn "flows" can give large boosts.
	- In [[AlphaCodium]], they switched from a single prompt to a multi-step workflow and increased GPT-4 accuracy (pass@5) on CodeContests from 19% to 44%! This workflow included:
		1. Reflecting on the problem
		2. Reflecting on the public tests
		3. Generating possible solutions
		4. Ranking possible solutions
		5. Iterating on the solutions on public and synthetic datasets
- ==Some things to try==:
	1. An explicit planning step, as tightly-specific as possible
	2. Rewriting the original user prompts into agent prompts (this might be lossy -- careful!)
	3. Model agent behaviors as linear chains, DAGs, and State-Machines; different dependency and logic relationships can be more and less appropriate for different scales.
	4. Planning validations; Your planning can include instructions on how to evaluate the responses from *other* agents to make sure the final assembly works well together.
	5. Prompt engineering with fixed upstream state -- make sure your agent prompts are evaluated against a collection of variants of what may happen before.

- The likelihood that an agent completes a multi-step task successfully decreases as exponentially as the number of steps increases, making it hard to deploy reliable agents.
	- In the end, the key to reliable, working agents will likely be found in adopting more structured, deterministic approaches, as well as collecting data to refine prompts and finetune models.

How to get more diverse outputs beyond temperature
- Increasing the temperature parameter makes LLM responses more varied (it makes the probability distribution flatter, meaning that tokens which are usually less-likely get chosen more often).
- ==Other tricks== include ==adjusting element orderings within the prompt== (eg shuffling the order of lists of items each time they're inserted into a prompt), ==keeping a short list of recent outputs== (and instructing the LM to avoid suggesting items from this list), or by ==varying the phrasing used in prompts== ("pick an item the user would love using regularly" vs "select a product the user would recommend to friends.")

Caching is underrated
- Caching saves cost/latency by removing the need to recompute responses for the *same input.*
- One straightforward approach to caching is to use unique IDs for the items being processed, like if we're summarizing new articles or product reviews. When a request comes in and a summary already exists in the cache, we can return it immediately.
- ==Features like autocomplete and spelling correction can help *normalize* user input and thus increase the cache hit rate.==

When to fine-tune
- We might have some tasks where even the most cleverly-designed prompts fall short. If so, then we might want to finetune a model for your specific task. This can be effective, but comes with significant costs:
	- We need to annotate fine-tuning data (To reduce the price here, we could generate synthetic data or leverage open-source data!)
	- Finetune and evaluate models
	- Eventually self-host them
- So consider if the upfront cost is worth it! If prompting is already getting you 90% of the way there, it might not be worth the investment.

Create a few assertion-based tests from real input/output samples!
- ==Create unit tests consisting of samples of inputs and outputs from production==, with expectations for outputs based on at least three criteria (a number to start with).
	- Could be assertions that specify phrases or ideas to either include or exclude in all responses, etc.
	- In certain situations, if a user asks for generated code for a new function named `foo`, then `foo` should be callable!
- These unit tests ==should be triggered by any changes to the pipeline, whether it's editing a prompt, adding new context via RAG, or other modifications!==
- Actually using your product as intended for customers (dogfooding) can provide insights into failure modes on real-world data.

LLM-as-a-Judge
- When implemented well, LLM-as-a-Judge achieves decent correlation with human judgements, and can at least help build priors about how a new prompt or technique might perform.
- Suggestions:
	- Use pairwise comparisons, rather than asking an LLM to score a single output. This tends to be more stable.
	- Control for [[Positional Bias]]; do each pairwise comparison twice, swapping the order of pairs each time!
	- Allow for ties; In some cases, both options may be equally good!
	- Use [[Chain of Thought]]; Asking the LLM to explain its decision *before* giving a final preference can increase eval reliability.
	- Control for response length; LLMs tend to bias toward longer responses; ensure response pairs are similar in length, if possible.

[Link on a simple but effective approach](https://hamel.dev/blog/posts/evals/#automated-evaluation-w-llms)

==LLM-as-a-Judge is NOT a silver bullet!==
- There are subtle aspects of language where even the strongest models fail to evaluate reliably.
- In some cases, ==conventional classifiers and reward models can achieve higher accuracy, with lower cost and latency!== In Code-gen situations LM Judges can be weaker than more direct evaluation strategies like execution-evaluation.

Simplify annotation to binary tasks or pairwise comparisons
- Providing open-ended feedback or ratings for model outputs on a Likert Scale (eg 1-5) is cognitively demanding, and data collected is more noisy due to variability among human raters.
- ==Compared to the Likert scale, binary decisions are more precise, have higher consistency among raters, and lead to higher throughput.==
- In pairwise comparisons, the annotator is presented with a pair of model responses and asked to say which is better. LLaMA 2 authors confirmed that this is faster and cheaper than collecting (eg) written responses.

[[Guardrails]] help to catch inappropriate or harmful content while evals help to measure the quality and accuracy of the model's output.
- Reference-free evaluations are evaluations that don't rely on a "golden" reference, like a human-written answer, and can assess the quality of output based solely on the input prompt and the model's response. Reference-free evaluations and guardrails can be used similarly.

Careful prompt engineering can help LLMs return "non applicable" or "unknown" responses, but we should complement them with robust guardrails that detect and filter/regenerate undesired output.
- Example: OpenAI provides a content moderation API that can identify unsafe responses such as hate speech, self-harm, or sexual output.
