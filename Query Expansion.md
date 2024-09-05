
Expanding a single query into multiple queries enriches the content of the query, providing further context to address any lack of specific nuances.
- *Multi-Query*: Expand to multiple queries that can be executed in parallel. The expansion of queries isn't random, but rather meticulously designed.
- *Sub-Query*: The process of sub-question planning represents the generation of the necessary sub-questions to contextualize and fully answer the original question, when combined.
- *==Chain of Verification (CoVe)==*: The expanded queries undergo validation by LLM to achieve the affect of reducing hallucination. 

Can be also done by applying any of:
- ==Semantic Analysis== to expand a user's query with synonyms, related terms, or semantically similar phrases.
- ==Entity Recognition==: Identifies entities mentioned in the user's query, like names of people, places, or specific objects.
- ==Query Reformulation==: Rephrase or reformulate the user's query to express the same *intent* in different words or structures.
Example ((I think there's some overlap, here, in reality))
- Original Query: "Tell me about the Eiffel Tower"
	- Semantic Expansion: "Provide information about the Eiffel Tower, also known as La Tour Eiffel, in Paris."
	- Entity Recognition: "Tell me about the famous Eiffel Tower."
	- Query Reformulation: "I'd like to learn more about the iconic Eiffel Tower in France."