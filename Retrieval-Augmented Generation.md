---
aliases:
  - RAG
---
Un-grounded LLMs:
- Hallucinate often
- Don't cite their sources
- Are difficult/expensive to update/edit with new knowledge

==Fusion Retrieval== refers to the technique of fusing or combining information retrieved from multiple sources to enhance the relevance and quality of the retrieved data. Valuable in scenarios where a single retrieval method might not have comprehensive/accurate results.

==Query Transformation== refers to techniques used to enhance the effectiveness of IR by modifying or expanding the user's original query. This is done by applying:
- ==Semantic Analysis== to expand a user's query with synonyms, related terms, or semantically similar phrases.
- ==Entity Recognition==: Identifies entities mentioned in the user's query, like names of people, places, or specific objects.
- ==Query Reformulation==: Rephrase or reformulate the user's query to express the same *intent* in different words or structures.
Example ((I think there's some overlap, here, in reality))
- Original Query: "Tell me about the Eiffel Tower"
	- Semantic Expansion: "Provide information about the Eiffel Tower, also known as La Tour Eiffel, in Paris."
	- Entity Recognition: "Tell me about the famous Eiffel Tower."
	- Query Reformulation: "I'd like to learn more about the iconic Eiffel Tower in France."