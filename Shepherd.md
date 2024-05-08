August 8, 2023 (9 months after ChatGPT, 5 months after GPT-4)
[[Meta AI Research]] 
Paper: [Shepherd: A Critic for Language Model Generation](https://arxiv.org/abs/2308.04592)
#zotero 
Takeaway: Shepherd is a 7B parameter "critique model" (a la [[Constitutional AI|CAI]]), specifically tuned to critique responses and *suggest refinements*. It's similar in performance to ChatGPT (which is pretty impressive for a 7B model).

----

Notes:
- Shepherd can pinpoint specific issues like ==factuality, logical errors, coherence, and alignment==, and suggest improvements via natural language feedback.
- To fine-tune and evaluate Shepherd, we created a high-quality feedback dataset composing of two distinct sets:
	1. ==Community feedback==, curated from online forums to capture more diverse interactions
	2. ==Human-annotated feedback==, collected on generations across different task types
	- Close inspection of influences of community feedback and human-annotated feedback confirms that the community data is more informative and diverse than human-annotated data, yet leans toward informality.
- Data Collection
	- Community Critique Data: Authors want to gather clean `(question, answer, critique)` triplets. ==We consider the title and the sub-title of a post as a question, its top-level comments as answers, and replies to these comments as critiques.== Community vote score is calculated as (upvotes-downvotes).
		1. ==Stack Exchange Data==
		2. ==Pushshift Reddit Dataset== (from the more formal, information-seeking subreddits)
		- ((It seems to me that many people upvote comments for the "wrong" reason))
		- ==Critique post-processing==
			- We noticed that much of the data didn't include *suitable* critiques, where ==a suitable critique is **one of two cases**==:
				1. Answer is ***largely accurate***, and the critique offers recommendations for further refinement or enhancement.
				2. Answer ***contains inaccuracies***, which the critique explicitly highlights.
			- We want to filter out invalid critiques, like joke sharing and follow-up questions that fail to provide feedback, so we use two methods:
				1. **Keyword filtering**: We *keep* examples that contain certain keywords matching the above two cases.
					- (eg "indeed," "absolutely," "agree", "wrong," "incorrect," "beg to differ")
				2. **User edit history**: In addition to keyword filtering, we collect critiques if users edit their answer posting the critique. ((?))
			- We incorporate additional filters linked with community score votes.
				- In ==Case #1==, we omit data where the answer score is lower than 10 and the critique score is lower than 2, ensuring that we only select instances where the initial answer is largely approved by the community, and the critique has received *some* level of endorsement.
				- In ==Case #2==, we focus on data where the critique score surpasses the answer score, and the critique score itself is higher than 2. This ensures we're considering instances where the critique, indicating an error in the initial answer, has garnered more community approval than the answer itself.
			- Lastly, we impose several additional filters to further refine our dataset:
				1. To maintain diversity, we retain only one instance per post, choosing the one with the highest critique score.
				2. To manage offensive language in community data


Abstract
> As large language models improve, there is increasing interest in ==techniques that leverage these models' capabilities to refine their own outputs==. In this work, we introduce Shepherd, a ==language model specifically tuned to critique responses and suggest refinements, extending beyond the capabilities of an untuned model to identify diverse errors and provide suggestions to remedy them==. *At the core of our approach is a high quality feedback ==dataset==*, which we curate from community feedback and human annotations. Even though Shepherd is small (7B parameters), its critiques are either equivalent or preferred to those from established models including ChatGPT. Using GPT-4 for evaluation, Shepherd reaches an average win-rate of 53-87% compared to competitive alternatives. In human evaluation, Shepherd strictly outperforms other models and on average closely ties with ChatGPT.


# Paper Figures

![[Pasted image 20240507224812.png|300]]

![[Pasted image 20240507232854.png]]
Above: I mean... in my opinion, the StackExchange response is a totally inappropriate critique/feedback, right? Pretty informal (as the authors noted) and caustic.


# Non-Paper Figures
- 
