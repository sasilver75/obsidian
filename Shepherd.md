August 8, 2023 (9 months after ChatGPT, 5 months after GPT-4)
[[Meta AI Research]] 
Paper: [Shepherd: A Critic for Language Model Generation](https://arxiv.org/abs/2308.04592)
#zotero 
Takeaway: Shepherd is a 7B parameter "Critique/Critic" model (a la [[Constitutional AI|CAI]]), specifically tuned to critique responses and *suggest refinements*. It's similar in performance to ChatGPT (which is pretty impressive for a 7B model).

Related paper: Selfee (Ye et al., 2023) (which this paper references multiple times, and seems slightly worse)

----

Notes:
- Shepherd can pinpoint specific issues like ==factuality, logical errors, coherence, and alignment==, and suggest improvements via natural language feedback.
- To fine-tune and evaluate Shepherd, we created a high-quality feedback dataset composing of two distinct sets:
	1. ==Community feedback==, curated from online forums to capture more diverse interactions
	2. ==Human-annotated feedback==, collected on generations across different task types
	- Close inspection of influences of community feedback and human-annotated feedback confirms that the community data is more informative and diverse than human-annotated data, yet leans toward informality.
- Data Collection
	- (1/2) Community Critique Data: Authors want to gather clean `(question, answer, critique)` triplets. ==We consider the title and the sub-title of a post as a question, its top-level comments as answers, and replies to these comments as critiques.== Community vote score is calculated as (upvotes-downvotes).
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
				2. To manage offensive language in community data, we incorporate a profanity check, and eliminate comments with profanity scores lower than 0.8.
				3. Given our model is a text-only model, we filter out instances that contain URLs, images, or videos.
				4. We identify and remove comments that pose further questions to the original question rather than the original answer to preserve the integrity of the Q&A format.
	 - (2/2) Human Data collection (of critiques)
		 - For diverse sets of contexts and outputs, we select 8 popular language understanding and entailment datasets requiring complex reasonings and step-by-step explanations to arrive at the final answer, as well as two summarization datasets (all of them roughly have golden correct answers in the dataset, or we can derive one in the summarization cases).
			 - Entailment Bank (deductive reasoning), Proofwriter (logical reasoning), GSM8K (arithmetic reasoning), PIQA (physical reasoning), CosmosQA (commonsense reasoning), ECQA (commonsense reasoning), e-SNLI (deductive and commonsense), Adversarial NLI (adversarial entailment), GPT-3 summarization, DeFacto (summarization)
		- ==To collect high-quality natural language feedback, for each question, we provide a context, a correct output, a candidate output, and ask annotators to give feedback on whether there are any errors in the candidate output.== (The correct answer is provided to help annotators identify errors more easily).
		- Postprocessing
			- We remove examples flagged with "Errors in the correct output" and "The context is too complex to work on"
			- We remove feedback on error types of "Redundancy" and "Consistency with Context", because we find that feedback on these two error types aren't helpful.
		- We end up with 1317 high-quality examples in total.
- The ==[[Shepherd]] Model==
	- We train Shepherd as a fine-tune of [[LLaMA]]-7B (interestingly not LLaMA-2, which came out the month before this paper; maybe too late).
		- Authors provide [[AdamW]] optimizer settings and some other hyperparameters (LR, BS, MaxSeqLength).
	- They train for 3,000 steps, checkpointing every 50 steps, and then use a GPT-4 evaluation protocol to pick the best checkpoint on the held-out test set.




Abstract
> As large language models improve, there is increasing interest in ==techniques that leverage these models' capabilities to refine their own outputs==. In this work, we introduce Shepherd, a ==language model specifically tuned to critique responses and suggest refinements, extending beyond the capabilities of an untuned model to identify diverse errors and provide suggestions to remedy them==. *At the core of our approach is a high quality feedback ==dataset==*, which we curate from community feedback and human annotations. Even though Shepherd is small (7B parameters), its critiques are either equivalent or preferred to those from established models including ChatGPT. Using GPT-4 for evaluation, Shepherd reaches an average win-rate of 53-87% compared to competitive alternatives. In human evaluation, Shepherd strictly outperforms other models and on average closely ties with ChatGPT.


# Paper Figures

![[Pasted image 20240507224812.png|300]]

![[Pasted image 20240507232854.png]]
Above: I mean... in my opinion, the StackExchange response is a totally inappropriate critique/feedback, right? Pretty informal (as the authors noted) and caustic.

![[Pasted image 20240508002222.png]]


![[Pasted image 20240508002949.png]]
Above: It's hard to evaluate the feedback for many questions!

![[Pasted image 20240508003136.png]]
Above: This is the instruction for GPT-4 and human evaluators! I think this sort of GPT-4-based evaluation is what they use to determine which checkpoint to stop on, during training. I like collecting and comparing these rubric/criteria instructions for language models (eg like those used in Self Rewarding Language Models)

![[Pasted image 20240508003854.png|300]]

![[Pasted image 20240508004121.png]]

![[Pasted image 20240508004233.png]]

![[Pasted image 20240508004416.png]]



# Non-Paper Figures
- 
