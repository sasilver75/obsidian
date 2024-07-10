Or: Everything you didn't realize to needed to ask about LM Evals (an opinionated summary)
From: Hailey Schoelkopf
- Research Scientist at [[Eleuther|EleutherAI]], maintainer on [[Eleuther LM Evaluation Harness]], originally started with the goal of 1:1 reproducing the evaluations from the GPT-3 paper, and has since grown to many other evaluations. It's used as the backend for the [[Open LLM Leaderboard]] from [[HuggingFace]].

---

Agenda
- Why is LM evaluation so hard?
- Crash course on how evals are commonly done under the hood
- Takeaways and minimal best practices

---

## Why is LM evaluation hard?
- Scoring is difficult
	- ![[Pasted image 20240625124500.png|400]]
	- One of these answers is correct, and the other isn't; you or I can easily eyeball this and say that one is correct and the other isn't, but when we do automated evaluation, how do we reliably get a measure of model performance?
	- The phrase Joe Biden appears in both answers, but with different meanings!
	- Some temping shortcuts/tactics, but there are always tradeoffs, nothing is a perfect solution (reliability, cost, brittleness)!
- Reproducibility is painful

- Ways to evaluate an LM
	- Three ways that we can use an LM/ways that we can obtain measurements of its capabilities/characteristics
		- ==Log likelihoods==
		- ==Perplexities==
		- ==Text Generation== 
	- 1) ==Log Likelihoods==
		- LMs emit *logits* that can be converted to probabilities or log probabilities over the vocabulary. To generate text, we use a softmax to turn our logits into a probability distribution, then sample from this probability distribution at each timestep.
		- Note: An LM can give P(x_n|X_0...x_n-1) simultaneously, in reality; when we train models, we can have the model guess for each word in a sequence what the next word will be. Note that logits L are of shape [SequenceLength, VocabSize], predicting the model's "guess" for the next token at each step.
		- So now that we have logits that we can turn into probabilities, why is this useful? 
			- Given an input string X and a gold label output string Y, we can tell how likely it is for our model to generate that "correct" Y.
			- ![[Pasted image 20240625125533.png|400]]
			- We can do this with one pass through the model by taking the concatenation of X+Y, and then passing it through the model to get some logits across the X+Y sequence.
			- Then we can look at the probabilities associated with token 0 of Y, token 1 of Y, and so on, and either sum (for log probabilities) or product (for probabilities, though this will give very small numbers for long tokens).
		- Why is this useful for evaluation? We can use it to evaluate our model on a ==multiple choice== question! Multiple Choice a common way of doing LLM evaluation because it's pretty cheap, and there's no way of the model giving an "invalid" response; it will always implicitly "pick" one of the answers, since we're just passing (eg) four (X+Y) sequences through, and doing some math.
		- ![[Pasted image 20240625125752.png|400]]
		- ![[Pasted image 20240625130047.png]]
		- It's possible that you model is good at MC, but can't generate long-form chains of thought, meaning it perhaps isn't going to be good at downstream real-world use cases like being a chatbot.
		- ==It's also a disadvantage that chain of thought CAN'T BE USED, given that some models are trained to use CoT.==
	- 2) ==Perplexities== ([[Perplexity]])
		- We're trying to measure how well a model fits a given data distribution.
		- We have a dataset; collection of documents that are just sequences of works. We measure the log probability that the model assigns to the next token... for every token in the dataset. We check how likely it was for the model output it correctly, and just average it over all of the tokens and documents in the dataset; Basically, per-token, how likely is the model to produce that data.
		- ![[Pasted image 20240625130236.png]]
		- This can be done trivially via self-supervision; we know we've got the labels. 
		- ![[Pasted image 20240625130512.png]]
		- ![[Pasted image 20240625130847.png]]
		- For both perplexity and log likelihood, one complication is that, because they take the sum over the number of tokens, or averages over the number of tokens, it really matters what tokenizer you're using! If there are fewer tokens to predict over, it might be easier to get a better perplexity score, for instance! There are some remedies for normalizing metrics with respect to tokenizers; the important part is that all of these small implementation details change what we're measuring.
	- 3) ==Text Generation==
		- For downstream use, we usually care about the generations in a free-form (eg chatlike) setting.
		- Using [[Chain of Thought|CoT]] is realistic/important for current models to use, especially in current models.... But there are downsides to doing this type of generation
			- We really don't know how to score open-ended generation well.
		- Often, we're forced to resort to heuristics for answer extraction.
		- ![[Pasted image 20240625131219.png|300]]
		- Another way people do this is to use LLMs to check if answers are correct, but these are also quite fallible.

There are other reasons that Generation can be finnicky
![[Pasted image 20240625131635.png]]
Which of these two do you think are going to give better performance?
- They look similar, but the two lines at the bottom... the second prompt ends with \n\n\t, if you look closely!
- HumanEval performance will be something like 5% worse on the right side! So the minutest differences can effect performance! If you evaluation trims off trailing whitespace in prompts, and others don't, you'll have divergent evaluations/understandings!

![[Pasted image 20240625131928.png]]
We need strong reporting standards, and for papers to report enough details about how evaluations were done.
- Basically, ==If evaluation code is not shared directly, it's likely not going to be reproducible.==
- This is where libraries like [[Eleuther LM Evaluation Harness]], [[HELM]], etc. come in; by having a vetted codebase for evaluation, it's great to not have to worry about these tokenization issues, etc... and more think about what evaluations you want to do.
	- In contrast to HELM, Eval Harness tries to put tools in the hands of researchers; many tasks are supported, and it's easy to define new ones, edit prompts, etc.


Distinction: Model Eval vs Downstream Eval
- Model Eval: eg [[Massive Multi-Task Language Understanding|MMLU]]; measures how "generally capable" your LM is, to measure against other base models
- Downstream Eval: I have a concrete use case in mind, and want to
	- The best eval is one that you run the code for yourself, or even one that you've designed yourself.
	- Find an evaluation where "hill climbing" is actually a *good thing* (hill climbing MMLU by checking it many times during finetuning doesn't mean that you model is getting better at what you actually care about)


![[Pasted image 20240625132656.png]]
![[Pasted image 20240625132817.png]]
Many of these lessons are in a recent Eleuther paper
	Lessons from the Trenches on Reproducible Evaluation of Language models












