Or: Everything you didn't realize to needed to ask about LM Evals (an opinionated summary)
From: Hailey Schoelkopf
- Research Scientist at [[Eleuther|EleutherAI]], maintainer on [[Eleuther LM Evaluation Harness]], originally started with the goal of 1:1 reproducing the evaluations from the GPT-3 paper, and has since grown to many other evaluations. It's used as the backend for the [[OpenLLM Leaderboard]] from [[HuggingFace]].

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
	- The phrase Joe Biden appears in both answers, but 
	- Some temping shortcuts:
		- Multiple choice
		- Heuristics + String Matching
		- Asking anotehr language model (LLM as a Judge)
- Reproducibility is painful


## Crash course on 












