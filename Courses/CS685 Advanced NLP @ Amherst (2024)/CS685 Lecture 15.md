https://www.youtube.com/watch?v=Um9gf-U0o1Q&list=PLWnsVgP6CzafDszSy-njjdqnliv5qT0EW&index=15

# Topic: Evaluation

---

In Machine Translation

![[Pasted image 20240604161902.png]]
Above: Note ==Adequacy== and ==Fluency==

The performance of [[Machine Translation]] systems varies depending on the source and destination languages. 
- EG French -> English is much better than Chinese -> English

![[Pasted image 20240604162036.png]]
Above: There's probably some disagreement among the students when they'r rating these System 1-3 translations based on Adequacy and Fluency.
- Humans aren't calibrated well, and there's disagreement among humans
- Humans are also expensive, slow, and get tired easily!

![[Pasted image 20240604162210.png]]
Precision in the context of MT
Recall in the context of MT
F-Score in the context of MT

What are some issues with the evaluation above? 
- It's looking for exact match... and if there are any synonyms, you might get no points, even though you should have gotten points. (eg "officials" vs "administrators")
- It doesn't consider word order at all; we could have shuffled all the words in the System A response (scrambling it, to a human) and gotten the same score.

[[BLEU]] Score 
![[Pasted image 20240604162559.png]]
Introduced as basically a modification of precision-based evaluation for Machine Translation, where they alleviate the order-of-words problem by considering bi-grams, tri-grams, and four-grams in addition to unigrams.

![[Pasted image 20240604162804.png]]
The problem with including Recall (In BLEU?):
- If we have multiple references, then recall is meaningless, because we're asking the model to include all references from all N-grams, which doesn't make sense.

![[Pasted image 20240604163128.png]]


[[ROGUE]] is the recall-based counterpart to BLEU:
![[Pasted image 20240604163142.png]]

Just string-matching based approaches for quick evaluation, but you definitely shouldn't rely on them for which decisions to use.


---

Let's look at a different task that's not translation -- [[Question Answering]]

Question: Why are all boats white?

A common way to evaluate this task is [[ROGUE]], the recall-based version of [[BLEU]] -- but this is a ==completely terrible metric of this task== - you can imagine several ways of gaming this to maximize your ROGUE score.
- A simple baseline is to take the question and copy is 20 times.

![[Pasted image 20240604163450.png|400]]

It turns out that, in terms of ROGUE, this is better than many answers, and almost as good as human answers! 

Can we use Neural models that are directly trained/finetuned on human judgements?

[[BLEURT]] was one of the first 
- (Doesn't have anything to do with BLEU, besides being useful for MT evaluation)
![[Pasted image 20240604163724.png|300]]
- After perturbing a sentence, train the model to predict the similarity of these two inputs z, z'. They initially trained BERT to predict the BLEU score, ROUGE score, and many other metrics of similarity.
- They then fine-tune the resulting model on small supervised datasets of ==human quality judgements==!

It does a lot better in terms of correlation with human judgements than BLEU or ROGUE do.

---

Okay, but what about a trickier problem, like open-ended text generation?

![[Pasted image 20240604164349.png]]
It's harder to evaluate these, but ==there IS a difference between good stories and bad stories!==
- Subjective and dull
- Bad grammar
- Offensive/toxic

![[Pasted image 20240604164422.png]]
We can do something like ChatBot Arena and ask the human to *explain their reasoning*, and then based on reading through those explanations, you might be able to identify systematic issues with the model.

![[Pasted image 20240604164534.png]]
You could use something like MTurk, and give them a rubric of sorts where they score on various categories on 1-5
- But predictably this isn't a very reliable way of getting an evaluation!
- People who do these tasks for money are incentivized to do the most evaluations they can in the shortest amount of time possible.

The lecturer's lab looked at this in 2021 @ EMNLP: "The Perils of using Mechanical Turk"
![[Pasted image 20240604164656.png]]
It turns out that English professors take 7x longer than these MTurk workers, and if one of *us* did the task, it would probably take a lot longer than English teachers to grade papers.

![[Pasted image 20240604164815.png]]
It's hard to determine which is a better response, especially if you aren't an anesthesiologist. How can we possibly evaluate the quality of the texts when we aren't experts? 
- It turns out the left one is human-written and the right is text-davinci-002
\
----

Can we use LLMs to evaluate generations?
- Prompt to say "how coherent is a summary from 1-5?"
- Prompt to break down a generation into chunks and rate the chunks, then aggregate.

Below: GPTEval
![[Pasted image 20240604165415.png]]


We can also use LLMs as a judge to estimate ==Win Rate==; this is the ==most popular way that LLMs are used to evaluate, nowadays==, and it's just another prompting approach.

![[Pasted image 20240604165656.png]]
Given two generations, which is better? 
This is an example of an [[LLM-as-a-Judge]] in a pairwise ranking context, I think.
[[AlpacaEval]] is an example of this, where we use GPT-4 as an annotator model. We'll use a baseline model, and then {some other model} you want to evaluate. The Win Rate is just "How often did GPT-4 prefer the generations of {candidate model} over the Baseline model (Which is Davinci003)"

There are a variety of known biases with LMs as a judge, including:
- [[Positional Bias]]
- [[Self-Enhancement Bias]]
- [[Verbosity Bias]]


We want cheap, fast, and reliable, and it seems like we can't have everything in every situation.


![[Pasted image 20240604170455.png]]
Obviously if I ask a normal person how factual this generation is, they wouldn't really know.
- But we can use the same decompose-and-verify framework, but implement it with language models!

![[Pasted image 20240604170532.png]]
(FActScore paper): Train a model to produce, from a piece of text, the list of claims in that text. Each of these is an independent, atomic claim in that text.
- It's not a hard task, we're basically rewriting text in the format!
- We can then verify each of them using LM prompts.
	- We might need good retrieval to find documents, based on these queries.

Above: 6 of the 9 atomic facts are supported by Wikipedia!

![[Pasted image 20240604171310.png]]
Above: This would have cost $26K if done by humans! But they were able to do that for a small fraction of the cost by automating the process with LLMs.
