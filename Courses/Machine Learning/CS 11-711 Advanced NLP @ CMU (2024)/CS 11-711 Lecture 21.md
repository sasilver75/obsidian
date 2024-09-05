# Topic: Complex Reasoning
https://www.youtube.com/watch?v=mPd2hFmzjWE&list=PL8PYTP1V4I8DZprnWryM4nR8IZl1ZXDjg&index=19

---

![[Pasted image 20240618141143.png]]
Formal Reasoning: Focusing on strict truth values (you can definitely say this is true, or definitely say this is not true -- rare in real life besides in things like Math)
Informal Reasoning - Based on experience and intuition; originally incredibly intractable, but we're starting to get some of this from LMs.

Induction is sort of like a soft version of Deduciton. "Every single creature I've seen with wings is a bird -> All creature with wings are bird". It's not necessarily true/logically-entailed, but you make a conclusion without it being formally verifiable.
Abductive means predicting the most likely explanation.

![[Pasted image 20240618141357.png]]
Even when you read descriptions about these various types of reasonings, the definitions are a little vague; take these as general directions and not strict rules.

## Pre-LLM Reasoning Methods

Computational Semantics
- Does derivational reasoning by starting out with certain premises, and getting to final conclusions.
![[Pasted image 20240618141551.png]]
We have symbols like "For All" and "There Exists"
- Neural networks can sort of do this using methods like CoT, but it's a rough approximation and doesn't always work well.
	- Recently there have been search algorithms through search spaces (eg math theorem proving) that use Neural Models to speed up the search; this is combination of more classical and modern methods.
- With Prolog, you can take a knowledge base and ask a knowledge base: "Do all people who work at CMU as professors have a PhD?" and you could actually examine that based on a knowledge base... 

Memory Networks (Sukhbaatar 2015)
- Memory networks have the ability to write and read from memory... basically, given a query, get the embedding of a query, attend over the memory/get a summary of the memory over a large memory base... and then later have the ability to go in and update the memory. So you can read/write from memory base.
- ![[Pasted image 20240618142236.png|300]]
- One of the big issues with LMs nowadays is that they don't get to update their memory; one way they *can* sort of do this is encoding that memory in text in their context, but there are limits to that and it's not likely that text is the best format for memory.


An idea that's been around for a while is solving questions using symbolic reasoning
![[Pasted image 20240618142335.png]]
If you have some text... and based on the text, run some symbolic operations like find, filter, find-max-num, relocate... some of the things that NNs are bat at are things like finding the largest number in a dataset, or finding places where X does or doesn't apply.
- People have been focusing on prompting techniques to do this sort of reasoning, but these older strategies are things that are important to consider when thinking about building systems, because current models still aren't especially good at them.

# The main event: Stuff that people are actually using these days


## Chain of Thought and Variants

![[Pasted image 20240618143321.png]]
To review:
- The basic idea is that, compared to standard prompting, we ask the model to come up with reasoning traces before coming up with an answer.


[[Self-Ask]] (Press et al 2022)
- LLMs aren't very good at asking follow-up questions -- or rather, they just haven't been trained to do it.
	- ChatGPT never asks a followup question; OpenAI must just think that it's a bad experience for a language model to ask a follow-up questions to your query, before producing a response.
- Self-Ask basically prompts the model to ask follow-up questions
- ![[Pasted image 20240618143548.png]]
- In this particular paper, it's really just another variety of [[Chain of Thought|CoT]], in the sense that the model itself is asking the questions and then answering its own questions.

Chain of Thought with Retrieval (He et al. 2023)
- On the previous Self-Ask page, how is the model going to know how to answer the "When was superconductivity discovered?" question that it asks itself, without hallucinating? Retrieval can help!
- ![[Pasted image 20240618144327.png]]
- They use BM25 to retrieve k=10 documents to prompt the model with.


Multilingual Chain of Thought Reasoning (Shi et al 2022)
- The interesting thing about multilingual CoT is that we have a design decision: Do we want to just do CoT reasoning in the language that we're asking questions in, or should we use another high-resourced/trained language?
![[Pasted image 20240618144722.png]]
	- If we ask a question in Japanese, should the entire CoT be in Japanese, or is it okay to somehow do CoT reasoning in English, if the model is primarily trained on English?
	- Basically, the answer is to do it in English
		- ((I imagine it's dependent on the question! There are probably certain things expressed in Japanese that are hard to culturally translate to English, no?))


Complexity-based Prompting (Fu et al 2022)
- Interestingly, for some tasks, a larger number of reasoning steps is indicative of improved accuracy... 
![[Pasted image 20240618144915.png]]
In general, if you have a reasonably strong model, doing the explanation first and then doing the prediction is better -- this is because CoT... (and decompositional strategies) actually works.
- ((Note: CoT works well for problems that can be decomposed into a serial set of tasks))
![[Pasted image 20240618145115.png]]
- People observe that if explanations are longer, they're going to be better. This is kind of interesting, but if they give more reasoning steps, it tends to be more accurate.
	- These are naturally occurring reasoning chains; they didn't train the model 
	- Authors smapled multiple reasoning paths and then performed [[Self-Consistency]] (majority voting) over the *longer* reasoning paths.

People have generally noticed that if you explanation is wrong, you ultimate prediction is also going to be wrong...

## Systematic studies of reasoning in LLMs

![[Pasted image 20240618145535.png]]
- MIRAGE paper showed that some emergent abilities are a result of ~poor measurement practices ("Did it do the thing?")

![[Pasted image 20240618145943.png]]

[[Orca]]: Training Small Models for REasoning (Mukherjee et al 2024)
- Generates a large and diverse CoT dataset from GPT 3.5 and GPT 4
- 5M complex instructions + CoT explanations
![[Pasted image 20240618150203.png]]
Replicated in the OpenORCA dataset (MSFT didn't release the Orca dataset for whatever legal/competitive reason)


Chain of Thought Reward Models (Lightman et al 2023)
- Get human supervision on the individual steps of CoT
- Then we use that to train a Reward Model that predicts whether each step of a CoT derivation is good or not, giving feedback on each sentence.
![[Pasted image 20240618150326.png]]
- They then use this to train a CoT-style model; they train the model to generate CoTs, and use this RM to upweight models with good CoTs.
	- We don't need correct answers to train the model, this way!


Rule Induction with LLMs 
- Propose hypotheses, verify with symbolic verifier (Qiu et al 2023)
	- Takes input/output examples, we predicts a variety of rules, and then we evaluate it (either using another LM or using a symbolic evaluator). Then we can go back and do more hypothesis refinement, etc.
- Use hypotheses in CoT, and keep the ones that result in correct answers (Zhu et al 2023)
	- They extract rules with an in-context prompt... they surround the rules with an xml tag saying "this is a rule that should be extracted", and when you get something right, you extract it.
![[Pasted image 20240618151030.png]]

It seems like all of these have been applied so far on pretty toy examples...


![[Pasted image 20240618151521.png]]
The idea of this paper is that we want to learn differences between text collections
- In the paper, they give examples of reports from papers who took a real medicine and those who took placebo ones. Real doctors try to figure out if there's any consistent difference between people who took a placebo vs an actual drug, during medical trials.

