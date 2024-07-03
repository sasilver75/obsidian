Fine-Tuning Workshop 3: Instrumenting & Evaluating LLMs (guest speakers Harrison Chase, Bryan Bischof, Shreya Shankar, Eugene Yan)

----

![[Pasted image 20240528100812.png]]
If you're trying to improve your model, you want the tightest iterations/feedback possible, so you can try a lot of things.

At the heart of this is evaluations!

![[Pasted image 20240528100944.png]]
Let's break evals into three categories:
- Unit tests/assertions
	- Run quickly, evaluate something you expect
- LLM as a judge
	- Response from our LM is fed to an evaluator LM
- Human Evaluation
	- A person just looks at the output of a model

We'll talk about where their use in two (non-exhaustive) settings/applications:
- Writing Queries (eg SQL, Honeycomb query language)
- Debiasing Text (Removing subconscious biases)

![[Pasted image 20240528101123.png]]
Above: Example of debiasing texts

The number one piece of feedback we've gotten so far is that we want to go through code. This takes time, but we have so little time in the workshops. We'll put this in some videos on the course page so that it doesn't take time away from our limited time as a group in workshops.

![[Pasted image 20240528101252.png]]
Despite that, let's go over this code at the highest level
- This code uses Pytest, but we could do it without Pytest if we wanted.

![[Pasted image 20240528101502.png]]
It's natural to think "hey, I can't write unit tests for AI, it's spitting out natural language -- everything would require a human!"
- Most of the time in practice, we always find dumb failure modes... things that are going wrong with the LLM that can be tested with code. We always find through looking at the data rigorously enough.
- Picture: Client looking for User UUIDs being spilled in responses
- You want a systematic way of tracking the results of these unit tests to the database, so you know that you're improving.

You have some AI application that's working off data
- One simple approach we like to use is enumerating all of the features that the AI is supposed to cover
	- Within each feature, we have various scenarios that the LM is supposed to handle
![[Pasted image 20240528101752.png]]
In the feature that finds listings for you as a user, there are multiple scenarios

![[Pasted image 20240528101911.png]]
You can use LLMs to systematically generate test data for these features/scenarios that you want your LLM to respond to!

![[Pasted image 20240528101943.png]]
When you're first starting off, you should be using what you have; don't necessarily buy stuff, though the tools are getting really good. You can get started by using existing tools.

![[Pasted image 20240528102141.png]]

Dan: There are sort of two types of unit tests to think about too
- One where, when they fail, it's a show-stopper, like a normal unit-test.
- A second where, sort of like a Kaggle competition, you want the number of "passing" ones to increase over time as you continue developing.

![[Pasted image 20240528102452.png]]
LLM as a judge is popular, you but you have to make sure that you align your LLM as a Judge *to something*, so that you know you can trust your Juge!
- You should measure its correlation to some human standard that you trust.
- Hamel likes to use spreadsheets when possible
	- Over the course of a few weeks, Hamel gave the client a spreadsheet like this; he had the client critique queries as good or bad, and write down why they were good or bad.
	- Over time he aligned the model to this human standard

![[Pasted image 20240528102606.png]]
Over time, he got the LM-as-a-Judge to align with humans most of the time.


![[Pasted image 20240528102804.png]]
Above: [[Positional Bias]] of LLM as a Judge
There are a bunch of biases with LLM as a Judge
- Most of Dan's experience when using it, he's been unimpressed, which is contrary to Hamel's experience.


Human Evaluation
![[Pasted image 20240528102900.png]]
"Look at your data"

![[Pasted image 20240528103038.png]]


![[Pasted image 20240528103047.png]]
If you have this tight iteration loop with multiple forms of evaluation, you can change your system (prompting, finetuning, data labeling) fairly quickly.

But it's not so easy!
- Project: Take scientific images and write the alt text so that visually impaired people can still understand images.

![[Pasted image 20240528103316.png]]
(First model was the top model, then proceeding down the y axis)
- For each, they have humans rate the output
- They made steady improvement, and it seems when they switched to the fifth one, there was a drop, and then they eventually caught up
- If you look at this, you'd think "Oh, you should just stop the project at the fourth one."
- Pause and think: Why is this misleading?
	- Why is Pipeline 4 not as good as the ones below it?
	- All of the labeling has in a bespoke piece of software for labeling. They changed it after 4.
	- By the humans seeing better and better models over time, their standards went up over time too! So what, earlier in the project, they might have rated as a 6/10, later in the project, they thought it might be a 4/10!

==Look at your data! Even people who SAY they look at their data don't actually looks at their data as much as they should!==


![[Pasted image 20240528103723.png]]
- A Trace is a logged series of related events (eg log in, add item to cart, check out).
- In LLMs, traces are relevant, because we have things like multi-turn conversations, RAG, function calls, etc.

==It's important to remove as much friction as possible to viewing your data. If there's too much friction, even YOU won't look at your data!==
- You need to be able to filter and navigate your data. 
- This is one of the ==most important== things you can build!

There are a lot of ways to render and log traces
- Langsmith
- Pydantic LogFire
- BrainTrust
- W&B Weave

OSS:
- Instruct
- Open LLMetry


![[Pasted image 20240528104322.png]]


## Harrison Chase from LangChain on LangSmith
- [[LangSmith]] works with and without [[LangChain]]
- The first thing that's important is looking at your data
- LangChain integrates into your project with 1-2 environment variables
	- Can set a decorator on your function, or log psans directly, etc.
- You log them into a project; on LangChain you can then see all of the things that were asked, and see exactly what's goign on under the hood
![[Pasted image 20240528104618.png]]
We can see that we made a call to Google LLM; we can see that we passed that to a retriever, got a list of documents, and then passed it to an LLM to generate an answer!

We strongly believe that people should still be looking through their data
You can even go directly from the trace into a playground, where 

Another half of Langsmith besides observability is Testing
- You can upload examples manually, see what they are, modify them
- You can also import these from traces
	- Maybe we want to filter to things that were given a score of 0, and then add some of these to our dataset
- You can organize datasets on LangSmith into different splits, and test on various splits, etc.

There are a few features for LLM-as-a-Judge testing.

----

## Bryan Bischof
- Bryan is an MLE who wrote a RecSys book; he'll describe his approach to doing evals, etc. and will walk through his workflow.
-  ![[Pasted image 20240528105742.png]]
- Spellgrounds is their name for an internal library for developing evaluations

![[Pasted image 20240528105826.png]]


### Miscasts and Fizzled Spells (Mistakes)
- Thinking that LLM Evaluations are entirely new
	- Data scientists have ben mapping complicated user-problems to nuanced objective functions for over a decade; they might have good instincts here on measuring unpredictable outputs.
	- The nuance and beauty of your LLM outputs are ineffable, right? No! You can quantify performance.
	- Ex: For code-gen, execution evaluation is your friend. Run the code generated by the model, compare it to the target setup, and make sure the outputs have the same state. 
	- Ex: For summarization, check for retrieval accuracy
- 
![[Pasted image 20240528110142.png]]
An example of comparing the output of code-gen for datascience code. We massage dataframes and ask: "Is there any way possible that these DFs contain the same data?"
- The response from the agent may be a different DF shape, but as long as it has that one number in there, that's good enough for Bryan

Next up
Failing to include experts in use-case creation!
![[Pasted image 20240528110256.png]]

People fail to recognize that Product Metrics and Evaluation MEtrics are similar, but different!
- LOOK AT YOUR DATA, but don't mistake product metrics for evals
	- Product metrics give you intuition about evals, but these production logs aren't sufficient for building evals.
	- At scale, production data can tach me a lot about what eval version 2 should look like, moving forward.

![[Pasted image 20240528110534.png]]

![[Pasted image 20240528110648.png]]

"Don't ask a shoeseller what's going to help you dunk, ask Michael Jordan"
- I promise; you DO NOT need an evaluation framework until you start feeling the pain; THEN it's time to start shopping.
![[Pasted image 20240528110723.png]]
This is an example of evaluating in a flexible way of evaluating whether what the agent responds with looks like your target. PLEASE invest in the things that matter; don't invest in complicated integrations early.

Don't reach too early for LLM-Assisted Evaluation! There are no free lunches.
- Thy can give you directional metrics/hints.
- Once you've built REAL evals, LLM-as-a-judge can help you scale.
![[Pasted image 20240528110851.png]]
He plugs [[Shreya Shankar]] as doing the best writing here


Part 2: Moderating Magic (How to build your eval system)

![[Pasted image 20240528111032.png]]
DON'T get too excited about your RAG system unless you have clear baselines. 

RAG Evals:
- Looking at different rerankers, trying different embedding models

Planning Evals:
- For agent systems, planning evals are super important
	- Using a state machine? Treat it like a classifier and checks its choice at each step

![[Pasted image 20240528111129.png]]
How often can you get agents to respond with structured output? How can you tie agent relationships together with tightly specc'd API interfaces, and evaluate that cosnistency

A lot of agent chains end a wrap up or summary stage
![[Pasted image 20240528111200.png]]
Sometimes you'll get examples where the wrap up step talks about shit the agent never did! yikes!
- Don't serve everything about the entire workflow to the wrap up step

![[Pasted image 20240528111245.png]]
Don't be afraid to run historical events through a new treatment!

Bonus: Production endpoints minimize drift
- We thought: "I'll build a clone of a production system in my eval framework, so that I can keep things as simple as possible to production"
	- This didn't work; We don't want to build tightly coupled systems that aren't identical
- Make your evals framework directly connect to your production environment; make them endpoints and call them!
![[Pasted image 20240528111416.png]]
Make sure that every step is exposed, and be able to hook into every step of the workflow.

Question:
- Where are you running your tests (local, GHA?)
- Where are you logging?
Answer:
- We run Jupyter notebooks that are orchestrated. No asterisks there.
- This means that they're reproducible, if I go back to an individual experiment and say "that's weird," we can pull the individual eval logs and look at every response from the agent
- "We just do it in Hex"


=="You shouldn't be passing all your evals. Your evals should be targeting evals that succeed 60-70% of the time, or you're missing out on a signal of improvement!"==


---
## Eugene Yan on Metrics
- Lots of people are asking about what Metrics (it will all be code and graphs, but the notebooks are being dropped in the discord). This is all fully available.

How to evaluate summaries for factual inconsistency/hallucination

If you look at it, Hallucinations happen 5-10% of the time, and sometimes it's really bad.

Can frame these as a [[Natural Language Inference|NLI]] task
- Given a premise and a hypothesis, determine if it is
	- Entailment
	- Contradiction
	- Neutral

If we apply this to factual inconsistency detection, we say that "Contradiction" = factual inconsistency!


![[Pasted image 20240528112116.png]]

So if we can get the probability of contradiction, we have a hallucination detector model! 😮
- We can use this to evaluate other generative models, or even use it as a guardrail.

Data

FIB dataset
![[Pasted image 20240528112331.png]]
For each input (2 rows per input), we have a consistent and inconsistent choice.

It's really worth going over his lecture and the notebooks (which are in Discord)


---

## Shreya Shankar: Scaling up Vibe Checks
- 4th year PhD @ UCB in DBs and HCI studying data-related challenges in production ML pipelines with a human-centered approach.
- Buzzwords: Reasonable scale, data quality, LLM opes


![[Pasted image 20240528114538.png]]
There are all sorts of mistakes that can happen when we're talking about open-ended generation!

![[Pasted image 20240528114805.png]]
Generic-to-task-specific
- Generic: Common NLP metrics (eg BLEU) that don't really tell us how 
- Middle: We know of some good metrics for common architectures (eg faithfulness, relevance, context recall)
- Specific: A task structure that you want to be followed, fine-grained constraints, etc.

![[Pasted image 20240528114900.png]]
We want to move these to the upper right quadrant!


![[Pasted image 20240528115007.png]]
Above: Is this referring to LLM as a Judge


![[Pasted image 20240528115054.png]]
The challenge is coming up with the criteria for our evaluation, and good ways to implement criteria.
- Is Markdown might need coding experience
- Has professional Tone might be hard to evaluate; need LLM?

# I'M GOING TO PAUSE HERE. WATCH SHREYA'S VIDEOS LATER!





