Topic: Intro

----

2012 vs 2014
![[Pasted image 20240417182058.png|300]]
![[Pasted image 20240417182114.png|300]]

It's a very exciting time to be doing this right now. The industry interest makes 2012 look like small potatoes.

==Even since 2022, it feels like there's been an acceleration; Some problems that we used to focus on seem like they're less pressing. I wouldn't say that they're solved, but there's less of a focus on that -- which is exciting, because it frees us up to focus on more challenging things!==

Things that are blowing up:
- Natural language generation
- Image generation
- Code generation
- Search

Throughout this course, we'll use this as a test:
=="Which US states border no US states?"==
- It's a simple question, but can be hard for our language technologies because of that negation there."

1980: Chat80 was an incredible system that could answer complex questions like: "Which country bordering the Med borders a country that's bordered bhy a country whose population exceeds that of India", but if you asked it our highlighted question above, it would say "I don't understand!"
- Things that fell outside of its capacity, it would just fall flat!

2009: Wolfram Alpha hit the scene; meant to be a revolutionary language technology. To our amazement, it still gives the following behavior when we used our highlighted question: {returns a list of all of the US state} -- fail!

2020: Ada, the first of the OpenAI models, when asked it:
- "The answer is: No {begins babbling}"

2020: Babbage, later OpenAI model:
- "The US states border no US states. {babbling}"

2021: Curie, later OpenAI model:
- "A. Alaska, Hawaii and Puerto Rico
  B. Alaska, Hawaii, and the US Virgin Islands
  C. ..."

2022: Davinci-Instruct beta (It has *instruct* in the name)
- "Alaska and Hawaii"

2022: Text-Davinci-001
- "Alaska and Hawaii are the only US states that border no other US states"

A microcosm of what's happened in the field: ==A lot of time with no progress, and then in the last few years, a lot of progress seemingly out of nowhere.==

And now, in Spring 2023:
![[Pasted image 20240417182742.png]]
Fluent on all counts, from text-davinci-002 and text-davinci-003


Suggested readings:
- "On our best behavior" by Hector Levesque
	- We should come up with examples that test whether models deeply understand! *Inspired* by the [[Winograd]] Schema Challenge.
	- Technique: Impose very unlikely questions where humans have obvious answers
		- =="Could a Crocadile run the Steeple Chase?"==
			- You've probably never thought about this, but it's obvious
		- =="Are professional baseball players allowed to glue wings on their caps?"==

![[Pasted image 20240417182929.png]]
Uh oh, two confident answers that are contradictory, across closely-related models... interesting!

Now, it almost feels like Blade Runner!
- The Turing test has been long forgotten; we're trying to figure out what kind of agents we're interacting with by being extremely clever about how we use them.


## Benchmarks saturate faster than ever!

![[Pasted image 20240417190151.png]]
((Above: The lines that are nearly vertical are a little bit misleading; performance didn't increase from -1 to -.3 in 0 time. The line should just start at the earliest datapoint!))

Though this looks undeniably like a story of progress.

So what's driven all this progress?

...

[[Self-Supervised Learning]]
- The model's only objective is to learn co-occurrence patterns in the sequence that it's trained on.
- Alternatively: To assign high probability to the attested sequences in whatever data you pour in.
	- Requires no labeling, just lots and lots of symbol streams.
- When we generate from these models, it really involves just repeated *sampling* from the model.
- The sequences can contain anything
	- Not just language


The result of this being so powerful is the advent of large-scale pretraining, which begins in the era of static word representations ([[Word2Vec]], [[GloVe]])
- GloVe was notable because it released the pretrained model as well as the code used to generate it! We started to see pretraining as an important component to doing really well.

A big moment for contextual representations was the [[ELMo]] model in 2018. The gains they reported from fine-tuning their parameters on hard tasks for the field were just *mindblowing* to researchers at the time.

The next year, [[BERT|BERT]] came out -- the paper had a huge impact by the time it was even published; they too released model parameters. The first of his sequence of things based on the [[Transformer]] architecture.

[[GPT]] came out
...
[[GPT-3]] came out, showing pretraining at a scale that was previously unimaginable! Also model size was huge -- The largest BERT, `BERT Large` was 340m parameter count. GPT-3 was ***175B***
- We also started to see incredible emergent capabilities ([[In-Context Learning]])

![[Pasted image 20240417191716.png]]
There's an undeniable trend here in terms of parameter counts; and note that there was a small number of people able to compete at this level, which caused a lot of consternation.

![[Pasted image 20240417192120.png]]
When you prompt a language model, you put it in a *temporary state* and then *generate a sample* from the model.


![[Pasted image 20240417192606.png]]
A model learning what "nervous anticipation" means through In-Context Learning


[[Reinforcement Learning from Human Feedback]]
There are two details that are important to us right now:
1. In a phase of training these models, *humans* are given inputs, and asked *themselves* to produce good outputs for those inputs.
	- This can be highly skilled work that depends on a lot of human intelligence.
2. The model, after training on those examples in a supervised manner, then produces outputs, and humans again *rate* those outputs, and we use that as signal to further train the model.


Short explanation of [[Chain of Thought]] eliciting some latent ability from the model.
- This is in large part a result of this model being instruct-tuned -- people taught it about how to think about prompts.


---

High-Level Overview of the course
1. Contextual representations
2. Multi-domain sentiment analysis
3. Retrieval-augmented in-context learning
4. Compositional generalization
5. Benchmarking and adversarial training/testing
6. Model introspection
7. Methods and metrics

The core goal is to have final papers that can *go on* to be published as NLP papers
- In 10 weeks, almost no one can create a publishable paper, but you can form the basis that can be continued to be developed into a great paper.

CS224N is a prerequisite for this course!

We'll talk about the following, in the context of Transformers:
1. Core concepts
2. Architectures
3. Positional encoding
4. Distillation
5. Diffusion objectives for these models
6. Practical pre-training and fine-tuning

![[Pasted image 20240420164059.png]]

~1:00:00+ in the video, he's given an incredible lambasting of LLM-for-everything strategies!
He's very bullish on [[Retrieval-Augmented Generation]] strategies to ameliorate many of these problems.


Retriever model
- Takes in text and produces text with scores
Langauge model
- Takes in text and produces text with scores

You can think of these as black-box devices that do input-out...
What if we just had them talk to eachother?
- That's what we'll do in the first homework!






