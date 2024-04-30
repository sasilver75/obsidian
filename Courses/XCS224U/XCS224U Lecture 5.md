#lecture 
Link: https://www.youtube.com/watch?v=eyNLkiQ89KI&list=PLoROMvodv4rOwvldxftJTmoR3kRcWkJBp&index=20

# Topic: In-Context Learning

Takeaways:
-  ==The sophisticated things we're seeing from language models these days are not emerging from pre-training, they're emerging from standard supervised training.==

----

# Part 1: Origins of In-Context Learning
- All credit to the ChomskyBot for bringing us to this moment! Just kidding.
- In the pre deep learning era, n-gram LMs were often massive! In 2007 (Brants et al), there was a 300B parameter language model trained on 2 trillion tokens!
- decaNLP (McCann et al, 2018): Multi-task training with task instructions as natural language questions
- Radford et al (2018): Some tentative prompt-based experiments with GPT.

![[Pasted image 20240429193813.png]]
Above: [[GTP-2]] introducing zero-shot and few-shot prompting. They also evaluated text completion, Winograd schemas, reading comprehension, and others.

The cultural moment, however, arrived with [[GPT-3]]
![[Pasted image 20240429194115.png]]

# Part 2: Core Concepts

## Terminology
- [[In-Context Learning]]: A frozen language model performs a task only by conditioning on the prompt text. There are no gradient updates; the only mechanism we have is by inputting some text that puts the model in a "temporary state"(\*) that we hope is useful.
- [[Few-Shot Prompting]]: The prompt includes examples of intended behavior. Ideally, no examples of the intended behavior were seen during training (this might be hard to verify, though. The idea is that we're "learning," not just eliciting). (He labels this as few-shot in-context learning, which is slightly different from how I've termed it here)
- [[Zero-Shot Prompting]]: The prompt includes *no examples* of the intended behavior, but some instruction/description of what the intended behavior should be. Ideally, no examples of the intended behavior were seen during training (he labels this as zero-shot in-context learning, which is slightly different from how I've termed it here).

## GPT: Autoregressive Loss Function

![[Pasted image 20240429194612.png]]
The essence of this is that scoring happens on the basis of:
1. The embedding representation of the token for which we're creating an output, at timestep t
2. The hidden state that the model has created up until timestep t

## Autoregressive training with Teacher Forcing
- Just describing how if we mispredict a token vector, we replace it with the "correct" one at the next timestep.

## Generation
- [[Greedy Decoding]]
- [[Beam Search]]
- The point is that the decoding strategy that we use is separate from the structure of the model that we pick. The models just output probability distributions over a single next token. We're the ones that sample from that distribution and then autoregerssively feed the new sequence back to the model for another inference.

## A question to debate with your researchers and loved ones
- Do autoregressive LMs simply predict the next token?
	1. Yes, that's all they do
	2. Well, they predict *scores* over the entire vocabulary at each timesteps, and we *compel* them to then predict some token over another.
	3. Actually, they also represent data in their internal and output representations.
	4. But on balance, saying they simply predict the next token might be the best in terms of science communication with the public.

Many things that look like common knowledge are just high-probability continuations of sequences. Those sequences are high probability because they reflect some regularity in the word that humans choose to reflect in language that we then end up training on... But there are probably many things about the world that don't make it into our dataset.

A small bit on [[Instruction-Tuning|Instruction Fine-Tuning]]: These models are clever because they've been trained to emulate clever humans.

# Part 3: The current moment
- (Shows some of the usual datasets used for Pretraining, not a lot that's new to me)

## Data used for instruction fine-tuning
- We don't know much about what the industrial labs are doing here
- We can infer that they're paying lots of people to generate Instruction data, using human experts across many domains.
- REMINDER: ==The sophisticated things we're seeing from language models these days are not emerging from pre-training, they're emerging from standard supervised training.==
- Check out the [[Stanford Human Preferences]] dataset a resource for [[Instruction-Tuning]] from Reddit posts.

## Self Instruct
![[Pasted image 20240429200503.png]]
- We begin in [[Self-Instruct]] from 175 seed tasks written by humans.
- We have a LM generate new instructions, via in-context learning
- The generated instruction is fed back into the LM with a new prompt that helps the model know whether it's a classification task or not.
- Depending on above, we feed the generated output into one of two prompts below, which gives us ==new input-output pairs that we can use for subsequent LM pretraining== 
	- There's some filtering that makes sure that the input-output pairs are high quality and diverse
- In this way, we can use a LM to bootstrap a dataset, and update the model with the dataset, in the hope of giving it better abilities.

Let's zoom in on that:
![[Pasted image 20240429201050.png]]

![[Pasted image 20240429201104.png]]
The model learns in-context to learn if it was a classification task prompt

![[Pasted image 20240429201135.png]]
(A) If it was a Classification task

![[Pasted image 20240429201150.png]]
(B) If it was not a classification task

Results give us new input-output pairs that we can use to augment our dataset

## Alpaca
The [[Alpaca]] paper showed that we could use self-instruct methods to take small models (eg a small 7B [[LLaMA]] model), and get a stronger model as an output by training on text-davinci-003-generated data in a self-instruct framework.

![[Pasted image 20240429201555.png]]
(This lecture was before GPT-4)


# Part 4: Techniques and Suggested Methods

![[Pasted image 20240429202330.png]]
(eg maybe we retrieve some context passage)

![[Pasted image 20240429202810.png]]
[[Chain of Thought]]
- For complicated things, it might be simply too much to ask the model to produce the answer in its immediate tokens.
- With CoT, we construct demonstrations that encourage the model to generate "reasoning" in a step-by-step process, before arriving at an answer.
- We illustrate CoT with expensive, hand-built prompts
- At test time, the demonstrations cause the model to walk through a similar CoT, before hopefully arriving at a correct answer.

There's a more generic, less onerous version of this that probably be as productive:
![[Pasted image 20240429202838.png]]
The continuation above is revealing: a customer can have auto loans without having other loans.

![[Pasted image 20240429202912.png]]
We give a description of what the reasoning/prompt should look like, using an information markup language that the LM probably acquired during some instruct-tuning phase.


## Self-Consistency
 - Another powerful method is [[Self-Consistency]], and it relates to an earlier model called [[Retrieval-Augmented Generation]] (the model).
- We use our LM to sample a bunch of different generated responses, each of which might use something like CoT.
- Those answers might vary across the different generated paths that the model took
- What we're going to do is do is trust/select the answer that was most popularly produced, among the different reasoning paths. The most probable answer given all those paths is likely to be the correct one, is the idea.

### Self-Consistency in DSP
- In DSP, we have a primitive called DSP.majority that makes it very easy to do self-consistency.
![[Pasted image 20240429203145.png]]
Makes self-consistency a drop-in, assuming that you can afford to do all of the sampling that it requires.


## Self-Ask

![[Pasted image 20240429205137.png]]
[[Self-Ask]]: Via demonstrations, we encourage the model to break down its reasoning into a bunch of questions that it poses to itself, and then seeks to answer. The idea is that by doing this, it tries to find the answer.
- This is especially useful for multi-hop questions that might involve multiple different resources/subproblems that need to be resolve.
- Self-ask can be combined with retrieval for answering the intermediate questions (eg a search engine).

## Iterative Rewriting
![[Pasted image 20240429205151.png]]
- Another powerful general idea is that it can be useful to iteratively rewrite parts of your prompt! You can be rewriting demonstrations, parts of the passages they contain, questions or answers, etc.
- In the context of multi-hop search, we're gathering together evidence passages for a bunch of sources and synthesizing them into one, and using them to answer complicated questions.
- This idea is very general -- especially given a limited prompt window.
- Is it helpful for language models to iteratively rewrite its prompt to get better results.


## Some suggested methods
- Create dev/test sets for yourself, based on the task you want to solve, aiming for a format that can work with a lot of different prompts.
- Learn what you can about your target model, paying particular attentinon to whether it was tuned for specific instruction formats.
	- If you can align with the structure of the finetuning data, you get better results (we often don't know what it was like, but we can sometimes discover it in a heuristic manner)
- Think of prompt writing as AI system design
	- Try to write systematic, generalizable code for handling the entire workflow, from reading data to extracting responses and analyzing results.
- For the current (and perhaps brief) moment, prompt designs involving multiple pretrained components and tools seem to be underexplored
	- EG how a retrieval and language model can work together? But we could also bring in other components/capabilities like calculators, weather APIs, you name it.

