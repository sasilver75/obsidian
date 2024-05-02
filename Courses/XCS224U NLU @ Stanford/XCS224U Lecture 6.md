Link: https://youtu.be/l_w05N0QGLk?si=ITYADh2UCvJuxkP5

Topic: Advanced Behavioral Evaluation of NLU Models

-----

Let's reflect on the variety of evaluations that we conduct in the field, and broadly.

Let's just focus on the ==Behavioral Evaluation== of our model -- just focusing on the inputs/outputs, and not caring about how to do that mapping.

Later, we'll go a layer deeper to understand how models manage those input/output mappings..

# Part 1: Overview
## Varieties of Evaluation
- ==Behavioral==: Do models produce the desired outputs, given inputs? We don't care how they do the mapping
	- Standard: ("IID"; Independent and Identically Distributed)
	- We have test examples disjoint from the ones we were trained on, but there's some underlying guarantee that the test ones are similar to those that we saw during training
	- Exploratory
		- We might start to venture outside the IID assumption; we might know or not know what the training data was like.
	- Hypothes-Driven
		- We probe to see if the model has capabilities that we'd expect/be curious about: "Does my model know about synonyms, or lexical entailment?". 
	- Challenge
		- Things we *know* should be difficult, given the known training experiments of the model.
	- Adversarial
		- You did a full study of the training data and properties of the model, so you construct a test to reveal weaknesses to highlight a need for improvement.
	- Security oriented
		- Examples that might fall outside of normal use interactions with model (eg unfamiliar characters). 
		- Want to see if with very out of distribution inputs, the model does anything toxic or unsafe.
- ==Structural Evaluations== (the topic of the next unit): We try to understand the causal mechanisms behind model behaviors.
	- Probing
	- Feature Attribution
	- interventions

## Standard Evaluations
1. Create a dataset from a *single process*
	- Scrape a website
	- Reformat a database
	- Crowd source examples
2. Divide the dataset into disjoint train and test sets, and set the test set aside, not looking at it until the end, where we'll use it to evaluate generalization ability
	- We have a guarantee in some sense that the test examples will be like those we saw in training, which is very kind to our model.
3. Develop the system on the training set.
4. Only after all development is complete, evaluate system based on the accuracy on the test set
5. Report the results as as providing an estimate of the system's capacity to generalize

Worry is that ==we create a dataset from a single process, and then provide our test score as an estimate of the generalization capability of the model, even though the test set came from the same process as the training set.==


## Adversarial Evaluations
- The idea is to expose some of the fragility of the standard evaluations

1. Create a dataset by whatever means you like
2. Develop/assess the system using that dataset, according to whatever protocols you choose.
3. Develop a ==new test set dataset== of examples that you suspect or know will be challenging, given your system and the original dataset.
4. Only after all development is complete, evaluate the system based on accuracy on the ==new test dataset==
5. Report results as providing an estimate of the system's capacity to generalize.

We've created some hard and diverse new test sets. It's sort of a call for action... to feel like you can get behind step 5, you should construct these adversarial datasets in a way that ==covers as much of the user behavior spectrum== as you expect the model to possibly see when it's deployed.

This is the mode that we should be in when we think about AI systems in today's era of ever-widening impact.


## Winograd sentences
- Ambiguous sentences that are difficult to answer for models, but obvious for humans, because we have world knowledge.
![[Pasted image 20240502142517.png]]


## Levesque's 2013 adversarial framing
- "Could a crocodile run a steeplechase?"
	- The intent is clear:
		- The crocodile has short legs
		- The hedges in a steeplechase would be too tall for the crocodile to jump over
		- So no, a crocodile can't run a steeplechase
	- The idea is that we humans are simulating this in our head. It's an attempt to foil cheap tricks.
	- This, back in 2013, was a call for constructing these adversarial datasets that Chris Potts pines over.

# Part 2: Analytical Considerations 

- Key questions:
	- What can behavior analysis tell us about the behavior of our systems?
	- What CAN'T behavioral testing tell us about the behavior of our systems?

## No need to be adversarial
- Here are some questions that start off exploratory and end up being adversarial:
	- Has my system learned anything about numerical terms?
	- Does my system understand how negation works?
	- Does my system work with a new style or genre?
	- The system is supposed to know about numerical terms, but here are some test cases that are outside its training experiences for such terms.
	- When applied to invented genres, does my system produce socially problematic (eg stereotyped) outputs?
	- Are their patterns of *random inputs* that lead my system to produce problematic outputs.

## Limits of behavior testing
- You're at the limit of the set of examples that you've decided to construct.
	- "The limits of scientific induction"
- If you only have a blackbox understanding of the model (it's an even or odd classifier!), you might not understand how to correctly test it to highlight its weakness (oh, it's just a finite lookup table under the hood! I just need to go outside its domain).

## Model failing or dataset failing?
- When we see a failure, is it a failure of the model, or of the underlying dataset?
- Liu et al, 2019 "Inoculation by Finetuning" paper:
- ![[Pasted image 20240502144451.png|300]]
- Dataset Weakness vs Model Weakness
	- People want to claim that they've found model weaknesses (If you can show that the Transformer architecture is fundamentally capable of representing some phenomenon, that's an important result), but it's more likely that you've found a dataset weakness, which is a less interesting result where we just need to supplement training with more targeted data.
- ![[Pasted image 20240502144610.png]]
- When we say "fair," we don't worry that the model is being mistreated; rather, we're worried about an analytical mistake where be blame a model for failing a test, when the blame should be on us, because we didn't give sufficient information for *any* model to solve it.
	- 3, 5, 7, ...   {Guess the next number!}
		- Should we say 9, because it's odd numbers?
		- Should we say 11, because it's prime numbers?

![[Pasted image 20240502144821.png|400]]
Given the training data (left), do they mean the Material Conditional (top), or the Injunction (bottom)? We can't scold systems for not being able to read our minds!

Dataset vs Model Weakness
- If we finetune on a few challenge examples, then retest on both the original and challenge datasets
![[Pasted image 20240502145213.png|200]]
- Three outcomes:
	- ==Dataset weakness==: we see good performance on both the original and challenge dataset (challenge performance increase); Indicates a gap in original training data of the model.
	- ==Model weakness==: Even after finetuning, we still see poo r performance on challenge dataset. There might be something about our new examples that are fundamentally difficult for this model.
	- Annotation Artifacts: By finetuning on this performance, we've *hurt* the performance on the original set of the data. This might make us stop and reflect on the nature of the promise we've proposed.


## Negation as a learning target

- Intuitive learning target
	- If A entails B, then not-B entails not-A
	- If "Pizza" entails "food", then "Not food" entails "not pizza"

Observation:
- Our top-performing NLI models fails to achieve this learning target!
	- *Tempting* conclusion: ==Our top-performing models are incapable of learning negation!==
	- But we have to pair this with the observation that *negation is severely underrepresented in the datasets behind the NLI benchmarks*! -- It might be a dataset weakness, not a model architecture weakness!
	- So we follow the "innoculation by finetuning" template (most recent picture above)... did our best to pose this in the NMoNLI dataset that we created... and we use this now as  challenge dataset.

Initial results:
![[Pasted image 20240502145812.png]]
BERT did great on SNLI and good on the Positive part of the MoNLI split, but essentially zero accuracy on the Negative part of the MoNLI split; the model is basically just ignoring these negative cases
But with some innoculation by finetuning:
![[Pasted image 20240502145852.png]]
Yay! We've maintained performance on SNLI and now have better peformance on NMoNLI, which suggests that ==it was a data weakness, not a model weakness!==

# Part 3: Compositionality

This is a principle that's important to Chris as a linguistic semanticist, and it's  a prerequesite for understand the COGS and ReCOGS benchmarks!

Compositionality Principle
- ==The meaning of a phrase is a function of the meaning of its immediate syntactic constituents, and the way they are combined.==

![[Pasted image 20240502150237.png]]
Says that the meaning of S is FULLY DETERMINED by the meaning of NP and VP, and recursively, NP is FULLY DETERMINED by the Det and N parts, and at the leaves, the recursive process grounds out, and the meaning is that you just have to learn the lexical meanings of the words in the language you speak.
*The meaning of the whole is function of the meaning of the parts, and how they combine.*
- ((But when you get down to a leaf, we know that words don't have a singular meaning, so... seems wrong. Ah, later they sort of escape hatch it by saying that words actually have "function" meanings that can be combined))

Why do linguists tend to adhere to the compositionality principle?
1. As semanticists trying to study language, we might try to model all of the meaningful units of the language.
	- This means there's a lot of abstraction around lexical semantics; What happens in practice is that the ==meanings assigned are functions==.
	- So in practice, *every* above delivers a function, than, when combined with the word *student* (which delivers a function too), yields a function for the NP bit, etc...
2. "Infinite" capacity of language (but humans are finitely able to understand language!)
3. Creativity
	- We have an impressive ability to be creative with language. By and large, the sentences you produce today are unlike ones that you've created or encountered before (perhaps even by anyone in the human histroy!), but nonetheless we're able to understand what they mean.
	- Compositionality can be seen as an explanation for that.
4. Systematicity

## Compositionality or Systematicity

![[Pasted image 20240502150716.png]]
Systematicity: The ability to produce/understand sentences is linked to the ability to understand other ones.
- if you understand "Sandy loves the puppy," you understand "Puppy loves sandy"... There's an instant explosion in the number of things that you *know* as a consequence of learning new things.
![[Pasted image 20240502150815.png|200]]

## COGS and ReCOGS
- Two benchmarks designed around assessing compositionality of language models
![[Pasted image 20240502151414.png]]

I could not give less of a fuck about this shit, going to skip.

I realize now that the rest of the lectures are mostly about evaluation

The remaining interesting one might be about The "Methods and Metrics," where I imagine he's going to talk about different scoring metrics.










