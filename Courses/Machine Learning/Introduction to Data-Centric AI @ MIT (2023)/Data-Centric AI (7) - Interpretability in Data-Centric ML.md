https://www.youtube.com/watch?v=w0Nn-SVYCL0&list=PLnSYPjg2dHQKdig0vVbN-ZnEU0yNJ1mo5&index=7

This lecture wasn't that good. It basically just says... try to create features for your models that make sense to normal people so that they can trust them. Don't use log(log(median(x))), just use like... median(x) or something.

----

If you start reading into the data-centric literature, you'll notice references to the concept of *interpretable features*; many algorithms claim to generate interpretable features, and users often clamor for the idea of interpretable features.

But what does this actually mean -- why do we care?

Agenda:
- Introduction to Interpretable ML
- WHY do we care about interpretable features?
- WHAT are interpretable features, really?
- HOW do we get interpretable features?

----
# Introduction to Interpretable ML
- In Machine Learning, we take data/features, input them into an ML model that does some math, and we get an output.
- Question: In the black box in the middle (eg a multi-billion parameters NN), what's happening?

Why do we need interpretable ML, if the model is doing well anyways?
1. ==Debugging and Validation==: If we've trained our model and it has good training and test performance, does that mean it will work in real world? Maybe!
2. ==Reviewing Decisions==: If we release the model into the wild and it does something bad (like your car hits a person), you need to know why, and fix it!
3. ==Improving Usability==: It's safe to say that in most domains, ML models aren't completely replacing humans -- the expert humans are *using* models. These expert humans need to understand *why* their model is saying "17" if that doesn't jive with their experience.


![[Pasted image 20240630131552.png]]
This model seems to show that people with asthma have a lower percent chance of dying from pneumonia, a respiratory condition -- this seems contrary to intuition.
- This is because when someone with asthma and pneumonia come into the hospital, they get *special attention* because the combinations of conditions is dire, so they *do* in fact often have lower mortalities.
- But if we trained a model that said "People who have asthma and pneumonia are fine, actually," then we'd have a bunch of people with asthma *dying* because they don't get the attention they need.

We need interpretable ML
- When the problem formulation is incomplete (and we don't know how to actually estimate the quality of our model)
- When there is associated risk (eg self-driving cars)
- When humans are involved in the decision-making loop.

# WHY do we care about interpretable features?
- This is a class on *data-centric* machine learning, so we won't be going into *model-interpretability*. But interpretability of models starts at the *feature* level, which is an aspect of *data*.

![[Pasted image 20240630132234.png]]
Decision Trees! People *love* decision trees because the appear to produce interpretable model outcomes. But we really don't understand *why* the model is making these predictions on house prices, according to the speaker -- $X[7]$ doesn't mean a lot, because in this example the features are a result of [[Principal Component Analysis]], so we don't really know what the features mean.
This might seem a bit contrived to some of you ((Yeah, just don't PCA)).

![[Pasted image 20240630133710.png]]
In this example, she uses an automatic feature creation algorithm to create features. The features are still a bit hard to reason about things like cosine of latitude... and a lot of these same things are popping up in different features. 
((This isn't a result of the model, this is a result of creating useful, but not easily interpretable features))

We might instead prefer something like this:
![[Pasted image 20240630134900.png]]
Which actually has pretty much the same performance as the previous one.

This is the kind of stuff we deal with in real world ML deployments, though!
- We learn that we need to explain our models
- We learn that interpretability of *features* themselves is important

Problems:
- Unclear language
- Irrelevant features

![[Pasted image 20240630135322.png]]
Plot from a DARPA paper that gets mentioned a lot, evidencing a "performance interpretability tradeoff" --==The plot isn't based on any specific data though==, it was just for example!
- In the real world, when we care about interpretability of the model and features, we find that we get:
	- More efficient training
	- Better generalization of models
	- Fewer adversarial examples resulting from spurious correlations

In many cases interpretability is non-negotiable ,while in other cases we just want maximum performance.



# WHAT are interpretable features, really?
- Let's go through what exactly it means for a feature to be interpretable. 

![[Pasted image 20240630141328.png]]
![[Pasted image 20240630141545.png]]
These are five features that we could use
Criteria:
- ==Readability of features==: Can I tell what the feature means by looking at it, or reading a description?
- ==Understandability==: Can I actually do logic about this feature? Something like a normalized measure of median income (.8/1) is likely less useful than the actual measure (eg $90,000).
- ==(Intuitive) relevance of features== is important too -- Users want to look and the model and say "Yes, this looks reasonable." ((It's so funny and such an engineer thing to think that the color of the house doesn't affecting the price that it sells at, lol))
- ==Abstract concept==: Have very digestible concepts, depending on your users. ((IMO average house size can be *more intuitive* than whatever "area quality" means. A good example might be College Rankings... where they have features like 'education quality,' 'sport quality', etc.))


# HOW do we get interpretable features?
For the most practical part of the talk

"[[Feature Engineering]] is the first step to making an interpretable model, even if we don't have a model yet." - Some data scientist
- If you think that your model *needs* to be interpretable, you can think about it from the get-go.

==Methods for Interpretable Features==
1. Including the users of the model
	- Flock, Ballet
2. Using interpretable feature transformations
	- Pyreal
3. Using interpretable feature generation
	- Some automatic feature generation algorithms keep interpretability in mind.
	- The "Mind the Gap" model, which can be used for binary features.

![[Pasted image 20240630141930.png]]
This is a big topic, but the idea is to include the end-users of your product in every step for the process.

Collaborate Feature Engineering with experts:
- ==Flock==: Choosing features through comparisons. 
	- Machine-generate features for a prediction task
	- Crowd-generate features
	- Cluster the crowd-generated features
	- Iterate on inaccurate model nodes
![[Pasted image 20240630142428.png]]
- ==Ballet==: Feature engineering with feedback
	- ![[Pasted image 20240630142443.png]]


![[Pasted image 20240630142746.png]]



![[Pasted image 20240630143140.png]]

![[Pasted image 20240630143144.png]]

![[Pasted image 20240630143204.png]]


![[Pasted image 20240630143207.png]]