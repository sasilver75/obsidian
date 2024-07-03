https://www.youtube.com/watch?v=WL2AlMIupC4&list=PL-Fhd_vrvisMYs8A5j7sj8YW1wHhoJSmW&index=5

# Topic: Multimodal Representations (Fusion + Coordination/Fision

Last week we covered primarily unimodal representations; let's talk about multimodal ones!

---


Agenda
- Multimodal representations
	- Primarily @ Cross-Modal *interactions*
- Representation fusion
	- Additive and multiplicative fusion
		- Tenso and polynomial fusion
	- Gated fusion (things that change given current input data)
		- Modality-shift fusion
		- Dynamic fusion
	- Fusion no raw modalities
	- Heterogeneity-aware fusion (to deal with very different modalities)
- Measuring non-additive interactions


![[Pasted image 20240614134922.png]]
Let's assume that we've got some data from varying modalities!
- Our whole goal is to learn representations of these modalities (and the fused modality) that can be useful for downstream tasks.


Challenge 1: Representation
- Definition: Learning representations that reflect cross-modal interactions between individual elements, across different modalities.
![[Pasted image 20240614135054.png|300]]
It single element from each modality (eg boiling an entire picture into a single representation; this can be seen as representation using holistic features).

Subchallenges in Representation:
- Fusion: 1 representation from 2
- Coordination: 2 representations, coordinated through some (eg) similarity fn
- Fission: Often 3 representations, with one A, one B, and one AB.
![[Pasted image 20240614135140.png|300]]

Interactions ONLY happen during inference!
![[Pasted image 20240614135231.png|300]]
Some examples of inference can be:
- Fusing them to produce a new representation
- Translation from one modality to anther
- Doing inference on some supervised learning task

![[Pasted image 20240614135413.png]]
Here's an example where the interaction (for the given query) is ==redundant==; in a sense -- but we *do* become more confident in our prediction. This is called ==Enhancement.==


![[Pasted image 20240614135405.png]]
Here's an example where the information across modalities in ==non-redundant== with respect to the query. In this case, when we combine them, one answer is correct, and the other is not correct -- this is an example of ==Dominance,== where one modality gives a correct answer that dominates the other.

![[Pasted image 20240614135516.png]]
An example where there's ==non-redundant== information, but the decision depends on whether you can work with a small table or big table, whether you're thirsty, etc. It depends on contextualizing both modalities, and can be different in different settings. This is called ==Emergence.==

![[Pasted image 20240614135706.png]]
The whole goal of representation fusion is to learn models that are able to capture these interactions.

We were categorizing the interactions via ==response== before, but we can also categorize by other aspects:
![[Pasted image 20240614135814.png]]


![[Pasted image 20240614140143.png]]
The latter (raw modality fusion) is "harder" to do, especially with modalities where you don't have good pre-trained unimodal encoders with which to do (eg) contrastive learning.
- Let's start with the basic fusion first!


![[Pasted image 20240614140251.png|300]]
Fusion using unimodal encoders
- Let's pretend we have some good pretrained encoders (eg for language, image)

Our historical view of fusion is that, before deep learning, there were two types of fusion:
- Early Fusion: We would extract the best features from each modality, concatenate these features, and use them to predict a label.
- Late Fusion: We would train individual predictors from each modality to the label, and take an ensemble/majority vote

![[Pasted image 20240614140401.png]]
- It's a good idea to start with unimodal modals/late fusion and see how it works for your project.

![[Pasted image 20240614141616.png]]
There have since been many fusion methods invented to learn cross-modal interactions.
- Simple univariate case; each feature is one dimension, and the target is also one dimension.
	- In this case, we might look at some simple linear regression/linear combination of the 

We can use these simple methods to test hypotheses:
![[Pasted image 20240614142623.png|300]]
Instead of a multiplicative model, we could train an additive model:
![[Pasted image 20240614142848.png|300]]

Or use a model that has both additive terms and multiplicative terms
![[Pasted image 20240614142910.png|300]]

In summary:
![[Pasted image 20240614143034.png]]

![[Pasted image 20240614143246.png]]
We can do additive fusion of either the raw modalities or the encoded modalities.

We can use multiplicative fusion to make it more expressive to (elementwise multiplication)
![[Pasted image 20240614143325.png]]
The matrix is richer; it has all pairwise relationships; this is called Bilinear Fusion, and can be achieved by an outer product between x_A and x_B.

![[Pasted image 20240614143459.png]]
A trick to get both the original modalities AND the fused modalities is to append a 1 behind xA and xB; we do the same bilinear outer product, and we get a z matrix where the individual xA and xB features are preserved as a column and row.
- You can do it for three dimensions too! That gives you the pairwise bimodal interactions as well as the trimodal interactions!

==But this weight matrix may end up as quite large...== enter Low-Rank Fusion!
- Any matrix can be decomposed into a sum of individual outer products
![[Pasted image 20240614143638.png]]
Can be decomposed into
![[Pasted image 20240614143659.png]]
We can approximate our W matrix using two lower-ranked matrices. 

![[Pasted image 20240614144611.png]]
Generalizing what we said: We can define a hyperparameter P and the model will learn interactions up to order P.

## Gated Fusion
- So far, the things we've seen so far, the weight matrices w1(xa) w2(xb) are all fixed... but here in gated fusion, the weights depend on the data themselves! The weight matrices are *functions* of xA and xB
	- gA and gB can be seen as attention functions.
- ![[Pasted image 20240614144723.png|300]]

Two interpretations:
- Can be thought as a model that (equivalently)
	- NN designed to mask unwanted signal from propagating forward (Gating)
	- NN that selects hte most prefereable signal to move forward (attention)
![[Pasted image 20240614144812.png]]
Hard attention can be harder to optimize, but gives much sharper attention (sparse). These inputs can be based on the modalities we started with (eg self attention), or it can be the other modality... or using both modalities (cross-modal attention in transformers).

## Modality-Shifting Fusion
- Like Gated fusion, but makes the assumption that one modality is more powerful, and the other modality is meant to help the other modality (an example of that these days is images helping out the stronger language model)
- Here, it seems that we use gating functions only for the secondary modality(s).
![[Pasted image 20240614145042.png|400]]

## Mixture of Fusions
- We can use multiple fusion strategies (if we don't know what to choose) each with a different gate.
	- The gating can be with soft or hard attention.
- This is basically admitting that you don't know which fusion strategies to prioritize, so you'll do it in a data-driven way.
![[Pasted image 20240614145129.png]]


## Nonlinear Fusion
- We just take our models, concatenate them as early as possible, and define a big model (NN transformer), and let the model figure out what fusion strategy to use based on our data.
- We can define many different fusion strategies
![[Pasted image 20240614145224.png]]
The question becomes: We've talked so much about inductive biases of fusion; if we just define a neural network, will they learn the same things (as if we used inductive bias?)

Speaker then sort of goes off the rails nerding out about all of the different ways that you can mix modalities
![[Pasted image 20240614150112.png|300]]


![[Pasted image 20240614150736.png]]
For modalities that are more heterogenous in nature, it might be unknown how much unimodal processing you might want to do before fuse.

![[Pasted image 20240614150904.png]]
Another way to figure out when to fuse modalities


Some modalities can be very similar with eachother, and others can be very different
- How well does my modality transfer to other modalities? After training on modality X, how well can I zero-shot when transferring to modality Y?
	- Some well
	- Some terribly
![[Pasted image 20240614151033.png|300]]
This can give you an estimate of how heterogenous two modalities are, which will inform your strategy about how to fuse them!


At this point at the end of the lecture, he's just throwing research papers at us
![[Pasted image 20240614151825.png]]


For different modalities, the raw data might be susceptible to different types of noise (Eg noise specific to cameras, time-series), or missing modalities (if we have a video, and we lose connection, etc.)
- There are several approaches towards more robust models
	- Introducing noise during training, based on the expected noise we might see during testing
		- typos, random drops in videos, camera blurs, etc.
	- Infer missing modalities
		- Try to use modality A to predict modality B, if we know that modality B is going to be noisy, missing, or not have perfectt raw data.


## Wrapup

We've looked at lots of methods for represnetation fusion
- Key idea: We have two elements from differnet modalities, and dwant to learn a joint representation that models cross-modal interactions between our individual elements of varying modalities.

![[Pasted image 20240614152410.png]]
Above: I think there were a few additional methods maybe on the right? But the video got cut off.

----

# Lecture 3.2: Multimodal Coordination and Fission

Last week we learned about Fusion

![[Pasted image 20240614152956.png|300]]


Coordination: 
- Learn separate representations for each modality that are coordinated using some sort of Similarity function

Fission:
- Looking at learning more representations than we started with, perhaps looking at clusters/factors that exist in the data.

## Representation Coordination
- Two modalities that are coordinated; two elements share some similarity, and we want to enforce some similarity across the representations.
	- Strong Coordination: Enforcing similarity a lot
	- Partial Coordination: Only enforcing similarity a little, or in some dimensions
![[Pasted image 20240614153145.png|300]]

A common way for doing this is taking our elements, passing them through encoders to find features, and applying some coordination function that measures some amount of similarity between representations.

A general learning paradigm is to pass elements through encoders, represent similarity with a coordination function, and then use that as our loss function (eg maximizing similarity between representations)
Coordination functions 
- ==Cosine similarity== (strong coordination)! We want our two representations to have a high dot product/are parallel in our representation space.
	- Normalized so that we don't have arbitrarily high dot products just because the vectors have high magnitude.
- ==Kernel Similarity== functions (linear kernel, polynomial, exponential, RBF). All bring relatively strong coordination between modalities.
- ==Canonical Correlation Analysis (CCA)==: Correlation is a bit weaker.



Aside on Kernel Functions
![[Pasted image 20240614154222.png|300]]
You datapoints might not be separable in your data's original representation space, but might be separable in a higher space.

![[Pasted image 20240614154335.png]]
Correlation-based metrics don't measure similarity on individual embeddings (like cosine or kernel), they measure similarity across an entire populations of vectors.
- Can we learn transformations u and v for our two embedding spaces X and Y so that they're correlated in the same direction?
- A slightly weaker form of similarity because it's measured at the population level, rather than between vectors.

![[Pasted image 20240614154719.png]]
There's a hypothesis that there's some underlying space of all concepts in the world (animals, people, tables, objects), and different views/modalities are simply transformations of this underlying latent space. In particular, this transformation isn't complete; it's in some sense degenerate.
- Underlying concept: Human
	- I can see them (it ignores everything besides what they look like)
	- I can hear them (it ignores everything besides what they sound like)
- We can try to recover this latent space, given the partial views that we observe in the world.

![[Pasted image 20240614154912.png]]
We're essentially here training two autoencoders to learn intermediate representations, which should be similar to an underlying latent representation

## Gated Coordination
- We have two modalities, and want to learn a representation that's coordinated; instead of being a static representation, they can be gated too, where the representation changes for every input.
![[Pasted image 20240614155353.png]]
- Similar to Gated Fusion we learned before, but
	- In Fusion, we were learning gates to figure out how to fuse modalities
	- Here, we're learning gates to create separate representations zA and zB that we can later coordinate using a coordination function.

## Coordination with Contrastive Learning
- Now that we've seen similarity functions, how can we actually train our models?
![[Pasted image 20240614160710.png|300]]
	- The general paradigm of [[Contrastive Loss|Contrastive Learning]] is that we start with some paired data (eg images+text descriptions) that have similarity between them that we want to coordinate.
		- We define positive pairs as captions that describe the image
		- Negative pairs as captions that don't describe the image
	- We want to push positive pairs closer and push negative pairs further apart, in the context of the similarity metrics we compute.

In practice, you can do some pretty cool things!
- If we have a representation space that's coordinated, we'll know that, given an image of a blue car, embed it, then subtract blue and add red, and look at other images in that space, and there will be red cars!
- Airplane - flying + sailing = sailboats
- Cat in bowl - bowl + box = cat in box

[[CLIP]] takes advantage of contrastive learning
- Our goal is to maximize the similarity of positive pairs and maximize the difference between negative pairs.
![[Pasted image 20240614161053.png]]
We can then use CLIP for zero-shot tasks, where we take a set of images, a set of captions we want to classify images into (and compute their embeddings), and see which caption embedding your image embeddings are closest to.

![[Pasted image 20240614161347.png|400]]
We also discard any information not present in images, and any information not present in text, when we do this.

What do we mean when we talk about information that's *shared* between two modalities/data sources/random variables?

Information theory
- "Information value" of a communicated message x depends on how random its content is
	- Low information: 1,1,1,1,1,1,1,1,1,1,1,1,1,
	- High information: 0,1,0,1,0,0,1,1,1,0,0,1
- Formalized:
![[Pasted image 20240614161531.png|300]]

![[Pasted image 20240614161645.png]]

If we use circles to define the entropy of random variables , the non-intersecting circles mean that there's no mutual information between them
![[Pasted image 20240614161744.png]]
But in reality, we know that there are all these difference ways in which they can interact...
So we generally draw the diagrams like this:
![[Pasted image 20240614162025.png]]
That's Conditional [[Entropy]]

[[Mutual Information]] is the overlap of the two modalities;
![[Pasted image 20240614162205.png|300]]
Ratio of two things: Numerator is the joint, denominator is the Px and Py marginal distributions; in other words, how different is my joint distribution when I have both X and Y, rather than the product of the Px and Py marginal distributions as if they were independent.
[[Kullback-Leibler Divergence|KL-Divergence]] is just a way of formalizing how far apart two distributions are.

![[Pasted image 20240614162356.png]]

Here's a simple derivation of the [[InfoNCE]] objective
- f represents all of the trainable parameters... everything from your data to your similarity function.
- We try to maximize the critic function score for positive pairs, and minimize it for negative pairs.
- We can see that when this loss is trained to be optimal, we're essentially training the critic function to be a binary classifier able to distinguish between positive and negative pairs.
![[Pasted image 20240614163032.png]]
Honestly my brain is turning off at this point.
The point is that InfoNCE/Contrastive Loss captures mutual information; Your InfoNCE optimizes a lower bound of mutual information; the better you train it, the more you can approach as much mutual information as there is in your data.
- If you modalities are independent and don't have connections, then maximizing any lower bound on mutual information will approach 0, and contrastive learning will not be useful.


![[Pasted image 20240614163350.png]]
Sometimes there's not enough overlap, and sometimes there's too much overlap. The sweet spot is where you aren't learning too much shared information that's irrelevant for your task (non-Y), but you have enough shared information to do your task (Y).


## Representation Fission
- In general, Fission is a way of reasoning about this assumption of shared information
- In general, we learn different representations that measure different parts of the shared modalities.
- ![[Pasted image 20240614164051.png]]
- If we have two modalities, we have 3 representations -- but we can have more representations than three.

![[Pasted image 20240614164121.png|450]]

Let's talk about Representation Fission via Information Theory:
![[Pasted image 20240614164253.png]]
Emergence: Can we quantify whether information will emerge that wasn't unique or shared to begin with, but emerges from the combination?

We'll see that classical information theory will fail?
![[Pasted image 20240614164539.png]]
We can use Information theory to describe the three-way mutual information between X1,X2, and Task Y. 

But there are some issues! 
- We can't use Mutual Information to quantity information not present in either of our modalities to begin with (emergent information)

Partial Information Decomposition is a sub-area of information science that solves this issue:
- S = Synergy
- R = Redundant
![[Pasted image 20240614164841.png]]

![[Pasted image 20240614165417.png]]

Fine-Grained Fission (not commonly seen)
![[Pasted image 20240614170135.png]]
- How to automatically discover individual clusters in our data?
