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


