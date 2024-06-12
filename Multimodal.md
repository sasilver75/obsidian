---
aliases:
  - Multimodality
---

![[Pasted image 20240608095521.png]]
Above: From CMU Multimodal Machine Learning

What is a modality?
- Refers to the way in which something is *expressed* or *perceived*
- Some modalities are more "raw" (closer to the sensor), and some are more abstract and processed (further from the sensor)
![[Pasted image 20240612125648.png]]
Above: It's important that the data is heterogenous, but what's really interesting is that the data is interconnected! That a duck's quacking sound is in some way related to the image of a duck!

![[Pasted image 20240608100225.png]]
The closer you are to the raw modalities, the more relevant to this class you are. The more abstracted, the more similar your modalities will look (maybe in their representations, eg being simple 0s and 1s). In the raw modalities, you really need to think about the differences related to the sources of data.
- A modality refers to the way in which something is ==expressed== or ==perceived.==
- Note that language is a processed modality, not a raw modality! It's a little more abstract than a raw modality.

The simple definition for Multimodal is "==having multiple modalities==," but we prefer: ==the scientific study of heterogenous (they're different) and interconnected (so that bringing them together brings an advantage) data.==


![[Pasted image 20240612125846.png]]
- Some will be very homogenous, and others will by very heterogenous. Abstract modalities are often more homogenous, and raw modalities are often more heterogenous from eachother.

Modalities will have:
1. Element representations: Discrete, continuous, granularity
	- In language, the tokenization is a lot clearer, but in an image with bounding boxes, we might say "A `teacup` is on the `right` of a `laptop` in a `clean room`."
	- Language is a modality that's been created, so here the tokenization is a lot clearer than in a raw modality.
2. Element distributions: Density, frequency. An image may have many many objects (or, if our representation is a pixel, we have many!). If we use character-level embedding instead of word-level embedding, we'll have many more.
3. Structure: Temporal, spatial, latent, explicit. Objects have relationships between them, as do pixels. 
4. Information: The amount of information of an individual element; a pixel gives you some information, and an object/word quite a bit more, and phrases even more.
5. Noise: How much uncertainty, or missing data is there? Language is "easier" as a modality, when you think about noise! Speech, electric brain scans, and vision has much more noise.
6. Relevance: How relevant is the data in your modality to your task that you're actually performing!

![[Pasted image 20240612131154.png]]


![[Pasted image 20240609094927.png]]
Above: How do many people in AI represent images these days (more than a year ago)? Pixel-level representations, or a vector-level representation (eg from an auto-encoder. In the middle are feature-based approaches like edges, etc (old).

These days, it might just be a list of objects (above), after we run some object recognition on it.
- But you still need to choose: "What is an element? What is our unit of analysis? Our meaningful atom, where there's meaning, but enough difference between atoms."


==Connection==: Shared information that relates modalities. Vision and language seem to be very related, with lots of mutual information. It may or may not be the case that CLIP-like methods will work for other modalities with less shared information.

![[Pasted image 20240612132020.png]]
Above: 
- You can connect modalities in two ways:
	- Bottom Up (Statistical)
	- Top Down (Semantic)
==Association== ("Co-occurrence"/"Correlation"): If every time you say "woof", we point at a table, we eventually do an association with these objects. We assume that if things co-occur often, they share some meaning.
==Correspondence==: Driven by human knowledge. We humans decide that there's a correspondence -- not just driven by co-occurrence. Maybe two things co-occur very rarely, but we humans decide that they are highly related! Maybe there's a temporal contingency (some event comes after/before the other).


==Interaction:==
When you have multiple modalities that are going to interact, you have to at some point ==Interact== them together! We have to learn some model that can learn some representation of the fusion of modalities.
![[Pasted image 20240612132624.png]]
- Here, the modalities come together and create something new that (ideally) neither modality had on its own. If you have two modalities whose meanings are *completely shared*, you won't learn much from the fusion of the modalities; there is very little interaction.
	- In contrast, there can be modalities that are diverse and seem unrelated that interact in a rich way.

![[Pasted image 20240612133558.png]]

What are the core technical challenges when talking about multimodal machine learning, in comparison with conventional machine learning?

There are six main challenges:
1. Representation
2. Alignment
3. Reasoning
4. Generation
5. Transference
6. Quantification

...

1. ==Representation==
	- Definition: Learning representations that reflect cross-modal interactions between individual elements, across different modalities.
	- A core building block for most multimodal modeling problems.
	- Sub-challenges
		- ==Fusion==: Where the number of input modalities is larger than the number of output representations. Classically, jointly fusing two representations into a single representation. The most popular, up to a few years ago.
		- ==Coordination==: Where the number of input modalities and output representations are the same.
		- ==Fission==: Where the number of output representations is greater than the number of input modalities. Could also be called factorization; used often when you're trying to do understanding of the data. You might create one unique to modality A, one unique to modality B, and one shared between both modalities.
	- ![[Pasted image 20240612135341.png|300]]
2. ==Alignment==
	- Definition: Identifying and modeling cross-modal connections between all elements of multiple modalities, building from a data structure.
	- Modalities have structure! Pixels in an image are structurally related to eachother, there might be temporal structure, etc.
	- We want to learn the relationship between the elements (between words, between objects), and learn a representation that takes into consideration this alignment/grounding.
	- Sub-challenges
		- Discrete grounding: My image has already been discretized into a list of objects. I already have discrete tokens in one modality, and a discrete list of objects in another. How to learn an alignment between these?
		- (His slide was messed up, here)
3.  ==Reasoning==
	- Definition: Combining knowledge, usually through multiple inferential steps, exploiting multimodal alignment and problem structure, ideally in a human-interpretable way. Might also include external knowledge.![[Pasted image 20240612140415.png|300]]
	- Sub-Challenges
		- Structure Modeling
		- Intermediate concepts
		- Inference paradigm
		- External knowledge
		- ![[Pasted image 20240612140502.png|200]]
4. ==Generation==
	- Definition: Learning a generative process to *produce* raw modalities that reflect cross-modal interactions, structure, and coherence. ![[Pasted image 20240612140540.png|400]]
	- We could be generating a modality because we're creating, summarizing, translating, etc. How much of our source information do we want to keep in our generated modality?
		- In traditional MT, we want to keep all of it
		- In summarization, we definitionally don't want to!
5. ==Transference== ("Modulation")
	- Definition: Transfer knowledge between modalities, usually to help the target modality, which may be noisy or with limited resources.
	- Often LLMs these days are helping us out with vision problems.
	- ![[Pasted image 20240612140905.png|300]]![[Pasted image 20240612140800.png|300]]
	- Co-Learning: At test time, you'll have some certain modality. Transfer will come from some completely different modality. You can two it in two ways, via representation or generation. 
		- Co-learning via generation: By generating the other modality, you're learning extra information for that first modality! Let's say we want to do language representation, but we also have the nonverbal... We could say: "Hey, I'm going to learn my langauge representation in such a way that I'm also able to predict the nonverbal expression that happens at the same time. But at test time I'll only have language. I'll just do the vision part during training to help with transference."
6. ==Quantification==
	- Definition: Empirical and theoretical study to better understand heterogeneity, cross-modal interactions, and multimodal learning process.
	- Subchallenges:
		- Heterogeneity
		- Interactions
		- Learning

![[Pasted image 20240612141205.png]]





