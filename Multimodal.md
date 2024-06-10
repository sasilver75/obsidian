---
aliases:
  - Multimodality
---

![[Pasted image 20240608095521.png]]
Above: From CMU Multimodal Machine Learning

What is a modality?
- Refers to the way in which something is *expressed* or *perceived*
- Some modalities are more "raw" (closer to the sensor), and some are more abstract and processed (further from the sensor)

![[Pasted image 20240608100225.png]]
The closer you are to the raw modalities, the more relevant to this class you are. The more abstracted, the more similar your modalities will look (maybe in their representations, eg being simple 0s and 1s). In the raw modalities, you really need to think about the differences related to the sources of data.

The simple definition for Multimodal is "==having multiple modalities==," but we prefer: ==the scientific study of heterogenous (they're different) and interconnected (so that bringing them together brings an advantage) data.==

Modalities will have:
1. Element representations: Discrete, continuous, granularity
	- In language, the tokenization is a lot clearer, but in an image with bounding boxes, we might say "A `teacup` is on the `right` of a `laptop` in a `clean room`."
2. Element distribtuions: Density, ffrequency. An image may have many many objects (or, if our represntation is a pixel, we have many!). If we use character-level embedding instead of word-level embedding, we'll have many more.
3. Structure: I

![[Pasted image 20240609094927.png]]
