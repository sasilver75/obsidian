---
aliases:
  - Multimodal Visual Patterns
---
Benchmark for vision models

![[Pasted image 20260503201912.png]]
- Asking incredibly obvious questions
	- Model gets it wrong, hallucinates details to support its claim
- Evidence that LLMs can't really se..

Due to a lack of visual features that they can perceive with.
![[Pasted image 20260503201953.png]]
This dataset created by finding pairs of images that are close in [[CLIP]] space but far in [[DINOv2]] (vision model) space. 

This shows ... that CLIP isn't discriminative enough to tell these two apart. This points to a failure in vision-language pretraining.

In CLIP pretratining, you come up with a bunch of captions and images, scramble them, and ask the model to pair the image with the caption, basically.
- But what is a caption that would distinguish the two images bove? Details that aren't included in the caption. ==If your loss function can't tell these image apart, why would the model be able to?==