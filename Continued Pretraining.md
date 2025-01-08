---
aliases:
  - Continuous Pretraining
---


Taking an existing pretrained model and pretraining it further on new domain data -- like taking a LLaMA 3 8B base model that's been trained on a general text corpus that we then want to adapt to finance/medical/legal/other domains. So it's more self-supervised learning on a different domain, but not IFT.

Note that the [[LLaMA 3.1]] paper refers to Continuous Pretraining, saying it pretrains the 405B model on 15.6T tokens using a context window of 8K tokens, and then follows that with a "continued pretraining stage" that increases the supported context window to 128K tokens.
- So in this case, it's not about a different domain, but it's about a difference in context window length.


![[Pasted image 20241215145641.png]]
From: https://youtu.be/iN2A1uxRaco?si=LqETcSnETTe7erPg&t=2116
These are quite "data-hungry" methods -- you often need billions of high-quality tokens to adapt a language model to a new domain, for instance. (Above: [[Minerva]], [[DeepSeekMath]])

