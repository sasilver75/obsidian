---
aliases:
  - Multimodal C4
---

October 28, 2023 -- Multiple collaborators, including universities, [[LAION]], and [[Allen Institute|AI2]]
Paper: [Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved with Text](https://arxiv.org/pdf/2304.06939.pdf)


Abstract
> In-context vision and language models like [[Flamingo]] [2] support arbitrarily interleaved sequences of images and text as input. This format not only enables [[Few-Shot Learning]] via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., “What do image A and image B have in common?” To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text. To date, however, large-scale data of this form have not been publicly available. We release ==Multimodal C4 (mmc4),== an ==augmentation of the popular text-only c4 corpus with images interleaved==. We ==use a linear assignment algorithm to place images into longer bodies of text using CLIP features== [24], a process that we show outperforms alternatives. mmc4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (88%) of images are topically relevant, and that linear assignment frequently selects individual sentences specifically well-aligned with each image (80%). After filtering NSFW images, ads, etc., the resulting mmc4 corpus consists of 101.2M documents with 571M images interleaved in 43B English tokens.


