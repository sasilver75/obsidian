December 8, 2021 -- [[DeepMind]]
Paper: [Scaling Language Models: Methods, Analysis, and Insights from Training Gopher](https://arxiv.org/abs/2112.11446)

This is a paper about the training insights from training the [[Gopher]] model, a 280B model from Deepmind.
The dataset is detailed within this paper, but ==for some reason the exact number of tokens is not explicitly defined -- it also wasn't made openly available for download.==

Existing text datasets at the time were typically based solely on web pages (eg [[C4]]). This work, like [[The Pile]], includes data from many sources, like web pages, books, and academic papers.

They decided only to use simple heuristics for filtering out low-quality tokens, rather than training classifiers based on "gold" sets of text. They say that filtering for quality (and preserving diversity/avoiding biases) is an important direction for future research.

Abstract
> Language modeling provides a step towards intelligent communication systems by harnessing large repositories of written human knowledge to better predict and understand the world. In this paper, we present an analysis of Transformer-based language model performance across a wide range of model scales -- ==from models with tens of millions of parameters up to a 280 billion parameter model== called Gopher. These models are ==evaluated on 152 diverse tasks==, achieving state-of-the-art performance across the majority. Gains from scale are largest in areas such as reading comprehension, fact-checking, and the identification of toxic language, but logical and mathematical reasoning see less benefit. We ==provide a holistic analysis of the training dataset ((MassiveText)) and model's behaviour==, covering the intersection of model scale with bias and toxicity. Finally we discuss the application of language models to AI safety and the mitigation of downstream harms.

![[Pasted image 20240419234617.png]]
A pretty normal filtering pipeline; You can find details on each step in the paper!

