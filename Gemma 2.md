June 27, 2024
[[DeepMind]], Gemma Team
[Gemma 2: Improving Open Language Models at a Practical Size](https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf)

Note: Gemma 2's Terms of Use has unrestricted us for the output, meaning models trained on Gemma-2 output can be used for anything. This makes it a interesting choice for (eg) synthetic data generation.

Abstract
> In this work, we introduce Gemma 2, a new addition to the Gemma family of lightweight, state-of-the-art open models, ranging in scale from 2 billion to 27 billion parameters. The ==9 billion== and ==27 billion parameter== models are available today, with a ==2 billion parameter model== to be released shortly. In this new version, we provide several technical modifications to our architecture, such as ==interleaving local-global attentions== (Beltagy et al., 2020a) and [[Grouped Query Attention]] (Ainslie et al., 2023). We also train the 2B and 9B models with knowledge distillation (Hinton et al., 2015) instead of next token prediction. The resulting models deliver the best performance for their size, and even offer competitive alternatives to models that are 2-3× bigger. We release all our models to the community.