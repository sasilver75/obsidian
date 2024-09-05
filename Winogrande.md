July 24, 2019 (The previous Winograd Schema Challenge was in 2011)
Paper: [WinoGrande: An Adversarial Winograd Schema Challenge at Scale](https://arxiv.org/abs/1907.10641)

The previous Winograd Schema Challenge saturated; This is a much larger dataset of harder problems in a similar vein.
See also: [[Winograd]]

Abstract
> The ==Winograd Schema Challenge (WSC==) (Levesque, Davis, and Morgenstern 2011), a benchmark for commonsense reasoning, is a set of 273 expert-crafted pronoun resolution problems originally designed to be unsolvable for statistical models that rely on selectional preferences or word associations. However, ==recent advances in neural language models have already reached around 90% accuracy on variants of WSC==. This raises an important question whether these models have truly acquired robust commonsense capabilities or whether they rely on spurious biases in the datasets that lead to an overestimation of the true capabilities of machine commonsense. To investigate this question, ==we introduce WinoGrande==, a large-scale dataset of ==44k problems==, ==inspired by the original WSC design==, but adjusted to improve both the scale and the hardness of the dataset. The key steps of the dataset construction consist of (1) a carefully designed crowdsourcing procedure, followed by (2) systematic bias reduction using a novel AfLite algorithm that generalizes human-detectable word associations to machine-detectable embedding associations. The best state-of-the-art methods on WinoGrande achieve 59.4-79.1%, which are 15-35% below human performance of 94.0%, depending on the amount of the training data allowed. Furthermore, we establish new state-of-the-art results on five related benchmarks - WSC (90.1%), DPR (93.1%), COPA (90.6%), KnowRef (85.6%), and Winogender (97.1%). These results have dual implications: on one hand, they demonstrate the effectiveness of WinoGrande when used as a resource for transfer learning. On the other hand, they raise a concern that we are likely to be overestimating the true capabilities of machine commonsense across all these benchmarks. We emphasize the importance of algorithmic bias reduction in existing and future benchmarks to mitigate such overestimation.

Example of Winograd Schema:

Example:
> The city councilmen refused the demonstrators a permit because they `feared/advocated` violence.

If the word `feared` is used, then `they` refers to the city council. If `advocated` is used, it refers to the `demonstrators`.