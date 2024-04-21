April 7, 2022 -- [[HuggingFace]], [[Meta AI Research]], Authors include [[Douwe Kiela]]
Paper: [Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162)

A dataset/eval for evaluating the ability of vision-language models to conduct reasoning.
==Given two images and two captions, the goal is to match them correctly -- but crucially, both captions contain the completely identical set of words, only in a diferent order==!
Inspired by [[Winograd]]

Abstract
> We present a ==novel task and dataset for evaluating the ability of vision and language models to conduct visio-linguistic compositional reasoning==, which we call ==Winoground==. ==Given two images and two captions, the goal is to match them correctly== - but crucially, both captions contain a completely identical set of words, only in a different order. The dataset was carefully hand-curated by expert annotators and is labeled with a rich set of fine-grained tags to assist in analyzing model performance. We probe a diverse range of state-of-the-art vision and language models and find that, surprisingly, none of them do much better than chance. Evidently, these models are not as skilled at visio-linguistic compositional reasoning as we might have hoped. We perform an extensive analysis to obtain insights into how future work might try to mitigate these models' shortcomings. We aim for Winoground to serve as a useful evaluation set for advancing the state of the art and driving further progress in the field. The dataset is available at [this https URL](https://huggingface.co/datasets/facebook/winoground).
