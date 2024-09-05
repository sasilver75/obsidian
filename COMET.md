November 2020
Unbabel AI
Paper: [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213/)

An evaluation metric for [[Machine Translation]], similar to [[BLEURT]].
Unlike BLUERT, COMET uses the source sentences in addition to the machine translation and reference translation.

This allows the model to assess the translation quality in the context of the input, rather than just compare the output to a reference.

Under the hood, COMET is based on the XLM-RoBERTa encoder, a multilingual version of the popular [[RoBERTa]] model.

Unlike BLEURT, COMET doesn't require a pre-finetuning phase on synthetic data; instead, the model is directly finetuned on triplets of source, translation, and reference from human-annotated datasets.

Note: ==COMETKiwi== is a ***reference-free*** variant of COMET, which is an ensemble of two models
- One is finetuned on human ratings from WMT
- The other is finetuned on human annotations from the Multilingual Quality Estimation and Post-Editing (MLQE-PE) dataset.
CometKIWI can assess translation quality without needing a reference translation, eliminating the bottleneck of human ratings.

![[Pasted image 20240604163950.png]]