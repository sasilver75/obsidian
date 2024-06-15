April 9, 2020
[[Google Research]]
[BLEURT: Learning Robust Metrics for Text Generation](https://arxiv.org/abs/2004.04696)

Introduced by Google Research in 2020 as an improvement over BLEU. It's built on the popular [[Bidirectional Encoder Representations from Transformers|BERT]] model to offer a more nuanced and human-like assessment of translation accuracy.

The model is trained via two steps:
1. "Pretraining" (🤢), where they generate 6.5M synthetic pairs by randomly perturbing (mask-filling, backtranslation, word dropout) 1.8M sentences from Wikipedia.
2. Finetuning phase, which, in the ==first step==, exposes the model to synthetic translations with errors and variations. The model is then trained to predict a combination of automated metrics ([[BLEU]], [[ROUGE]], [[BERTScore]], Backtranslation likelihood, Entailment, Backtranslation flag) for synthetic pairs. The intuition is that by learning from multiple metrics, BLEURT can capture their strengths while avoiding their weaknesses. This step is costly and typically skipped by loading a checkpoint that has completed it. In the ==second finetuning step==, BLEURT is finetuned on *human ratings of machine translation*. This aligns the model's predictions with human judgements of quality, ***which is the thing we actually care about.***

To use BLEURT, we provide pairs of candidate and reference translations, and the model returns a score from each pair.

It does a lot better in terms of correlation with human judgements than BLEU or ROUGE do.

![[Pasted image 20240604163905.png|300]]