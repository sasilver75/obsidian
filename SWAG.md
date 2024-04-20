---
aliases:
  - Situations with Adversarial Generations
---
August 16, 2018 -- [[Allen Institute]], others
Paper: [SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference](https://arxiv.org/abs/1808.05326)

An August 2018 paper debuting an adversarial dataset for language model evaluation, around "grounded commonsense reasoning".
==113k multiple choice questions== about a rich spectrum of grounded situations.

Succeeded by [[HellaSWAG]] after 9 months

Abstract
> ==Given a partial description like "she opened the hood of the car," humans can reason about the situation and anticipate what might come next ("then, she examined the engine").== In this paper, we introduce the task of grounded commonsense inference, unifying natural language inference and commonsense reasoning.  
> We present SWAG, a new dataset with ==113k multiple choice questions about a rich spectrum of grounded situations==. To address the recurring challenges of the annotation artifacts and human biases found in many existing datasets, we propose ==Adversarial Filtering== (AF), a novel procedure that constructs a de-biased dataset by iteratively training an ensemble of stylistic classifiers, and using them to filter the data. To account for the aggressive adversarial filtering, we use state-of-the-art language models to massively oversample a diverse set of potential counterfactuals. Empirical results demonstrate that while humans can solve the resulting inference problems with high accuracy (88%), various competitive models struggle on our task. We provide comprehensive analysis that indicates significant opportunities for future research.

![[Pasted image 20240420144029.png]]