---
aliases:
  - Universal Language Model Fine-Tuning
---
January 18, 2018
Paper: [Universal Language Model Fine-Tuning for Text Classification](https://arxiv.org/abs/1801.06146)

A transfer learning approach with three main stages:
1. General Language Model Pretraining: We pre-train a language model on a large/general corpus of text with a language modeling objective.
2. Target Task Language Model Fine-Tuning: We adapt the pre-trained model to domain-specific language of the target task, still using the language modeling objective of next-token prediction
3. Classifier Fine-Tuning: The model is finetuned for some specific NLP-task in-domain, such as sentiment analysis or question answering. We replace the head of the model with some task-specific layers and use techniques like [[Gradual Unfreezing]] to train these layers.

This set a [[Transfer Learning]] precedent for later models like [[Bidirectional Encoder Representations from Transformers|BERT]] and [[GPT]], which follow a similar approach of pre-training of a large corpus and then finetuning for specific tasks.

Abstract
> Inductive transfer learning has greatly impacted computer vision, but existing approaches in NLP still require task-specific modifications and training from scratch. We propose Universal Language Model Fine-tuning (ULMFiT), an ==effective transfer learning method that can be applied to any task in NLP==, and introduce techniques that are key for fine-tuning a language model. Our method significantly outperforms the state-of-the-art on six text classification tasks, reducing the error by 18-24% on the majority of datasets. Furthermore, with only 100 labeled examples, it matches the performance of training from scratch on 100x more data. We open-source our pretrained models and code.

### Discriminative Learning Rates
- As different layers capture different types of information, they should be fine-tuned to different extents!
- Instead of using the same learning rate for all layers of the model, discriminative fine-tuning allows us to tune *each layer* with different learning rates.
### Gradual Unfreezing
- Rather than fine-tuning all layers at once, which risks catastrophic forgetting, we propose to gradually unfreeze the model, starting from the last layer, as this contains the least *general* knowledge.
- We first unfreeze the last layer and fine-tune all unfrozen layers for for one epoch.
- We then unfreeze the next lower frozen layers and repeat, until we finetune all layers until convergence at the last iteration.



![[Pasted image 20240124181308.png]]