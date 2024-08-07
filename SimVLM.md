---
aliases:
  - Simple Visual Language Model Pretraining with Weak Supervision
---


August 24, 2021 (18 months after [[SimCLR]], if they're related)
Paper: [SimVLM: Simple Visual Language Model Pretraining with Weak Supervision](https://arxiv.org/abs/2108.10904)

Abstract
> With recent progress in joint modeling of visual and textual representations, Vision-Language Pretraining (VLP) has achieved impressive performance on many multimodal downstream tasks. However, the ==requirement for expensive annotations including clean image captions and regional labels limits the scalability of existing approaches, and complicates the pretraining procedure== with the introduction of multiple dataset-specific objectives. In this work, we relax these constraints and present a minimalist pretraining framework, named ==Simple Visual Language Model (SimVLM).== Unlike prior work, SimVLM reduces the training complexity by ==exploiting large-scale weak supervision==, and is ==trained end-to-end with a single prefix language modeling objective==. Without utilizing extra data or task-specific customization, the resulting model significantly outperforms previous pretraining methods and achieves new state-of-the-art results on a wide range of discriminative and generative vision-language benchmarks, including VQA (+3.74% vqa-score), NLVR2 (+1.17% accuracy), SNLI-VE (+1.37% accuracy) and image captioning tasks (+10.1% average CIDEr score). Furthermore, we demonstrate that SimVLM acquires strong generalization and transfer ability, enabling zero-shot behavior including open-ended visual question answering and cross-modality transfer.



![[Pasted image 20240417140211.png]]