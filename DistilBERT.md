October 2, 2019
Paper: [DistilBERT, a distilled version of BERT: smaller, faster, cheaper, and lighter](https://arxiv.org/ abs/1910.01108)("Distilled" BERT)
Compare: [[BERT|BERT]]

Abstract
> As Transfer Learning from large-scale pre-trained models becomes more prevalent in Natural Language Processing (NLP), operating these large models in on-the-edge and/or under constrained computational training or inference budgets remains challenging. In this work, ==we propose a method to pre-train a smaller== general-purpose language representation ==model==, called ==DistilBERT==, which can then be fine-tuned with good performances on a wide range of tasks like its larger counterparts. While most prior work investigated the use of distillation for building task-specific models, ==we leverage knowledge distillation during the pre-training phase and show that it is possible to reduce the size of a BERT model by 40%,== while retaining 97% of its language understanding capabilities and being 60% faster. To leverage the inductive biases learned by larger models during pre-training, we introduce a triple loss combining language modeling, distillation and cosine-distance losses. Our smaller, faster and lighter model is cheaper to pre-train and we demonstrate its capabilities for on-device computations in a proof-of-concept experiment and a comparative on-device study.


# Paper Figures

# Non-Paper Figures
![[Pasted image 20240617110639.png]]


# Non-Paper Figures
![[Pasted image 20240619172757.png]]![[Pasted image 20240619174305.png]]