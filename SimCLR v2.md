June 17, 2020 (4 months after [[SimCLR]]) -- [[Google Research]] 
Paper: [Big Self-Supervised Models are Strong Semi-Supervised Learners](https://arxiv.org/abs/2006.10029)

Like its predecessor [[SimCLR]], it uses a contrastive loss function to bind image and language modalities.

Using a big network to do self-supervised pre-training and fine-tuning, and then distilling it into a smaller one with little loss in classification accuracy. 
The proposed [[Semi-Supervised Learning]] algorithm is:
1. Unsupervised pretraining of a big [[Residual Network|ResNet]] model using [[SimCLR v2]]
2. Supervised fine-tuning in a few labeled examples
3. Distillation with unlabeled examples for refining and transferring the task-specific knowledge.

Abstract
> One paradigm for learning from few labeled examples while making best use of a large amount of unlabeled data is ==unsupervised pretraining followed by supervised fine-tuning==. Although this paradigm uses unlabeled data in a task-agnostic way, in contrast to common approaches to semi-supervised learning for computer vision, we show that it is ==surprisingly effective for semi-supervised learning on ImageNet==. A key ingredient of our approach is the use of big (deep and wide) networks during pretraining and fine-tuning. We find that, the fewer the labels, the more this approach (task-agnostic use of unlabeled data) benefits from a bigger network. After fine-tuning, the big network can be further improved and distilled into a much smaller one with little loss in classification accuracy by using the unlabeled examples for a second time, but in a task-specific way. The proposed semi-supervised learning algorithm can be summarized in ==three steps==: ==unsupervised pretraining== of a big ResNet model using ==SimCLRv2==, ==supervised fine-tuning== on a few labeled examples, and ==distillation with unlabeled examples== for refining and transferring the task-specific knowledge. This procedure achieves 73.9% ImageNet top-1 accuracy with just 1% of the labels

![[Pasted image 20240420175503.png]]

