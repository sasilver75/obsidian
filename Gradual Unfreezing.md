As described in the [[ULMFiT]] paper (2018), along with [[Discriminative Learning Rate]]s.

In the ULMFiT paper, there are three stages:
1. The LM is trained on a general-domain corpus to capture general features of the language (language modeling objective)
2. The LM is fine-tuned on the target domain-data, using discriminative learning rates to learn task-specific features (but still a language modeling objective)
3. The now-classifier is fine-tuned on the target domain-data, using gradual unfreezing.

### Gradual Unfreezing
- Rather than fine-tuning all layers at once, which risks catastrophic forgetting, we propose to gradually unfreeze the model, starting from the last layer, as this contains the least *general* knowledge.
- We first unfreeze the last layer and fine-tune all unfrozen layers for for one epoch.
- We then unfreeze the next lower frozen layers and repeat, until we finetune all layers until convergence at the last iteration.
