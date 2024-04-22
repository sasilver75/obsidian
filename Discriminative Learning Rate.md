Introduced in the [[ULMFiT]] (2018) paper, along with [[Gradual Unfreezing]].

In the ULMFiT paper, there are three stages:
1. The LM is trained on a general-domain corpus to capture general features of the language (language modeling objective)
2. The LM is fine-tuned on the target domain-data, using discriminative learning rates to learn task-specific features (but still a language modeling objective)
3. The now-classifier is fine-tuned on the target domain-data, using gradual unfreezing.

### Discriminative Learning Rates
- As different layers capture different types of information, they should be fine-tuned to different extents!
- Instead of using the same learning rate for all layers of the model, discriminative fine-tuning allows us to tune *each layer* with different learning rates.