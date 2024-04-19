April 18, 2024
Paper: {Coming}

Family: 8B, 70B, 405B parameters
Trained on: 15T tokens
Context window: 8192 tokens (up from 4096 tokens in [[LLaMA 2]]) -- still quite small, but there may be fine-tunes that extend this shortly.

Interestingly, these are *dense* models (ie not [[Mixture-of-Experts]])

The [[Chinchilla]] "compute-optimal" point for an 8B model would be to train it for ~200B tokens, meaning that this training is ~75X beyond that point, an extremely welcome fact (the only time that really matters is inference time!).

Initial reaction by [[Andrej Karpathy]] here: https://twitter.com/karpathy/status/1781028605709234613