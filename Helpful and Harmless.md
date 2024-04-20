---
aliases:
  - HH
---
April 12, 2022 -- [[Anthropic]] (5 months before ChatGPT)
Paper: [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862)

A dataset released in the paper above

Access two two types of data:
1. Human preference data about HH from the paper linked above. ==This data is meant to train preference/reward models fro subsequent RLHF training== -- NOT meant for supervised training of dialogue agents.
2. Human-generated and annotated red-teaming dialogues from *Red Teaming LMs to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned*. These data are meant to understand how crowdworkers red team models and what types of red team attacks are successful or not.

Abstract
> We apply preference modeling and reinforcement learning from human feedback (RLHF) to finetune language models to act as helpful and harmless assistants. We find this alignment training improves performance on almost all NLP evaluations, and is fully compatible with training for specialized skills such as python coding and summarization. We explore an iterated online mode of training, where preference models and RL policies are updated on a weekly cadence with fresh human feedback data, efficiently improving our datasets and models. Finally, we investigate the robustness of RLHF training, and identify a roughly linear relation between the RL reward and the square root of the KL divergence between the policy and its initialization. Alongside our main results, we perform peripheral analyses on calibration, competing objectives, and the use of OOD detection, compare our models with human writers, and provide samples from our models using prompts appearing in recent related work.