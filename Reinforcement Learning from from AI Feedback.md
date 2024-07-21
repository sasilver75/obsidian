---
aliases:
  - RLAIF
---
Uses foundation models acting as "AI critics"

It's just the usual RL process (eg with [[Proximal Policy Optimization|PPO]]), but the reward model was trained using synthetic preference annotation, rather than human preference annotation. The best reward models are shown to have high correlation with human ratings, and can more cheaply scale to a larger number of annotations.