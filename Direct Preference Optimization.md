---
aliases:
  - DPO
---
May 2023
Paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)

See the [[Zephyr]] paper "Zephyr: Direct Distillation of LM Alignment", which was trained with DPO.

Variants: 
- Iterated DPO ((?))
- [[cDPO]]
- [[Kahneman-Tversky Optimization|KTO]]
- [[Identity-Mapping Preference Optimization|IPO]]
- [[Binary Classifier Optimization|BCO]]
- [[Direct Nash Optimization|DNO]]
- [[Stepwise Direct Preference Optimization|sDPO]]


![[Pasted image 20240418171549.png]]
![[Pasted image 20240418171641.png]]


==One downside of these models is that they either sample $y_w$ or $y_l$ from the SFT model or take them directly from existing datasets (thus, sampling from other models), creating a distribution mismatch.==
(y_l and y_w are the winning output and losing output in a preference dataset)