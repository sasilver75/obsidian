---
aliases:
  - DPO
---
May 2023
Paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
Resources:
- [AIF and DPO: Distilling Zephyr and Friends](https://youtu.be/cuObPxCOBCw?si=JSgXQGcareJU2mJd)


Attempts to avoid much of the complexity of [[Proximal Policy Optimization|PPO]] and [[Reinforcement Learning from Human Feedback|RLHF]].

See the [[Zephyr]] paper "Zephyr: Direct Distillation of LM Alignment", which was trained with DPO.

==One downside of these models is that they either sample $y_w$ or $y_l$ from the SFT model or take them directly from existing datasets (thus, sampling from other models), creating a distribution mismatch.==
(y_l and y_w are the winning output and losing output in a preference dataset)

Variants: 
- Iterated DPO ((?))
- [[cDPO]]
- [[Kahneman-Tversky Optimization|KTO]] (Only requires a binary label, rather than a *pair* of accepted/rejected generations)
- [[Identity-Mapping Preference Optimization|IPO]] (DPO easily overfits; IPO is more robust to overfitting))
- [[Binary Classifier Optimization|BCO]]
- [[Direct Nash Optimization|DNO]]
- [[Stepwise Direct Preference Optimization|sDPO]]


# Paper Figures

# Non-Paper Figures

![[Pasted image 20240424155551.png]]
- We're optimizing over our policy $\pi$ , the new LM we're trying to learn
- The optimization samples a prompt $x$, and two responses: a winning/losing response
- The first term in the parens is applied to the good response, $y_w$ . The term consists of the log ratio of the new model, divided by the base model.
- The second term in the parens is appleid to the losing response, $y_l$ 
- The full objective tells us to maximize the difference between these two terms; upweighting the good examples, and downweighting the back examples.
- (The additional coefficient $\beta$  represents how close we want our new model to stay to the base model)

Optimization Process:
1. Sample a good/bad pair
2. Run the base model on these two examples, saving the score that it gave them.
3. Run the new model on these two examples
4. Backprop through the loss function

Doesn't require any of the sampling or the tricks that you're used to in RL; can really just implement in PyTorch and backprop through the model directly.


![[Pasted image 20240418171549.png]]
![[Pasted image 20240418171641.png]]


![[Pasted image 20240627140357.png]]
Note that it's  now common for DPO to be performed prior to [[Proximal Policy Optimization|PPO]] ([link](https://www.interconnects.ai/p/rlhf-roundup-2024))