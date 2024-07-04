---
aliases:
  - DPO
---
May 29, 2023
Stanford (incl [[Christopher Manning]])
Paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)Takeaway: A method for human-preference finetuning that avoids much of the complexity of [[Proximal Policy Optimization|PPO]] and [[Reinforcement Learning from Human Feedback|RLHF]], where we have to train both a reward model and our language model.

Resources:
- Video: [CornellTech Talk:AIF and DPO: Distilling Zephyr and Friends](https://youtu.be/cuObPxCOBCw?si=JSgXQGcareJU2mJd)
- Video: [Luis Serrano: DPO - How to finetune LLMs directly without RL](https://www.youtube.com/watch?v=k2pD3k1485A)
- Useful example of use: [[Zephyr]]

Variants: 
- Iterated DPO, [[cDPO]], [[Kahneman-Tversky Optimization|KTO]] (Only requires a binary label, rather than a *pair* of accepted/rejected generations), [[Identity-Mapping Preference Optimization|IPO]] (DPO easily overfits; IPO is more robust to overfitting)), [[Binary Classifier Optimization|BCO]], [[Direct Nash Optimization|DNO]], [[Stepwise Direct Preference Optimization|sDPO]]

----

Re RLHF: Training reward model is a drag, and it's expensive once you've trained it, because you have to run both the reward model and the main model together. Can we somehow do away with the reward model process?

"It doesn't actually totally do away with the reward model from the RLHF objective -- instead, it *embeds* it in the language model's loss function."
![[Pasted image 20240703112224.png]]
- Notice this RLHF objective has the reward from the reward model $r_{\phi}(x,y)$ , as well as the policy from the language model $\pi_{\theta}(y|x)$, which we we can think of as the probability of the next word.
![[Pasted image 20240703113451.png]]
- This first main component tries to maximize the expected reward for model generations on prompts in our dataset..
- The second part prevents the model from changing too drastically from one iteration of DPO to the next, using the [[Kullback-Leibler Divergence|KL-Divergence]] between the policy and the reference policy (meaning the language model "after" and "before"). The Beta is just a hyperparameter that we can tune, telling us how much we want to punish divergence.
==We want to kind of "get rid" of this *reward* in the first term...we want to turn it into a probability so that the loss function only has probabilities, and not rewards.==
- We do this using the [[Bradley-Terry Model]], which turns rewards into probabilities.
![[Pasted image 20240703114015.png|450]]
How do we turn these scores into probabilities, though?
- We want $p(time)$ to be higher than $p(banana)$... so why don't we try making them proportional to 5 and 2... but we need our probabilities to add to one, so we normalize by the sum.
![[Pasted image 20240703114141.png]]
Does this work? No... we need something more, because this fails in some cases -- like what if our reward for $p(banana)$ were *negative* 2? We can't have negative probabilities, so this model doesn't work.
How can we turn numbers into always-positive numbers? Often in ML, we *exponentiate* the number. Let's try that, keeping the condition that these probabilities have to add to one:
![[Pasted image 20240703115154.png]]
See that we can rewrite this as [[Sigmoid Activation Function|Sigmoid]] functions!
- Math: For the left side, divide both the numerator and denominator by $e^5$ to get $1/(1+(e^2/e^5))$ . Dividing exponents with the same base means you can just subtract the exponents, so we can rewrite that as $1/(1+e^{(2-5)})$  or $1/(1+e^{-3})$ . Given that the sigmoid function is $\sigma(x) = 1/(1+e^{-x})$, we can rewrite ours as $\sigma(3)$, which is equivalent to their $\sigma(5-2)$ shown above.
![[Pasted image 20240703115644.png]]
If we plug these in to our sigmoid function, we see that the probabilities are positive and sum to one, which is what we want. Generally, we can turn rewards into probabilities using:
![[Pasted image 20240703115819.png]]
Of course, there are other ways of turning rewards into probabilities, but this way works pretty well, so we stick with it.
![[Pasted image 20240703123604.png]]
Skipping the mathematical manipulation from the paper, out plops the loss function that we want! Note that it doesn't have the letter $r$ in it -- there's no reward directly, so we don't have to train a reward model (the reward is still there, implicitly, though).
![[Pasted image 20240703112020.png]]
- $\pi_{\theta}$ is the probability of a response ($y_w$ if a winning response, $y_l$ if a losing response), given a prompt input $x$. This comes from our language model.
- ==See that we want to maximize the left term, and minimize the right term.== 
- We have an expected value, because we're averaging over generated responses.
- Why do we have a logarithm? When we see sums of logarithms, we like to see it as the logarithm of a product... where the numbers being multiplied are often between zero and one. 
![[Pasted image 20240703123854.png]]




==One downside of these models is that they either sample $y_w$ or $y_l$ from the SFT model or take them directly from existing datasets (thus, sampling from other models), creating a distribution mismatch.==
(y_l and y_w are the winning output and losing output in a preference dataset)




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
Note that it's  now common for DPO to be performed prior to [[Proximal Policy Optimization|PPO]] ([link](https://www.interconnects.ai/p/rlhf-roundup-2024)), which was done in LLaMA 3.