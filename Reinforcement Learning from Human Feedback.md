---
aliases:
  - RLHF
---
March 4, 2022 -- [[OpenAI]]
Paper: [Training Language Models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)

The paper that introduced [[Reinforcement Learning from Human Feedback|RLHF]] via training [[InstructGPT]].
As a result of the human-preference tuning, 1.3B InstructGPT was able to beat the 175B GPT-3 in human preference evaluations, despite being over 100x smaller.

See also: [[Reinforcement Learning from Human Feedback with AI Feedback]] (RLAIF)

Abstract
> Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, we show an avenue for aligning language models with user intent on a wide range of tasks by ==fine-tuning with human feedback==. Starting with a set of ==labeler-written prompts and prompts submitted through the OpenAI API==, we ==collect a dataset of labeler demonstrations of the desired model behavior==, which we use to ==fine-tune GPT-3 using supervised learning==. We then ==collect a dataset of rankings of model outputs==, which we use to ==further fine-tune this supervised model using reinforcement learning from human feedback==. We call the resulting models ==InstructGPT==. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.


![[Pasted image 20240415231812.png]]

![[Pasted image 20240418171453.png]]


![[Pasted image 20240426021803.png]]
An awesome picture from the [[LLaMA 2]] technical report showing how the RLHF process shifts the generated texts towards higher rewards. The X axis is "Reward Model Score".

![[Pasted image 20240426132052.png]]
Showing the improvement on Helpfulness and Harmlessness benchmarks from both human and LM judges as we do multiple rounds of SFT and then RLHFing