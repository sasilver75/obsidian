---
aliases:
  - RLHF
---
March 4, 2022 (6 months before ChatGPT)
[[OpenAI]] - Authors include [[John Schulman]] and [[Paul Christiano]], among others
Paper: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
#zotero 
Significance: This is the paper promoted (but did not introduce) the use of [[Reinforcement Learning from Human Feedback|RLHF]] (created in 2017) *in language models*, and produced and [[InstructGPT]], a model instruction-tuned and human-preference tuned using Reinforcement Learning from Human Feedback.

See also: [[InstructGPT]]

Note: RLHF as a concept was introduced in this *Christiano, 2017* paper *[Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741)*, which I *believe* was about use in Robotics.

----

Takeaways:
- 


Abstract
> Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, ==we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback==. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we ==collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning==. We then ==collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback==. We call the resulting models ==InstructGPT==. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.


# Paper Figures

![[Pasted image 20240415231812.png]]

![[Pasted image 20240427145737.png]]

![[Pasted image 20240427150536.png]]
![[Pasted image 20240427150634.png]]
# Additional Pictures

![[Pasted image 20240418171453.png]]


![[Pasted image 20240426021803.png]]
From LLAMA-2 paper: An awesome picture from the [[LLaMA 2]] technical report showing how the RLHF process shifts the generated texts towards higher rewards. The X axis is "Reward Model Score".

![[Pasted image 20240426132052.png]]
From LLaMA-2 paper: Showing the improvement on Helpfulness and Harmlessness benchmarks from both human and LM judges as we do multiple rounds of SFT and then RLHFing

