March 4, 2022 (6 months before ChatGPT)
[[OpenAI]] - Authors include [[John Schulman]] and [[Paul Christiano]], among others
Paper: [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
#zotero 
Significance: This is the paper promoted (but did not introduce) the use of [[Reinforcement Learning from Human Feedback|RLHF]] (Christiano, 2017, use in Robotics) *in language models*, and produced and [[InstructGPT]], a model instruction-tuned and human-preference tuned using Reinforcement Learning from Human Feedback.

See also: [[Reinforcement Learning from Human Feedback|RLHF]]

----

Takeaways:
- InstructGPT (1.3B) produces responses that are preferred to the outputs of the 175B GPT-3 model (but still makes simple mistakes).
- Notes the misalignment with the language modeling objective of "predict the next token" with what we *really want* from our language models, which is "follow the user's instructions helpfully and safely." 
	- We want our LMs to be helpful (help the user solve their task), honest (not fabricate information or mislead the user), and harmless (should not cause physical, psychological, or social harm to people or the environment).
- [[Supervised Fine-Tuning]] (SFT)
	- We collect a dataset of human-written demonstrations of desired behavior on prompts, and use this to further-train supervised learning baselines.
	- Re: creation of the SFT and RM-training datasets, labelers were asked to write three kinds of prompts: Plain, Few-Shot, and "User-Based" (using some of the use-cases stated in waitlist applications to the then-closed OpenAI API)
- [[Reinforcement Learning from Human Feedback]] (RLHF)
	- We collect a dataset of human-labeled comparisons between multiple outputs from our models on a larger set of prompts. To scale this effort, we then train a reward model (RM) on this dataset to predict which model output our labelers would prefer. Finally, we use this RM as a reward function, additionally fine-tuning our supervised-learning baseline to maximize this reward using [[Proximal Policy Optimization|PPO]].
- Introduces (?) the idea of an "==alignment tax==," where the alignment procedure comes at the cost of lower performance on certain tasks that we may care about -- they noticed lower performance on (among others) [[SQuAD]], [[HellaSWAG]], and a few others.
- For the Reward Model, they use a cross-entropy loss, with the comparisons as labels; the difference in rewards represents the log odds that one response will be preferred to the other by a human labeler.
	- They use a ==Likert scale; 1-7== for the overall quality rating of responses.
- They note that there are many stakeholders of these models, and it is impossible to train a system that's aligned to everyone's preferences at once, or where everyone would endorse the tradeoffs.

Specs:
- InstructGPT is a 1.3B parameter model 
- RM is a 6B parameter model


Abstract
> Making language models bigger does not inherently make them better at following a user's intent. For example, large language models can generate outputs that are untruthful, toxic, or simply not helpful to the user. In other words, these models are not aligned with their users. In this paper, ==we show an avenue for aligning language models with user intent on a wide range of tasks by fine-tuning with human feedback==. Starting with a set of labeler-written prompts and prompts submitted through the OpenAI API, we ==collect a dataset of labeler demonstrations of the desired model behavior, which we use to fine-tune GPT-3 using supervised learning==. We then ==collect a dataset of rankings of model outputs, which we use to further fine-tune this supervised model using reinforcement learning from human feedback==. We call the resulting models ==InstructGPT==. In human evaluations on our prompt distribution, outputs from the 1.3B parameter InstructGPT model are preferred to outputs from the 175B GPT-3, despite having 100x fewer parameters. Moreover, InstructGPT models show improvements in truthfulness and reductions in toxic output generation while having minimal performance regressions on public NLP datasets. Even though InstructGPT still makes simple mistakes, our results show that fine-tuning with human feedback is a promising direction for aligning language models with human intent.

# Paper Figures

![[Pasted image 20240427143830.png|400]]

![[Pasted image 20240427144621.png|400]]

![[Pasted image 20240427145737.png]]

![[Pasted image 20240427150527.png]]
Reward Model (RM) training objective
![[Pasted image 20240427150630.png]]

![[Pasted image 20240427151424.png]]
Above: They also used PPO to fine-tuned a variant of the GPT-3.5 model; part of OpenAI's efforts to refine the capabilities of LLMs in following instructions more effectively.

![[Pasted image 20240427151714.png]]
Simple mistakes in the 175B PPO-ptx model (InstructGPT 175B) compared to GPT-3 175B.
# Additional Figures