---
aliases:
  - STaR
---
March 28, 2022
Stanford, [[Google Research]] (lead [[Eric Zelikman]])
[STaR: Bootstrapping Reasoning with Reasoning](https://arxiv.org/abs/2203.14465)
#zotero 
Takeaway: ...

"We propose what is, to our knowledge, the first technique to allow a pre-trained LM to iteratively use its LM capacity to improve itself."

Thought: This paper involves a problem dataset (x, y), and takes advantage of an ICL dataset (x, r, y), to prompt the language model to generate both $\hat{x}$ and $\hat{y}$ for the $x$s in the problem dataset. This paper only really seems to care about whether $\hat{y}$ matches $\hat{x}$, but why not care about the generated rationale $\hat{r}$ too? There's a lot of research about [[Process Reward Model]]s that provide granular feedback over reasoning trajectories. Right now, as far as I know, they're just *assuming* that the reasoning trajectories they generate under rationalization are going to be correct, which might not be true. Even using PRMs there would perhaps be useful.

Similar papers include: 
- [[Self-Taught Optimizer]] (STOP, same author, October 3, 2032)
- [[Reinforced Self-Training]] (ReSTEM, Dec 11, 2023)
- [[Quiet STaR]] (Same author, March 14, 2024)

----
## Introduction
- Recent work has shown that explicit intermediate reasoning "rationales" can improve LM performance, but it's hard to induce correct rationale generation in models without drawbacks:
	1. Construction of fine-tuning datasets of rationales by humans or templates is expensive and only work when general solutions are already known.
	2. Leveraging [[In-Context Learning]] and including a few rationale examples in the prompt. This improves accuracy on mathematical/symbolic reasoning tasks relative to prompting directly without rationales, but these generally substantially *underperform* models that are fine-tuned to directly predict answers, using larger datasets.
- In this paper, they adopt a different approach: ==we iteratively *bootstrap* the ability to generate high-quality rationales==:
	- We few-shot prompt a LM to self-generate rationales, and then refine the model's ability further by fine-tuning on those rationales that lead to correct answers.
	- We repeat this procedure, using the improved model to generate the next rationale training set next time!
	- Idea: Improvements in rationale generation improve the training data and improvements in training data further improve rationale generation.
- ==This loop *eventually fails to solve any new problems in the training set,* because it receives no direct training for problems it *fails* to solve!==
	- To combat this, we propose ==rationalization==: For each problem the model fails to answer correctly, we generate a *new rationale* by *providing the model with the correct answer* (this requires having the correct answer, of course)!
	- Idea: If the model can't reason its way forwards to the correct solution, we let the model reason *backward,* generating a useful rationale given the correct answer. These are also collected as part of the training data.
- [[Self-Taught Reasoner|STaR]]
	- In each iteration, 
		- We first construct a finetuning dataset by attempting to solve the dataset using the current model's rationale generation ability.
		- Then we augment this dataset using **==rationalization==**, where we justify ground-truth answers to problems that we failed to otherwise solve. (==Interestingly, I think that the reasonings generated in the rationalization case are just assumed to be correct, which feels like it might be a source of error.==)
		- Finally, we fine-tune the language model on the combined dataset.

## Background and Related Work
- Regarding in-context learning, differences in prompt configurations have dramatic effects on performance, some have found that replacing few-shot prompts with optimizable "soft prompts" results in gains, etc. Instead of emphasizing the representation of the question, we focus on model output.
- Regarding rationales, it's been demonstrated that using few-shot CoT reasoning can improve model performance without fine-tuning, and it's been shown that curriculum learning could help solve formal math problems.
- Regarding iterated learning, they were inspired by an "Expert Iteration (ExIt)" paper, where a loop of self-play is done by an "apprentice," followed by imitation learning with feedback from a slower "expert" and then the replacement of the expert with the now-improved apprentice. 

## Method
- We start with both:
	- An initial dataset of problems $x$ with answers $y$, called $D=\{(x_i, y_i)\}_{i=1}^D$ 
	- A smaller prompt set of problems $x$ with rationalizations $r$ and answers $y$, called $\mathcal{P}=\{(x_i, r_i, y_i)\}_{i=1}^P$ , where $P$ is generally much smaller than $D$, e.g. $|P|=10$.
- We concatenate our prompt set to each example in D, which encourages the model to produce a rationale $\hat{r}$ for every $x$, followed by some predicted answer $\hat{y}$.
- For those that don't result in a correct predicted answer, we use "rationalization," where we provide the correct answer y and just ask it to generate the rationalization r.
- We filter/accumulate the correct (x, r, y) tuples, and finetune the model on the constructed training set.
- We repeat this process n times, or until performance plateaus.

## Discussion and Challenges
- An essential question is what role rationalization plays
	- It could be framed as an off-policy estimate of the objective.
	- Due to the low sampling temperature, outputs without rationalization correspond to the examples where the model is most confident in its answer. This results in these examples providing a weaker gradient signal to learn from.
- An alternative to rationalization is to use high-temperature sampling, but in practice we found this counterproductive -- it substantially increases the likelihood of a correct answer despite incorrect reasoning, and training on bad or irrelevant reasoning prevents generalization. Authors note that maybe you could use multiple higher-temperature samples along with [[Self-Consistency]] to maybe apply STaR to a dataset of only questions without answers ((but it's not clear to me which rationale you select, once you select the modal y)).
- Authors talk about the idea of "disabling" the few-shot exemplars in later stages of training, since the presence of them seems to dramatically reduce "drift" of later rationales -- it's sort of a similar dial to temperature that could be played with. When exactly to disable few shot exemplars is unclear.


## Appendix
- We came across a variety of interesting failure cases for common-sense reasoning in [[CommonsenseQA]], including "Question Implies Answer," "Begging the Question," "Exercise to Reader," "World State Assertions," and "Red Herrings", and "Hint Shortcutting"


Abstract
> Generating step-by-step "chain-of-thought" rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering. However, inducing language model rationale generation currently requires either constructing massive rationale datasets or sacrificing accuracy by using only few-shot inference. We propose a technique to ==iteratively leverage a small number of rationale examples and a large dataset without rationales, to bootstrap the ability to perform successively more complex reasoning==. This technique, the "==Self-Taught Reasoner" (STaR),== relies on a simple ==loop==: ==generate rationales to answer many questions, prompted with a few rationale examples; if the generated answers are wrong, try again to generate a rationale given the correct answer; fine-tune on all the rationales that ultimately yielded correct answers; repeat==. We show that STaR significantly improves performance on multiple datasets compared to a model fine-tuned to directly predict final answers, and performs comparably to fine-tuning a *30x larger state-of-the-art language model* on CommensenseQA. Thus, STaR lets a model improve itself by learning from its own generated reasoning.


# Paper Figures

![[Pasted image 20240715214956.png|600]]
It's a pretty simple workflow! Given a problem dataset of (x, y) pairs and a much smaller prompt dataset of (x, r, y) to use for ICL, we prompt our language model to generate $\hat{r}$ and $\hat{y}$ for the $x$ in the problem dataset. The correct examples make it into a training set. For those generations that result in a in incorrect $\hat{y}$ (meaning the model wasn't able to "reason forwards" to the answer), we use "rationalization," where we give the model the correct $y$, and just ask it to generate the correct reasoning/rationale $\hat{r}$. Here, I think all of the rationalizations are assumed to be correct, which seems like a source of possible error...

![[Pasted image 20240715232122.png|600]]
Algorithm

![[Pasted image 20240715232322.png|200]]
An example of an in-context exemplar from our prompt dataset.


![[Pasted image 20240715232354.png|500]]
See the improvement that rationalization makes to the efficiency of the process! I'm still curious what the effect of training on incorrect rationales might be, in the case of rationalization. Even if it these bad apples don't result in a meaningful decrease in accuracy, if they result in the misaligned behavior of generating incorrect rationales for correct answers (which wouldn't be reflected in this chart), that's something to worry about!


Below are some examples of failures of rationales observed when testing generations against the [[CommonsenseQA]] dataset.
![[Pasted image 20240715233724.png]]
![[Pasted image 20240715233728.png]]
![[Pasted image 20240715233733.png]]
![[Pasted image 20240715233747.png]]
![[Pasted image 20240715233801.png]]
![[Pasted image 20240715233804.png]]


Some examples of rationalizations on [[CommonsenseQA]] (where the answer is provided, and we want to generate a rationalization... which seems to also include an answer. I wonder if the answer even doesn't accurately identify the correctly-marked answer, hah!)
![[Pasted image 20240715234230.png|600]]
See description above

