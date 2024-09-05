May 31, 2023 (6 months after [[Solving math word problems with process- and outcome-based feedback]])
[[OpenAI]] (incl [[Ilya Sutskever]])
[Let's Verify Step by Step](It is a significant battle that has unfortunately ended up getting drown out by all the warfare that has happened since it occurred. I wish it would get some more attention and study.)
#zotero 
Takeaway: The canonical [[Process Reward Model]] (PRM) paper! Instead of using outcome supervision (providing feedback for a *final result*), this paper provides *process supervision* (providing feedback for each intermediate reasoning step). Authors find that process supervision outperforms outcome supervision in the context of the [[MATH]] dataset. Authors release [[PRM800K]], the complete dataset of 800,000 step-level human feedback labels used to train the best process reward model.


---

## Introduction
- LMs can solve tasks requiring complex multi-step reasoning by generating solutions in a step-by-step [[Chain of Thought|CoT]] format, but even SoTA models are prone to producing falsehoods/hallucinations during moments of uncertainty.
	- In CoT, even a small logical error mid-chain is enough to derail a larger solution.
	- Detecting and mitigating these mistakes is essential to improving reasoning capabilities!
- This paper involves training [[Process Reward Model]]s to discriminate between desirable and undesirable outputs; the model can then be used in a reinforcement learning pipeline, or to perform search via [[Rejection Sampling]]
	- Outcome-supervised Reward Models (ORMs) are trained using only the final result of the model's CoT, while Process-supervised reward models (PRMs) receive feedback for each step in the CoT.
	- PRMs have some benefits:
		- More granular feedback, specifying the exact location of errors that occur.
		- Easier for humans to interpret
		- More directly rewards models for following a human-endorsed chain-of-thought.
		- ==Models trained with *outcome* supervision regularly use incorrect reasoning to reach the correct final answer, and PRMs have been shown to mitigate this misaligned behavior.==
	- Despite these advantages, Uesato et al. (2022) found that ORM and PRMs led to similar final performance in the domain of grade school math ([[GSM8K]]). We conduct our own detailed  analysis using a more capable model, more human feedback, and on a more challenging [[MATH]] dataset.
- Authors construct their own 1.5B "MathMix" large scale dataset for use in a lightweight pretraining stage, before finetuning on comparably smaller dataset like MATH and PRM800K; the goal is just to bridge the distribution shift.
	- Large model trains for 2 epochs (3B tokens)
	- Small model trains on a 1B subset that excludes critiques data, trained for 6 epochs (~6.6B tokens)
- Contributions:
	- We show that process supervision can train *much more reliable reward models* than outcome supervision. They reach SoTA 78.2% of problems on a subset of MATH.
	- We show that a large RM can reliable approximate human supervision for smaller reward models, and that it can be used to conduct large-scale data collection ablations.
	- That [[Active Learning]] leads to a 2.6x improvement in the data efficiency of process supervision.
	- Release the full process supervision dataset, [[PRM800K]], to promote related research.

## Methods
- Because all of the problems in the MATH dataset have automatically-checkable answers, it's easy to automate outcome supervision. ==In contrast, there's no simple way to automate process supervision==, so authors rely on human data-labelers to provide process supervision, labeling the correctness of each step in model-generated solutions.
- Authors have experiments in two regimes:
	- Large-scale: They finetune all models from GPT-4; focusing on training the most reliable ORM and PRM possible; because the training sets aren't comparable (human-generated vs automatic?), the authors say these aren't ideal for making an apples-to-apples comparison.
	- Small-scale: I think they use an LLM to "supervise small-scale model training," meaning to give the PRM feedback.
- We evaluate reward models by its ability to perform Best-of-N search (see: [[Best-of-N Sampling]]) over uniformly-sampled solutions from the generator. 
	- For each test problem, we select the solution ranked highest by the reward model and automatically grade it based on its final answer, and report the fraction that are correct.
- All large-scale models are finetuned from the base [[GPT-4]] model, which hasn't underwent [[Reinforcement Learning from Human Feedback|RLHF]]. The small-scale base models are similar in design to GPT-4, but pretrained with roughly 200 less compute. As an additional step, we finetune the models on a dataset of roughly 1.5B math-relevant tokens.
- To collect process supervision data, human data-labelers are presented with step-by-step solutions to MATH problems sampled by the large-scale generator; the task is to assign each step in the solution a label of *positive, negative, or neutral*.
	- We label solutions exclusively from the large-scale generator to maximize the value of our limited human-data resource. We refer to the entire dataset of step-level labels collected as [[PRM800K]]. It contains 800k step-level labels across 75k solutions to 12k problems.
	- Authors want to surface solutions to the expensive human data labelers that involve obvious errors -- they'd prefer to surface solutions that are more likely to fool our best reward model.
	- ==Authors choose to surface *convincing wrong-answer solutions* that are highly-rated by the current best PRM (convincing) but still reach a wrong final answer (wrong-answer).==
	- So at each iteration, they generate N solutions per problem and surface only the top-K most convincing wrong-answer solutions to data-labelers.
- They train their ORMs for comparisons sake following a similar methodology to Cobbe et al. (2021), uniformly sampling a fixed number of solutions per problem from the generator, and training the ORM to predict whether each solution is correct or incorrect, using the ORM's prediction at the final token as the overall score for the solution.
- They train their PRMs to predict the correctness of *each step* after the last token in each step. This prediction takes the form of a single token, and we maximize the log-likelihood of these tokens during training. To determine step-level predictions at test time, we can perform a single PRM forward pass over the whole solution. 
	- To compare multiple solutions, it's necessary to compute a single score for each solution -- this is important! ==We define the PRM score for a solution to be the probability the every step is correct under the PRM, which we implement as the product of the probability that each step is correct under the PRM...== but other scoring strategies are possible and explored in Appendices.
	- When providing process supervision, they *deliberately choose to supervise only up to the first incorrect step*, making comparison between outcome and process supervision more straightforward -- both methods provide the same information, namely that every step is correct.

## Large-scale Supervision
- We train the large-scale PRM using the step-level labels in PRM800K.
- The PRM reachers higher performance than the ORM (and a majority voting baseline) for all values of N (in a best-of-N), and the  performance gap only widens as N increases.

## Small-scale Synthetic Supervision
- We find that the PRM outperforms the ORM at large scale, but we need to isolate two confounding factors:
	1. Training sets for the ORM and PRM are not directly comparable: The PRM training set was constructed using [[Active Learning]], and is biased towards answer-incorrect solutions, and is an OOM smaller.
	2. The final-answer grading will provide positive labels to spurious solutions that reach the correct final answer, despite incorrect reasoning.
- We can't easily ablate these factors... so they use the large-scale PRM to supervise small models (?), simulating a large amount of data collection at a modest cost.
- ...

## Out-of-Domain Generalization
- They evaluate the ORM and PRM on a held-out set of 224 STEM questions from recent AP Physics, Calculus, Chemistry, AMC10, AMC12 exams, released after the pre-training dataset was compiled, and report the best-of-100 performance of each model in Table 1.
	- PRM outperforms the ORM and the majority voting baseline, showing that the PRM can at least tolerate a moderate amount of distribution shift and that its strong performance holds up on a fresh test questions.


## Discussion
- Credit Assignment
	- A clear advantage of PRMs is that they provide more precise feedback than outcome supervision.
		- ORMs face a difficult credit-assignment task -- to generalize well, it must determine where an incorrect solution went wrong. This is difficult for hard problems!
		- PRMs instead specify how many of the first steps were in fact correct, as well as the precise location of the incorrect step.
- Alignment Impact
	- Process supervision has several advantages over outcome supervision related to AI alignment!
		- Process supervision is more likely to produce interpretable reasoning, since it encourages models to follow a process endorsed by humans.
		- Inherently safer, directly rewarding aligned CoTs rather than relying on outcomes as as proxy signal for aligned behavior. Outcome supervision can result in models that become misaligned after learning to exploit/game the reward signal.
	- Process supervision seems to incur a *negative [[Alignment Tax]],* meaning it improves performance.


## Related Work
- Outcome vs Process Supervision
	- *Uesato et al. (2022)* is closely related work, comparing the impact of outcome and process supervision in the domain of grade school math. It found that both methods led to similar final-answer error rates. There are 3 main differences in our study:
		- We use a more capable model, more human feedback, and on a more challenging [[MATH]] dataset.
		- We believe that Uesato's claim that PRM == ORM can be explained primarily by the difference in the scale of the supervision -- if only use a small amount of process supervision and a large amount of outcome supervision, it does in fact lead to similar performance.
- Synthetic Supervision
	- Gao et al (2022) similarly uses a large reward model to supervise the training of smaller models, using a gold-standard reward model to replace human feedback. 
	- Our paper's use of a large-scale reward model to supervise the smaller reward model shares similarities.


Appendix includes information on 
- MathMix construction
- Training data collection
- PRM training details
- examples of PRM scored trajectories







Abstract
> In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even ==state-of-the-art models still regularly produce logical mistakes==. To train more reliable models, we can turn either to ==outcome supervision==, which ==provides feedback for a final result==, or ==process supervision==, which ==provides feedback for each intermediate reasoning step==. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

# Paper Figures
![[Pasted image 20240714123617.png|400]]
The interface that they used to collect human feedback over reasoning trajectories to then train the PRM with.

![[Pasted image 20240714125157.png|600]]
An example of the granularity of PRMs

![[Pasted image 20240714130130.png|400]]
What's interesting here to me is that you can improve your % of problems solved (using a Best-of-N, on MATDH) approach by increasing from 10 solutions to 1,000 solutions, and it seems there's still juice left to squeeze. Curious about what sort of sampling/decoding strategy was used here, and what's optimal.

![[Pasted image 20240716162809.png|600]]
See PRMs outperforming ORMs 

![[Pasted image 20240716181533.png]]
MathMix composition

PRM800K consists of 1,085,590 step-level labels over 101,599 solution samples. He present the whole unfiltered datasets as PRM800K. During training they discard labels used for QC, as well as any step-level labels for which the labeler was unable to complete the task, so the filtered dataset contains about 800k step-level labels over 75,000 solutions.
In the first phase, they generate multiple alternative completions at each step of a solution, and ask raters to add supervision. This resulted in labeled spending a lot of time supervising long uninteresting solutions.
For most of the data collection (phase 2), they sample N solutions per problem from the generator, rank them with our current best PRM, and surface the highest-scoring wrong-answer solutions to our labelers. We retrain the PRM using latest data, and do this for 10 generations. This [[Active Learning]] strategy changes the balance of our dataset considerably.
![[Pasted image 20240716182010.png]]
Above: How the difference in phase 1 vs phase 2 (larger) differ in terms of dataset balance.

![[Pasted image 20240716182318.png]]
Cherry-picked examples showing best-of-1860 solution from the generator as ranked by the large-scale PRM.
![[Pasted image 20240716182343.png]]
Another true positive

![[Pasted image 20240716182357.png]]
A true negative, in which the reasoning trajectory is incorrect, which is reflected by the score given by the PRM 

![[Pasted image 20240716182457.png]]
A false positive, in which the reward model assigns good scores to a bad trajectory. It's interesting that in this case the error is like... local to one reasoning step.


# Non-Paper Figures