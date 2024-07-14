May 31, 2023
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
	- To compare multiple solutions, it's necessary to compute a single score for each solution -- this is important! ==We define the PRM score for a solution to be the probability the every step is correct under the PRM, which we implement as the probability that each step is correct under the PRM...== but other scoring strategies are possible and explored in Appendices.
	- When providing process supervision, they deliberately choose to supervise only up to the first incorrect step, making comparison between outcome and process supervision more straightforward -- both methods provide the same information, namely that every step is correct.


## Large-scale Supervision



## Out-of-Domain Generalization



## Discussion












Abstract
> In recent years, large language models have greatly improved in their ability to perform complex multi-step reasoning. However, even ==state-of-the-art models still regularly produce logical mistakes==. To train more reliable models, we can turn either to ==outcome supervision==, which ==provides feedback for a final result==, or ==process supervision==, which ==provides feedback for each intermediate reasoning step==. Given the importance of training reliable models, and given the high cost of human feedback, it is important to carefully compare the both methods. Recent work has already begun this comparison, but many questions still remain. We conduct our own investigation, finding that process supervision significantly outperforms outcome supervision for training models to solve problems from the challenging MATH dataset. Our process-supervised model solves 78% of problems from a representative subset of the MATH test set. Additionally, we show that active learning significantly improves the efficacy of process supervision. To support related research, we also release PRM800K, the complete dataset of 800,000 step-level human feedback labels used to train our best reward model.

# Paper Figures
![[Pasted image 20240714123617.png|400]]
The interface that they used to collect human feedback over reasoning trajectories to then train the PRM with.

![[Pasted image 20240714125157.png|600]]
An example of the granularity of PRMs

![[Pasted image 20240714130130.png|400]]
What's interesting here to me is that you can improve your % of problems solved (using a Best-of-N, on MATDH) approach by increasing from 10 solutions to 1,000 solutions, and it seems there's still juice left to squeeze. Curious about what sort of sampling/decoding strategy was used here, and what's optimal.

# Non-Paper Figures