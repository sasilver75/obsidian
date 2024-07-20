November 25, 2022 (6 months before [[Let's Verify Step by Step]])
[[DeepMind]] (Uesato et al)
[Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)
#zotero 
Takeaway: Authors use ~1500 or so reasoning samples from a filtered GSM8K dataset to train [[Reward Model|ORM]] and [[Process Reward Model|PRM]] reward models, and see (using them via a form of [[Expert Iteration]]) which of them results in the training of a better policy/LM. In this case, both seem to perform equally well, though the PRM gives more reasonable reasoning traces. Authors disambiguate between *final answer error rate* and *trace error rate* (where the answer is correct, but reasoning is not).

Note: A year after this DeepMind paper, OpenAI did their [[Let's Verify Step by Step]] paper, which similarly did an analysis of PRMs applied to math, but the used a more capable base model, *significantly* more human feedback (800k vs 1.5k), and used the more challenging [[MATH]] dataset, instead of [[GSM8K]] (these were all critiques that OpenAI had of this GDM paper).


---

## Introduction
- Recent work has shown language models using step-by-step reasoning (eg) [[Chain of Thought|CoT]] to solve tasks.
- How should we best supervise such models?
	- Outcome-based approaches: Supervise the final result
	- Process-based approaches: Supervise each step of the reasoning process, including the last step outputting the final result.
- Process-based approaches emphasize human understanding -- in order to either demonstrate or select good reasoning steps, human annotators need to understand the task.
- An answer without an understandable explanation may often confuse more than it implies.
	- ((What you cannot explain, you don't understand.))
	- ((Correct answer with incorrect explanation isn't aligned behavior, because in some cases they clearly weren't following that reasoning to get to the correct answers -- they're "lying."))
- Authors conduct a comprehensive comparison between process and outcome-based approaches in the context of [[GSM8K]] math work problems.
	- We vary whether or not supervision is provided only on the final answers (outcome-based) or on individual reasoning steps (process-based).
	- All models used are based on [[Chinchilla]] (70B params, 1.4T tokens)
		- ((This was later a criticism by the [[Let's Verify Step by Step]] paper, but TBH I think this is large enough of a model to benefit from a PRM... It must come down to the scale of the supervision provided?))
	- When comparing these two, authors consider the ==trace error rate== (how often the model makes any mistake in reasoning trace *and* produces the final correct answer) and the ==final-answer error rate== (considers the final answer, ignoring reasoning trace)
		- Where a "reasoning trace" refers to all textual steps of reasoning including the last step where the answer is provided.
- Key Results
	- LMs supervised with outcome-based and process-based approaches lead to ==similar final-answer error rates==.
	- ==Both process- and outcome-supervised reward models learn to emulate process-based feedback.==
		- Surprisingly, we find that even reward models trained with outcome-based labels result in predictions that agree closely with the process-based labels.
		- Authors wonder whether this may be dataset-specific (GSM8K, where there's only one (?) real reasoning "route" to follow).
	- ==Low trace error requires either process-based *feedback or a reward model that emulates it*.== ((This is just basically saying that reward models of any sort are useful lol))
		- Models using RL directly against final-answer correctness resulted in high trace error (12.4%, compared to 3.8% for the best process-based method).
		- ==Note== that RL against a reward model rather than against final-answer correctness directly closes much of this gap, reducing trace error to 5.5%.


## Problem and Methods
Dataset and Evaluation Metrics
- Experiments are conducted on [[GSM8K]], composed of grade school math word problems.
	- Chosen because it's competitive, and contains natural language reasoning traces.
	- Split into: 7118 training, 256 validation set, 1319 test samples
- Authors report the *==final-answer error== rate* and the *==trace error rate==* (fraction of problems *with correct* final answer but incorrect reasoning) Also report *==Selective final-answer error  rate==* to assess performance when abstaining is allowed, and ==*out-of-distribution (OOD) error rate*== on pre-algebra problems from the [[MATH]] dataset to assess generalization.
	- Re: trace error rate, we're interested from a safety perspective in errors that remain undetected after applying easy-to-compute proxy metrics (eg final-answer errors).

Training Overview
- For math word problems, we want to generate a full reasoning trace: a newline-separated sequence of *steps* where the last step is expected to provide the final answer. For GSM8K, this is always an integer.
- We use an LM as a policy that maps the problem statement and steps-so-far to the next step.
- We also train LM [[Reward Model]]s, which score proposed completions or partial completions from the policy.

Supervised finetuning
- In SFT, we finetune an LM to maximize log likelihood of a sequence of target tokens, given a sequence of input tokens. We use the entire reasoning trace as a target.

==Reward Models== (data labeling for them)
- We evaluate two main approaches to training reward models (RMs)
	- In the ==outcome-supervised RM== ([[Reward Model|ORM]]), the binary label for each step indicates whether the resulting final answer of the *full sample* matched the reference final answer.
		- ((==Note:== This means that even if you have correct reasoning... but later provide wrong answer, the reasoning steps are marked as incorrect? This seems bad))
	- In the ==[[Process Reward Model]]== (PRM), the binary label after each steps indicates whether the steps so far are correct. We use human annotation for these labels.
		- ((==Note:==, Wait so if reasoning step 3 is bad, but reasoning step 4 is good, does the PRM rate 4 badly too, because the steps "so far" aren't correct? This seems bad))
- ((So it seems like both models are producing a score at every step, but the reference supervision/signal for an outcome-based model is "stupider" is the sense that it's all 0s or all 1s during training? But the PRM also has bits that are dumb.))

Decoding
- For test-time decoding, they generate K-96 samples of full solutions, then select the best sample (either by ensembling across the samples or by using an RM). Use temperature T=1.0 and the syntax from *Cobbe et al (2021)* to allow the model to decide when to use a calculator.
	- Authors also tried RM-reranking after each generated step (rather than the full solution), but found this led to slightly worse performance).
- ==When no RM is available==, we use *majority voting* ([[Self-Consistency]]), selecting the most common final answer from K samples, then selecting a random sample from among those yielding this selected answer
- ==When an RM is available==, we use *RM-weighted decoding* (also called ==[[Verifier Voting]]==), where we weight each sample according to the RM-estimator correctness probability, and select the final answer with the largest total weight, then select the sample with the highest RM score from those yielding the selected final answer.
	- This works slightly better compared to simply selecting the sample with the highest RM score.

RL via [[Expert Iteration]]
- All RL experiments use *Expert Iteration*. As a meta-algorithm, expert iteration alternates between two high-level operations:
	1. In ==Policy Improvement==, they combine a base policy with a search procedure to produce samples from a so-called "expert policy."
	2. In ==Distillation==, we perform supervised learning on these expert samples to improve the base policy towards the expert policy.
- We use 5 epochs and select the best model of the 5, based on final-answer test error with RM-weighted decoding, or majority voting if no RM is available.

==Policy Improvement==
- They consider three versions of a policy-improvement procedure.
	1. ==Final-Answer RL Approach==: Also called [[Self-Taught Reasoner]] (STaR), we generate K full traces per problem, and filter by final-answer correctness.
		- ((I don't see how this is STaR... or maybe it's kind of like STaR but without the rationalization part, or any of the iterative learning))
	2. ==ORM-RL==: We generate K full traces per problem, and select the sample with the highest score according to the ORM model.
	3. ==PRM-RL Approach==: We treat each step as an individual episode. At each step, we generate K candidate steps, and select the candidate with the highest PRM score, and continue from the selected step until the model outputs a step with the final answer indicator text (or otherwise, a maximum of 15 steps).

==Distillation==
- They use the same hyperparameters as the SFT training, similarly applying [[Early Stopping]] via validation loss, where the validation set is constructed from expert policy samples on the validation set.

Data Annotation
- [[Process Reward Model|PRM]] is trained on stepwise labels indicating whether the steps so far are correct, collected from human annotators.
- ==Annotators are presented with a problem statement, the reference solution from GSM8K, and the generated model solution, and we ask them to indicate the *first model step* with a *major mistake*, if any exist.==
	- From these annotations, we can label every step wit ha binary label indicating whether the steps *so far* are correct; all steps before the first major mistake are labeled "correct," while the remainder are labeled "incorrect."
	- ((==Note==: This isn't exactly what I thought PRM annotation was... I don't know if it's differently done in the later papers. Because it seems to me that you can still have correct reasoning steps *after* an incorrect reasoning step... but this doesn't allow for that.))
- Authors remove samples from annotators with low inter-annotator agreement, and for problems flagged by annotators as ambiguous... ending up with ==1560 model samples across 530 training set problems== (and then 162 samples for validation and 200 problems for test). 
	- ==((This is one of the critiques of the [[Let's Verify Step by Step]]/PRM800K paper, which used a lot more training data than this.))==


## Results
- Supervising final-answer correctness alone suffices for low final-answer error rate
	- When it comes to just the *final answer error rate*, it doesn't seem like either the ORM or PRM provided much additional lift compared to the first option of just doing policy improvement using the final-answer approach. Note that for a math problem, ORM-reward and outcome-based-reward are going to be *sort of* similar...
- ORM-supervised reward models approximate PRM labels
- Low trace error requires either [[Process Reward Model]]-based feedback or a RM that emulates it
- Comparing the process-based and outcome-based approach, they have similar final-answer error rates...
- Reward model for reranking only 
	- RM provide a boost to both trace and final answer accuracy
	- RM reranking significantly improves trace error...
- Optimizing against an RM outperforms optimizing final-answer correctness directly.  ((Seems to contradict the first bullet point in this section...))



## Discussion

When to use process vs outcome-based feedback?
- Outcome-based tends to be appropriate when a reliable complete evaluation metric is available, while process-based feedback is most appropriate otherwise.
- When low final-answer error is sufficient, outcome-based approaches provide a label-efficient method for obtaining this.
- When low trace error is desired, it's helpful to use process-based feedback, or an approximation of it.

Process-based approaches are more inline with human understanding, and facilitate human understanding, while helping to avoid *tampering incentives,* where RL agents corrupt their feedback mechanisms to receive positive feedback.

Authors expect process-based and outcome-based feedback to align more closely for math compared to other domains. For math problems, incorrect traces are typically harmful for reaching correct final answers.

## Related Work
- Math word problems are a popular domain for understanding "reasoning" in LMs; recent papers have demonstrated that few-shot prompting alone can lead to impressive GSM8K performance, with methods like CoT and Self-Talk...
- A large body of work studies multistep reasoning for LMs... our focus on supervision techniques for finetuning is complimentary to such improvements.
	- Authors call out the OpenAI *Recursively summarizing books with human feedback* paper as an example of process-based supervision, where they summarize full-length books by supervising individual summaries which are recursively composed.
		- This is an example where the outcome-based approach would be prohibitively expensive due to the cost and sparsity of such feedback, but process-based supervision of each step individually is possible.
- 




## Conclusion






Abstract
> Recent work has shown that asking language models to generate reasoning steps improves performance on many reasoning tasks. When moving beyond prompting, this raises the question of how we should supervise such models: outcome-based approaches which supervise the final result, or process-based approaches which supervise the reasoning process itself? Differences between these approaches might naturally be expected not just in final-answer errors but also in reasoning errors, which can be difficult to detect and are problematic in many real-world domains such as education. We run the first comprehensive comparison between process- and outcome-based approaches trained on a natural language task, GSM8K. We find that pure outcome-based supervision produces similar final-answer error rates with less label supervision. However, for correct reasoning steps we find it necessary to use process-based supervision or supervision from learned reward models that emulate process-based feedback. In total, we improve the previous best results from 16.8% → 12.7% final-answer error and 14.0% → 3.4% reasoning error among final-answer-correct solutions.


# Paper Figures

![[Pasted image 20240720115719.png|600]]
Lol at this chart... maybe it will make more sense after reading the remainder of the paper.

![[Pasted image 20240720140557.png]]
Three different options for Policy Improvement in the Expert Iteration RL training scenario (One using just final-answer string matching, another using an [[Reward Model|ORM]], and another using a [[Process Reward Model|PRM]]. We then "distill" (really, just SFT the same model) on these artifact/technique-aided solutions).

![[Pasted image 20240720140549.png]]
Comparison of various methods.

![[Pasted image 20240720141553.png]]

![[Pasted image 20240720142208.png]]
A "lemon-picked" example of incorrect reasoning/grammar, but a correct final answer. This is a trace error, as opposed to a final-result error.