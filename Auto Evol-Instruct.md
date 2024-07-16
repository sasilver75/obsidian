June 2, 2024 (But was "released" on Twitter on July 15, 2024)
[[Microsoft Research]]
[Automatic Instruction Evolving for Large Language Models](https://arxiv.org/abs/2406.00770#:~:text=This%20paper%20proposes%20Auto%20Evol,models%20without%20any%20human%20effort.)
#zotero 
Takeaway: A method of complexifying and diversifying instructions with automatic evolution of an instruction-rewriting prompt, with the goal of learning an instruction-evolving prompt $e^*$ that can transform a seed instruction dataset into one that optimizes the performance of a model. At every iteration we sample a minibatch of data from X, and, in the "==Evol Trajectory Analysis==" (ETA) step, we use the *evol-LLM* to evolve each instruction $l$ times to create an evaluation trajectory, then, the *optimizer-LLM* scrutinizes teh trajectory and provides feedback on detected issues. In the following "==Evolving Method Optimization==" (EMO) step, we ask the *optimizer-LLM* to use the feedback to create an updated version of the evolving instruction (this is actually done $m$ times in parallel, and they use a Best-of-N approach to decide which new $e_t$ to keep). We do multiple iterations of this (ETA, EMO) process to yield an optimal evaluation instruction $e^*$.
- This differs from [[Evol-Instruct]] from the get-go, because the former method requires human experts to give a prior on the way in which instructions should be evolved (diversified, complicated) for a given domain.


---

## Introduction
- It's been hard to scale high-quality instruction-following datasets that have good complexity and diversity. 
- The [[Evol-Instruct]] framework, which takes high-quality starting data and iteratively diversifies or complicates it, has shown good performance across instruction-following, code-generation ([[WizardCoder]]), and mathematical reasoning ([[WizardMath]]).
	- As a downside, it relies on heuristic efforts -- whenever it's used for a completely new task, the methods for execution need to be redesigned, which requires expertise and cost.
	- The [[Auto Evol-Instruct]] process hopes to automate this, and aims to design evolving methods that can both automatically make instruction more complex, and keep the instruction-evolution process from avoiding evolution failure.
- [[Auto Evol-Instruct]] automatically designs evolving methods that make the given instruction data more complex, enabling almost cost-free adaptation to different tasks by only changing the input data of the framework.
	- A universal initial evolving method doesn't require humans to specify the rules of evolution -- instead, it can automatically analyze the input instruction and brainstorm evolution rules suitable for the given data.
	- Rather than having a fixed evolution method that won't generalize to all cases, they use an =="evol LLM"== and an =="optimizer LLM"== to evolve and optimize their universal initial evolving method to iteratively try to ensure the lowest failure rate for a given instruction dataset.
- Optimization process involves two critical stages:
	1. ==Evol Trajectory Analysis==
		- The ==optimizer LLM== carefully analyzes the potential issues and failures exposed instruction evolutional performed by an ==evol LLM==, generating feedback for subsequent optimization.
	2. ==Evolving Method Optimization==
		- The ==optimizer LLM== optimizes the evolving method by addressing these identified issues in feedback by the ==evol LLM==.
	- These two stages alternate and repeat to progressively develop and effective evolving method using only a ***SUBSET*** of instruction data.
		- Once the "optimal" evolving method is identified, we direct the evol LLM to convert the **ENTIRE*** instruction dataset into more diverse and complex forms, thus facilitating improved instruction tuning.


## Auto Evol-Instruct
- [[Evol-Instruct]] involves using a human-designed evolving method *e* to transform an original instruction dataset *X* into an *X_e* dataset that yields superior model performance.
	- The methods for complicating instructions vary greatly across different domains -- in coding, it might be "propose higher time or space complexity requirements," but that doesn't apply in the chat domain.
- Evol-Instruct's dependence on high expertise and limited scope restrict its broader use. We want to develop an automated framework that identifies an optimal instruction method $e^*$ that maximize the performance of the resulting $X_{e^*}$ .
	- $e^* = \underset{e}{argmax}Q(X_e)$ , where Q(X_e) is the performance of the resulting model tuned on X_e.
- [[Auto Evol-Instruct]] develops evolving methods for instruction evaluation that surpass those crafted by human efforts, while minimizing failures and ensuring successful execution of instruction evolution.
- Starts with a carefully-designed universal evolving method and a seed instruction dataset, then iteratively optimizes this initial evolving method to obtain the optimal evolving method over multiple steps of iteration.
- In each step, we randomly sample a minibatch from X and utilize the ***evol LLM*** to evolve each instruction in the batch $l$ times (it seems like recursively, maybe?). 
	- Then we use the ***optimizer LLM*** to evaluate the evolutionary trajectory of all instructions in the current batch to identify existing issues and generate feedback (==Evol Trajectory Analysis==). 
- The optimizer LLM will then make corresponding optimizations to the evolving method $e_{t-1}$ to obtain $e_t$ (==Evolving Method Optimization==).
	- To improve stability, they execute analysis optimization multiple times with sampling decoding in parallel to obtain $m$ optimized evolving methods, then we select the method with the lowest *evolution failure rate* as the final $e_t$.
- Once the optimal evolving method is identified, we apply it to guide the instruction evolution across the entire instruction dataset, resulting in an evolved dataset.
- ==Evol Trajectory Analysis== (ETA)
	- At optimization step t, the evolving method e_t-1 steers the ==evol LLM== to perform l rounds of evolution over a batch of data $X_t$, culminating in the evolutionary trajectory {$X_t, X_t^1, ..., X_t^l$}.
	- Following this, the ==optimizer LLM== scrutinizes the evolutionary trajectory to pinpoint and provide feedback $f_t$ on any issues detected. The optimizer LLM is used to identify issues emerging during the instruction evolution process and offer subsequent feedback for the optimization of the evolving method in the EMO stage.
- ==Evolving Method Optimization== (EMO)
	- We use the optimizer LLM to optimize the evolving method in response to insights gathered from the previous ETA stage, in accordance with the overall instruction evolution requirements.
	- During step $t$, the ==optimizer LLM== refines the evolving method $e_{t-1}$ by leveraging the feedback $f_t$, yielding an updated version of the evolving method $e_t$.
	- The optimizer LLM sometimes struggles to provide constructive feed *and/or* enhance the evolving method -- we draw some inspiration from [[Self-Consistency]], and implement a strategy where ==at each step the optimizer LLM conducts *m* times of decoding analysis and optimization with sampling decoding== (?).
		- This results in m different potential improved evolving methods $e_t^1$ to $e_t^m$.
		- We divide the instruction data into training data X and development set D, and use the obtained potential methods to evolve instructions in D and generate corresponding response sets, eg $R_{e_t^1}$. For a given evolving method $e_t^i$, we calculate its evolution failure rate:
		- ![[Pasted image 20240715175446.png|300]]
		- Above: |D| is the size of the development set, F(r) is a function determining whether the instruction evolution failed (1 for failure, 0 for success). 
			- They have a series of rules to determine whether evolution has failed based on the reaction of the evol LLM when generating answers for evolved instructions. ==((Honestly these seem heuristic and pretty shitty))==
		- ==The evolving method (of the $m$) generating the lowest evolution failure rate is selected as the subsequent step's evolving method $e_t$.==

## Experiment and Analysis
- Authors investigate seed datasets, models of varying sizes, and configurations of both the evol LLM and optimizer LLM.
- Instruction Following: Improves by .44-.64 on MT-Bench compared to seed data (0-10 is the range for MT-Bench).
- Math Reasoning: On Mistral-7B, improved GSM8K by 13.84 compared to the seed data  (GSM8K is usually out of 100, or as a percentage).
- Code Generation: At 33B scale, improved HumanEval by 5.4 compared to the seed data (HumanEval is usually out of 100, or as a percentage).
	- Authors say their result remains competitive even when compared with [[DeepSeek-Coder]]-Instruct-33B, which uses the same base model but uses a larger instruction dataset (about 2B tokens (more?)).
- ((==For all of these, I'm curious about the quality of the seed data.==))
- 


## Appendices
- Authors categorize prevalent scenarios of instruction evolution *failure*:
	1. Stagnant Complexity: Doesn't exhibit enhanced complexity
	2. Insufficient Qualification: Evolved instructions lack necessary qualifications, necessitating additional inquiries for generating a meaningful response.
	3. Loss of Key Information: Evolved instruction omits crucial details from the original instruction, leading to a need for supplementary information before a 
- Authors analyze 200 instructions from GSM8K, and subject them to evolution using the *initial evolving method*. It fails to adequately account for the complexity inherent in the task of evolving instructions.
	- Tendency to alter the core nature of the problem
	- The introduction or irrelevant details
	- The generation of contradictions with the original problem setup
	- Overlooks the unique attributes of mathematical instructions
	- (More on this in the paper figures, see tables)
- Settings
	- Mini-batch size 10
	- Development set size 50
	- Optimizer LLM temperature .6
	- Top P to 0.95 ([[Top-P Sampling]]?)
	- Evol LLM temperature 0 (This makes sense that we want our implementation to be "to-the-letter," and our evolutions (especially given that we're trying a few $m$) to have a little "creativity")
	- 5 optimizations performed in each step by default.


Abstract
> Fine-tuning large pre-trained language models with [[Evol-Instruct]] has achieved encouraging results across a wide range of tasks. However, designing effective evolving methods for instruction evolution requires substantial human expertise. This ==paper proposes Auto Evol-Instruct, an end-to-end framework that evolves instruction datasets using large language models without any human effort.== The framework automatically ==analyzes and summarizes suitable evolutionary strategies for the given instruction data and iteratively improves the evolving method based on issues exposed during the instruction evolution process==. Our extensive experiments demonstrate that the best method optimized by Auto Evol-Instruct outperforms human-designed methods on various benchmarks, including [[MT-Bench]], [[AlpacaEval]], [[GSM8K]], and [[HumanEval]].


# Paper Figures
![[Pasted image 20240715162211.png|600]]
The optimization process has two stages:
1. ==Evol Trajectory Analysis==: The EvolLLM evolves our minibatch $X$ $l$ distinct times (it seems recursively, maybe?), and the Optimizer LLM analyzes issues/failures in instructions produced by the Evol LLM.
2. ==Evolving Method Optimization==: Optimizer LLM optimizes the evolving method by addressing these identified issues in feedback.
These two stages alternate and repeat to develop an effective evolving strategy, using only a SUBSET of the initial instruction data. Once we reach our optimal evolving strategy, we direct the evol LLM to convert the entire instruction dataset into more diverse and complex forms.
So in comparison to [[Evol-Instruct]], in which human experts craft the evolving instructions, it turns out there here in [[Auto Evol-Instruct]], our iteratively optimized evolving instructions produce better results.

![[Pasted image 20240715173917.png|600]]
The initial evolving method that's always used as $e_0$ (or is it $e_1$), from which we ultimately evolve the $e^*$.
- Note how they use these (eg) `#Methods List#` tags in their instruction.
- Generate possible methods to make an instruction more complex, then generate a plan based on list of methods, then execute the plan step by step to provide a rewritten instruction. They "limit" this rewriting to only adding 10-20 words into the instruction. They then review the rewritten instruction for any "unreasonable" parts.
	- ((Honestly, there's so much ambiguity in this prompt, and I feel like it suffers from the authors not necessarily being native english speakers))

![[Pasted image 20240715182542.png|600]]
It's unclear to me what the "large" size is (or how to compare it to [[WizardMath]]/MetaMath's 70B, but it's interesting to see that the "large" Auto Evol-Instruct model very slightly edges them out on math performance. That would be impressive if the model were a 33B, but perhaps a little disappointing if it were 70B, which would mean that automatically-generated instructions aren't much better than expert human instructions.
What this chart doesn't show is low large instruction sets are for either the comparison models or the Auto Evol-Instruct models. For example -- DeepSeek-Coder-Instruct gets the highest HumanEval score of the open-source models, but this table doesn't show how much larger its instruction set is than the "large" Auto Evol-Instruct model.
On that subject, I'm also curious about at what point this technique will "top out."

![[Pasted image 20240715183348.png|300]]
![[Pasted image 20240715183401.png|300]]
![[Pasted image 20240715183422.png|300]]
This chart isn't clear to me... I think the left refers to the number of rounds of optimization, where a round includes both ETA and EMO, and the right shows that the number of optimization steps increases, performance can increase, but then monotonically decline???

![[Pasted image 20240715184945.png|300]]
Above: It seems like it's the general idea to use a stronger LLM for the optimizer LLM, which has the more difficult task of both critiquing evolutions and editing the evolution instructions.

![[Pasted image 20240715191504.png|600]]
The prompt for ==Evolutional Trajectory Analysis== (ETA)
- This seems to make me think that the "l" instruction transformations it seems like are done recursively.

![[Pasted image 20240715192123.png|500]]
The prompt for Evolving Method Optimization (EMO)
See that we slot in our $e_t$ into the {Evol Prompt} bit...
ESL as fuck

![[Pasted image 20240715192330.png|600]]
Examples of instruction evolution failure cases, and the hokey heuristic detection rules that they use to find them. I'm sure that these detection rules are specific to the use of GPT-4 as the responding model? 

![[Pasted image 20240715192644.png|600]]
Problems that result from using the initial instruction evolution method.

![[Pasted image 20240715192914.png|600]]
More issues that result from the initial instruction evolution method.
