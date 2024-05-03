April 24, 2023
[[Microsoft Research]]
Paper: [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)
Takeaway: A [[LLaMA]] model fine-tuned using synthetic instruction-tuning data generated from the [[Evol-Instruct]] technique, which was introduced in the same paper.

----
Note also that Microsoft has released datasets used in training WizardLM:
- WizardLM_evol_instruct_70k
- WizardLM_evol_instruct_V2_196k

Notes:
- Evol-Instruct is a novel method using LLMs instead of humans to automatically mass-produce open-domain instructions of various difficulty levels, to improve the performance of LLMs.
	- Starting from a simple instruction "1+1=?", the method randomly selects either ==In-depth Evolving== or ==In-breadth Evolving== to upgrade a simple instruction to a more complex one, or to a new one (to increase diversity)?
	- ==In-depth Evolving==: Includes five types of operations:
		- Add constraints
		- Deepening
		- Concretizing
		- Increasing reasoning steps
		- Complicate input
	- ==In-breadth Evolving==: Mutation, i.e. generating a *completely new instruction*  based on the given instruction
	- All six of the above operations are implemented by prompting an LLM with specific prompts. 
	- Since the evolved instructions are generated from LLMs, sometimes evolving will fail! So we adopt an *instruction eliminator* to filter the failed instructions, which is called ==Elimination Evolving==.
- Authors fine-tune LLaMA, and compare their dataset to the one used by [[Alpaca]] (generated with [[Self-Instruct]]) and the 70k ShareGPT (shared by real users) used by [[Vicuna]].
	- This is nice, because one of these is a synthetic dataset, and the other a crowd-sourced (and volunteer-biased) dataset.
- Alpaca's training data (generated from only 175 human-created seed instructions) is the initial starting point; They execute four epochs of evolution using OpenAI's ChatGPT API, to finally obtain 250k instructions (from 52k starting). 
- From the 250k, for fairness's sake, they sample 70k examples (same as the number of examples used for Vicuna) and fine-tune a LLaMA 7B model, with the resulting model being called [[WizardLM]].


Abstract
> ==Training== large language models (LLMs) with ==open-domain instruction following data brings colossal success==. ==However, manually creating such instruction data is very time-consuming and labor-intensive==. Moreover, humans may struggle to produce high-complexity instructions. In this paper, ==we show an avenue for creating== large amounts of ==instruction data== with varying levels of complexity ==using LLM instead of humans==. ==Starting with an initial set of instructions, we use our proposed [[Evol-Instruct]] to rewrite them step by step into more complex instructions.== Then, we mix all generated instruction data to ==fine-tune LLaMA. We call the resulting model WizardLM==. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that ==instructions from Evol-Instruct are superior to human-created ones==. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation, WizardLM achieves more than 90\% capacity of ChatGPT on 17 out of 29 skills. Even though WizardLM still lags behind ChatGPT in some aspects, our findings suggest that fine-tuning with AI-evolved instructions is a promising direction for enhancing LLMs. Our code and data are public at [this https URL](https://github.com/nlpxucan/WizardLM)

# Paper Figures
![[Pasted image 20240502180347.png]]


# Non-Paper Figures
- 