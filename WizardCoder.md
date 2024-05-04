June 14, 2023 (About a month after the [[StarCoder]] paper, a collaboration of many institutions. About two months after the initial [[WizardLM]] paper)
[[Microsoft Research]]
Paper: [WizardCoder: Empowering Code Large Language Models with Evol-Instruct](https://arxiv.org/abs/2306.08568)
Takeaway: ...

----

Takes a cue from the 2-months-earlier [[WizardLM]] paper, and applies a modified version of the [[Evol-Instruct]] technique (Which they just call *Code Evol-Instruct* to a code-generating model, to finetune Code Alpaca.

Notes
- Despite its much smaller size, WizardCoder even surpasses Claude and Google's Bard in terms of pass rates on HumanEval and HumanEval+ (Which IIRC were datasets introduced in the [[Codex]] paper)
	- WizardCoder also ==beats all other open-source Code LLMs by a substantial margin, including [[StarCoder]]==, as well as superior code-gen results compared to the largest closed-source LLMs, like Claude, Bard, [[PaLM]], [[PaLM 2]], and [[LaMDA]], despite being considerably smaller in size.
- To adapt [[Evol-Instruct]] as proposed by WizardLM to the realm of code, we make the following modifications to the evolutionary prompt:
	1. Streamlined the evolutionary instructions by *==removing* deepening, complicating input, and In-Breadth Evolving==!
	2. Simplifier the form of evolutionary prompts by *==unifying* the evolutionary prompt template==.
	3. To address the specific characteristics of the code domain, we ==add two new evolutionary instructions==: *==code debugging==* and *==code time-space complexity constraints==*.
- Re: #2 above, the code evolutionary prompt template is as follows:
```text
Please increase the difficulty of the given programming test question a bit.

You can increase the difficulty using, but not limited to, the following methods:
{method}

{question}
```
Where {question} represents the current code instruction awaiting evolution, and {method} is the type of evolution, where we use one of five types:
```text
Add new constraints and requirements to the original problem, adding approximately 10 additional words.

Replace a commonly-used requirement in the programming task with a less-common and more specific one.

If the original problem can be solved with only a few logical steps, please add more reasoning steps.

Provide a piece of erroneous code as a reference to increase misdirection.

Propose higher time or space complexity requirements, but please refrain from doing so frequently.
```
- The authors use StarCoder 15B as a foundation (which makes the later comparisons to StarCoder sort of... eh), and initialize our dataset with Code Alpaca's 20k instruction-following examples.
For the fine-tuning, we use the following prompt:
```
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that approximately completes the request.

### Instruction:
{instruction}

### Response:
```
- Starting with the 20k instruction-following dataset called Code Alpaca, we iteratively employ the Evol-Instruct technique, ultimately ending up with ~78k samples. After each round of data evolution, we merge the evolved data from all previous rounds with the original dataset to finetune StarCoder and assess the pass@1 metric on HumanEval. ==Once we observe a decline in the pass@1 metric, we discontinue the usage of Evol-Instruct, and choose the model with the highest pass@1 as the ultimate model.==





Abstract
> Code Large Language Models (Code LLMs), such as StarCoder, have demonstrated exceptional performance in code-related tasks. However, ==most existing models are solely pre-trained on extensive raw code data without instruction fine-tuning==. In this paper, we introduce ==WizardCoder, which empowers Code LLMs with complex instruction fine-tuning==, by adapting the ==Evol-Instruct method== to the domain of code. Through comprehensive experiments on four prominent code generation benchmarks, namely HumanEval, HumanEval+, MBPP, and DS-1000, we unveil the exceptional capabilities of our model. It surpasses all other open-source Code LLMs by a substantial margin. Moreover, our model even outperforms the largest closed LLMs, Anthropic's Claude and Google's Bard, on HumanEval and HumanEval+. Our code, model weights, and data are public at [this https URL](https://github.com/nlpxucan/WizardLM)

# Paper Figures
![[Pasted image 20240504013945.png]]

![[Pasted image 20240504014135.png]]
Above: Domination by WizardCoder! But wait, why does this table show WizardCoder beating GPT-3.5 on HumanEval, but the previous Figure 1 seems to show GPT-3.5 slightly edging out WizardCoder?

![[Pasted image 20240504014612.png]]
