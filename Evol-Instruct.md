April 24, 2023 -- [[Microsoft Research]]
Paper: [WizardLM: Empowering Large Language Models to Follow Complex Instructions](https://arxiv.org/abs/2304.12244)

See: [[WizardLM]], the model for which this technique was developed

---
Note that Evol-Instruct is also sometimes used to refer to the dataset produced in the [[WizardLM]] paper (this dataset is also just sometimes referred to as "WizardLM Dataset")
A ***rewritten*** set of 250k English instruction-response pairs based on the [[Alpaca]] data.
- Instructions are rewritten to:
	- Make them more complex
	- Create a new, more-specialized instruction by prompting ChatGPT.
- In the next step, ChatGPT is used to generate the corresponding responses.
- Low-quality instruction-response pairs are filtered using heuristics.
- The process is repeated three times.
---
