---
aliases:
  - SDSD
---
References
- Video: Nathan Lambert [Self-Directed Synthetic Dialogues](https://youtu.be/qP3rXJc_L5Y?si=1gVCNJVFkp0-3TB9)

Given:
- Topics (Subject of conversation; anything from Taylor Swift to cooking)
- Principles (watch for violation, like [[Constitutional AI|CAI]])
- Goals (planing principles for how the LM talks to itself; how to steer conversation to an end)
We feed these to a language model, which generates a Plan for the conversation.


"Not going to promise that training on this dataset is going to solve your IFT problems, but it can dramatically shift the style, and doing something like this in your own work on your own isn't too expensive... Encouraging people to go forth and try new wacky things with synthetic data. When generating synthetic data, you need to be serious about filtering. Everything can go wrong -- we try to get the LM to generate special tokens of its outputs when principle violations occur (it does this with 90% accuracy). You really need to debug the plan, then look at the dialogues, then look at the revisions. It takes a lot of careful reading. Revisions/critiques/LM feedback are sensitive and fragile. If you're doing procedural generation (eg from a set of topics/principles), it can become extremely unbalanced; about 15% of our conversations are about fruit, which doesn't reflect its relevance in the world. If you want every principle represented, you might have to do more sampling to get that data and then rebalance before doing RLHF."
- Mentions other interesting synthetic datasets:
	- ==NuminaMath-TIR== (From the Numina group that won a math competition; they essentially scraped math exams and formulated datasets, and added tool use to make sure the outputs were verified. This is like what the LLaMA 3.1 team is doing for Math.)
	- NVIDIA's ==Daring-Anteater== dataset, high quality instruction data used to train [[Nemotron-4]] 340B; helps on IFEval and other tests in AI2's early use.
	- [[Magpie]], using chat templates to get the model to fill in instructions on its own... language model writes what the user would have written! It can be used to generate both prompts and completions; a good example of creativity needed to close the gap with labs.
	- [[Persona Hub]], generating 1B personas to help generate diverse completions; it's interesting because it sends the signal of trying to generate the right scale of data needed to compete.