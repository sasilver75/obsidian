
# The Why of DSPy

Say you're building some app that generates cover letters, given a user's resume and a job description. You go with GPT 3.5 as the LLM of choice, and start with a simple prompt -- but the generated letters are two verbose!
- As a result, you adjust the prompt, trying a suite of prompt hacks:
	- "Take a deep breath before you begin"
	- "Count to 25 silently and then proceed", etc.
- You even try some tested templates like [[Chain of Thought]]

Somehow, you make it work after hours of prompt-tweaking and evaluation.
Off to deployment!

But wait, Claude's Haiku model gets released with better performance and *half* the inference cost of GPT 3.5.
Switching out a model is easy enough, but you notice that Haiku is now performing poorly on your evaluation dataset.

> Epiphany Strikes: LLMs are very prompt sensitive. You'd have to essentially start from scratch to find the optimal prompt for the Haiku model.

# The What

DSPy exists to solve this problem
![[Pasted image 20240413164227.png]]
==By feeding a few examples of expected input/output for your pipeline, DSPy's compiler can automate optimizing your prompt against an evaluation metric of your choosing==.

This naturally raises questions: "How is this optimization being done?"

# The How

We first need to understand the components of a prompt:

![[Pasted image 20240413164532.png]]
((Above: A shitty diagram))


Optimizing a prompt involves modifying one or more of these components:
1. Prompting techniques
	- Strategies like [[Chain of Thought]] and [[ReAct]] that define the reasoning steps you want the LM to emulate.
2. Instructions
	- The "how" of a prompt; the string used to guide the LM to get desired output, given input.

One last thing to address before we talk about DSPy's optimizers -- *metrics*
- How can an optimizer evaluate a prompt (or any of its components) to be good or bad? We have to define a ==Metric.==
	- If the instruction is for the LM to answer math questions, you can create a metric that checks if the LM's answer matches the correct answer exactly. 
	- Note that ==the output of a metric function here can be a continuous number, a discrete rating, a boolean, etc.==


Let's talk about DSPy's Optimizers, and how they work:

1. ==Bootstrap Fewshot Optimizer==
	- Uses a teacher LM to ***==select the best demonstrations to include for the prompt==*** from a larger set of demonstrations provided by the user.
		- The teacher LM can be the original LM itself, or a more powerful model.
![[Pasted image 20240413171202.png]]
(Above: "demo" = demonstration (exemplar))

2. ==COPRO optimizer==
	- ==Finds the best-performing instruction for the model==.
	- ***Starting with a set of initial instructions, it generates variations of those instructions, evaluates each variation, and finally returns the best-performing instruction.***
![[Pasted image 20240413171315.png]]

3. ==MIPRO optimizer==
	- Finds the best-performing *combination of instruction and demonstrations!*
	- Works similarly to the COPRO optimizer; returns the best-performing combination of instructions and examples.
![[Pasted image 20240413171507.png]]

In a lot of these optimizers, the temperature ($\tau$) for the instruction or demonstration-generating model is *set high* in order to get more creative candidate samplings that are later filtered down, based on performance on the metric.


# The Verdict

Here's what he likes about DSPy:
1. It's not as expensive as it might seem: It costs $3-10 on average for 100 examples considered by the optimizer; you can do a lot of experimenting cheaply with DSPy.
2. ==Super easy migration==. If you want to switch your LM of choice, all you need to do is recompile and get your newly optimized prompt.
3. Integrations: As a framework focused on optimization, DSPy sits comfortably in the center of all frameworks, allowing you to integrate components from other libraries like Langchain for chain optimization, or Phoenix for evaluations.

That being said, some things to be cautious about.
1. Abstractions always come at a cost. If the domain of your LM is general, like generating custom letters, then DSPy can work really well, as it will likely know a lot about the domain and can generate great instructions and examples. This won't be the case if your domain is very specialized though -- in that case, you might be a far superior prompt engineer yourself.
2. Still early days -- the library is new and there are a lot of code changes being made.


# Things we didn't touch upon:
1. Fine-tuning: In DSPy you can optimize the fine-tuning data fed to your LM! It essentially uses a teacher model and the Bootstrap Few Optimizer (above) to generate a set of demonstrations that serve as additional training data to finetune the student model.
2. Assertions: There are functions you can call in DSPy to introduce constraints on LM output, ==like restricting the output to follow a certain format, like JSON.==
3. DSPy abstractions like signatures, modules , programs, traces, etc.
	- These tend to distract newcomers from the core selling point of DSPy, which is pipeline optimization. Read the original paper to learn more about this!