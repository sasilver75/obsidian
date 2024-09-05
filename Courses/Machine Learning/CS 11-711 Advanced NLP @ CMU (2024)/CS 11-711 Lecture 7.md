# Topic: Prompting

---

## Prompting Fundamentals

Basic Prompting
- Append a textual string to the beginning of the sequence and complete it.
	- x = When a dog sees a squirrel, it will usually
		- GPT-2 Small: "be afraid of anything unusual. As an exception, that's when a squirrel is usually afraid to bite." (Ancestral Sampling)
		- GPT-2 XL: "lick the squirrel. It will also touch its nose to the squirrel on the tail and nose if it can." (Ancestral Sampling)

Standard Prompting Workflow
- We tend to use prompting to solve problems, not just to complete text.

![[Pasted image 20240615130836.png|300]]
Input: {An Amazon Review}
Template: {Insert Input}. The sentiment of this review is {blank}


A particular variety that we use very broadly nowadays is a Chat Prompt
- Usually, we specify inputs in a format like the ==OpenAI Messages Format==:
	- We have a list of outputs, and each list is given a *role* and *content*.
![[Pasted image 20240615131211.png]]

There are some other templates that are out there, though -- here's the ==LLaMA Chat Template== and ==Alpaca Template==
![[Pasted image 20240615131309.png]]
- Basically we've got square bracket INST, and then for the SYStem message
- And then the user is surrounded by INST
- And the user is just a regular string

Note that on the Alpaca one, there's no explicit designation of the system message that differentiates it from the user message, like LLaMA's does.

A look called [LiteLLM](https://github.com/BerriAI/litellm) helps us query all sorts of LLMs using the OpenAI Messages Format (which then is converted under the hood to the appropriate format for your model).

==Post-Processing==
- Based on this, we want to select the actual output out of the generated output.
	1. We might take the output as is; for interacting with a Chat model, we might just be looking at the text as-is.
	2. We might formatting the output for easy visualization
	3. Selecting only the parts of the output that we want to use
	4. Mapping the outputs to other actions
- To give an example of post-processing formatting, a fesature of ChatGPT is that it can generate Markdown tables; code execution is another example:
![[Pasted image 20240615132326.png]]

Output Seleciton
- From a longer response, select the information indicative of an answer
![[Pasted image 20240615132406.png]]
(This seems like #3 above...)
There are various methods for this extraction
- For Classification tasks: Identify Keywords
- For Regression/Numerical problems: Identify Numbers
- For Code: Pull out code snippets in triple-backticks and execute the code.

![[Pasted image 20240615132610.png]]


![[Pasted image 20240615132945.png]]
((I think after the exemplars, there should be a new Input and then an Output:))

![[Pasted image 20240615133028.png]]
Above: It looks like for OpenAI models for the exemplars we still use the System role (rather than the user/assistant roles), with the names of "example_user" and "example_assistant". That's interesting, I suppose.


It turns out that LMs are sensitive to small changes in in-context examples:
- Example ordering
- Label Balance (If you only have positive examples for sentiment classification, that will hurt your accuracy; try to have a similar distribution in your examples, but do some experiments for your use case!)
- Label Coverage (Have examples of all of the classes in your multi-class classification!)

Some effects are counter-intuitive
- Replacing correct labels with random examples sometimes barely hurts accuracy
- More demonstrations can sometimes hurt accuracy

==[[Chain of Thought]] prompting==

![[Pasted image 20240615133702.png]]
- Teaching the model to build up its answer through a series of 
- Provides the model with adaptive computation time; instead of immediately trying to solve the problem in a single go, it will first solve smaller subproblems.

![[Pasted image 20240615134036.png]]
I think this just refers to "zero-shot CoT"; this works because on the internet there are a bunch of examples of (eg) high-quality math explanations that start with "let's think step by step."

(Some results from his collaborators below)

Structuring Outputs as Programs can Help get better results, even if the task isn't a programmatic task.
![[Pasted image 20240615134538.png]]
- They were looking at predicting structured outputs; they wanted to know dependencies between some actions so that they could create something like a dependency tree.
- They structured things in a couple varieties; they had a textual format, a DOT format, and trie structuring the output in Python too (these all say the same thing). They found that structuring the output in Python is the most effective way.
	- This is because Programs are highly structured and included in pre-training data; it's very good at predicting Python, and less good at predicting the obscure DOT format, or your strange text format.
- Asking for JSON is useful too, but there's no guarantee that it will always generate valid JSON. This is where tools like [[Instructor]] and [[Outlines]] come in.

Another paper: Program-aided Language Models
- Using a program to generate outputs can be more precise than asking the LM to do so.
- We created a few examples where we wrote the text in english, and then had the corresponding code, at each step.
- They then generate that code and execute it to get the answer.
==This is implemented in ChatGPT now==, and is great for numeric questions like "How much tax should I pay?"
![[Pasted image 20240615134734.png]]


What about designing prompts? How should we go about doing this?
![[Pasted image 20240615134933.png]]

For manual prompt engineering...
- Make sure the format matches that of the trained model (the chat format) is absolutely ==critical to do!==
![[Pasted image 20240615135144.png]]
If you modify the spacing between the fields, it increases your score by several percentage points -- and if you get it wrong, you can get catastrophically bad results!
==This is a really interesting paper, to me; or rather, an interesting graphic.==

Another interesting thing is how you give instructions models; they should be clear, concise, and easy to understand.
- promptingguide.ai is a decent site; the "General Tips for Designing Prompts" part of the site is cool.
![[Pasted image 20240615135643.png|300]]
((In my opinion, what's really important is having some evaluation set up so you can do some tight iteration loops on your prompts!))


![[Pasted image 20240615135731.png]]

==Prompt Paraphrasing== (Jiang et al 2019)
- This is from Graham and his collaborators.
- Take your prompt, put it through a paraphrasing model, and get a new prompt; you can paraphrase 50 times, see which one gives you the highest accuracy, and use that one.
	- You can do this iteratively, where you paraphrase, filter down candidates, and paraphrase again.


==Gradient-based Discrete Prompt Search== (Shin et al 2020)
- Create a seed prompt, and then calculate gradients into that seed prompt, so you treat each of the tokens in the seed prompt as their own embeddings; you backprop into these embeddings and optimize them to get high accuracy on your dataset. Then when you're done, you clamp them onto the nearest neighbor embedding you already have (eg the nearest neighbor to the embedding I learned is "atmosphere"), so you end up like a prompt with "a real joy. atmosphere alot dialogue Clone totally"
- ![[Pasted image 20240615140034.png|450]]
This has been widely used in adversarial attacks on language models.


==[[Prompt Tuning]]== (Lester et al 2021)
- In Prompt Tuning, they demonstrate that instead of taking your 11B model and training the whole 11B model for many tasks on many datasets, they just train these prompts ... and train it on all the datasets (???) 
![[Pasted image 20240615140445.png|300]]
This only train the embeddings you input into the model


==[[Prefix Tuning]]== (Li and Liang 2021)
- Instead of training just the embeddings that go into the model, they train a prefix that they append at every layer of the model
![[Pasted image 20240615140528.png|450]]

In the next class, we'll talk about PEFT methods that are a more general version of finetuning
![[Pasted image 20240615140716.png]]
