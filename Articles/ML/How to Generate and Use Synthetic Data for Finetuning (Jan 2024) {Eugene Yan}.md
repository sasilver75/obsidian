#article 
Link: https://eugeneyan.com/writing/synthetic/

This was an article that Eugene said that he wrote in order to better understand the landscape of synthetic data, in late Jan 2024 (pre-Claude3, pre-Sora, pre-Gemeni 1.5)

Notes:
- One note is that the Self-Instruct paper used GPT-3 to generate data for GPT-3, while the Unnatural Instructions paper used GPT-3.5 to generate data for a weaker t5 model.
- Both the Self-Instruct and Unnatural Instructions paper have bits on how to evaluate the data that was produced. Both required sampling and then manual evaluation, and in both cases, they were only like 50-60% correct; in both cases, even though the data was partially incorrect, it was still useful for instruction-tuning.
	- ((This makes sense if you think about what Wolfe says about fine-tuning mostly being about style transfer than knowledge learning...))

---------

==Synthetic data== refers to data generated via a model or simulated environment, instead of naturally occurring on the internet or annotated by humans.
- Useful for pre-training, instruction-tuning, and preference-tuning.

----
Aside: [[Instruction-Tuning]] vs [[Preference-Tuning]]
- Preference Tuning
	- A method where the model is fine-tuned to align its outputs with human preferences.
	- This can be based on factors like coherence, relevance, politeness, etc.
- Instruction Tuning
	- Involves fine-tuning the model on a dataset of prompt that are accompanied by the desired outputs.
	- Aims to improve the model's ability to follow specific instructions or understand tasks as described in natural language.
	- The training data for instruction-tuning typically consists of a wide variety of tasks, along with examples of how those tasks should be completed.

These are both subsets of [[Supervised Fine-Tuning]]

-----

Relative to human annotation, ==synthetic data is *faster* and *cheaper* to generate task-specific synthetic data.==

Furthermore, ==synthetic data's *quality* and *diversity* of synthetic data often exceeds that of human annotators, leading to improved performance and generalization when models are finetuned on synthetic data!==
- ((Note that this cuts both ways -- Unless you specifically engineer it, I don't imagine that GPT-4 is going to produce synthetic data that has mis-spellings... which *might* be important to you for robustness?))

Finally, ==synthetic data sidesteps privacy and copyright concerns==.

There are two main approaches to generate synthetic data:
1. [[Distillation]] from a stronger model
	- Transfers knowledge and reasoning skills from a stronger learner to a weaker (but more efficiently-trained student), optimizing for response quality + compute efficiency.
2. Self-Improvement on the model's own output
	- +: Gives the model the ability to learn from its responses via an iterative loop. Avoids external dependencies and contractual restrictions.
	- -: Limits the learning to the model's *initial* abilities, and can amplify biases and errors.

Notable Models in table:

|                  | Pretraining                     | Instruction-tuning                                                                                                         | Preference-tuning           |
| ---------------- | ------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| Distillation     | TinyStories, [[Phi-1]], Phi-1.5 | Unnatural Instructions, [[Alpaca]], [[Vicuna]], [[Orca]], [[Orca 2]], WizardLM, WizardCoder, MagicCoder, Phi-1 (exercises) | Starling-7B                 |
| Self-Improvement | AlphaGeometry, [[WRAP]]         | Self-Instruct, [[SPIN]], Instruction Backtranslation, ReST, [[Constitutional AI]]                                          | SteerLM, Self-Rewarding CAI |
In the context of synthetic data, *==pretraining==* would involve building the model's base knowledge on generated data, or by augmenting real-world datasets with model-generated data.

In the context of synthetic data, *==instruction-tuning==* would involve us building synthetic *instruction-response pairs* to improve the model's comprehension of nuanced queries, as well as improve response precision.

In the context of synthetic data, *==preference-tuning==* relies on a mix of positive and negative *synthetic feedback* to reward the model for desired behaviors.

-----

# Self-Improvement vs Distillation

- The [[Self-Instruct]] and Unnatural Instructions papers were published a day apart (in Dec 2022), but take very different approaches to generate synthetic data.
	- Self-instruct bootstraps synthetic data from the model itself
	- Unnatural Instructions gets it from an external, stronger model

### Self-Instruct

Self-Instruct improves the instruction-following ability of a non-finetuned model by bootstrapping off of its own generations!
![[Pasted image 20240312190430.png]]
- Above:
	- Given a dataset of task/instruction pairs...
	- The LM generates *instructions*, *input context*, and *responses* from the model
	- Then, they filter invalid or similar examples before using the remaining samples to finetune the original model.

For seed data, the Self-Instruct authors started with 175 human-written tasks from the authors and their colleagues.

The tasks look something like this:
```json
{
  "id": "seed_task_0",
  "name": "breakfast_suggestion",
  "instruction": "Is there anything I can eat for a breakfast that doesn't include eggs, 
                  yet includes protein, and has roughly 700-1000 calories?",
  "instances": [
    {
      "input": "",
      "output": "Yes, you can have 1 oatmeal banana protein shake and 4 strips of bacon. 
                 The oatmeal banana protein shake may contain 1/2 cup oatmeal, 60 grams 
                 whey protein powder, 1/2 medium banana, 1tbsp flaxseed oil and 1/2 cup 
                 water, totalling about 550 calories. The 4 strips of bacon contains 
                 about 200 calories."
    }
  ],
  "is_classification": false
}
```

To generate synthetic instructions, they use eight randomly sampled seed instructions for few-shop prompting. For the first round of generation, all sampled instructions come from the seed set. For subsequent rounds, 2 of 8 instructions are sampled from model-generated instructions to promote diversity.

After sampling a task and generating an instruction:
1. They classify (probably using a LM) whether the generated instruction is a classification task or not.

```
ask: Given my personality and the job, tell me if I would be suitable.
Is it classification? Yes

Task: Give me an example of a time when you had to use your sense of humor.
Is it classification? No

...
```

2. Next, for each synthetic instruction, they generate input context and output responses.
	- This is done in two main ways:
		1. Input-First
		2. Output-First (when the input isn't required to be generated first. This is usually in a classification task)

```
Come up with examples for the following tasks. 
Try to generate multiple examples when possible.
If the task doesn’t require additional input, you can generate the output directly.

Task: Which exercises are best for reducing belly fat at home?
Output:
- Lying Leg Raises
- Leg In And Out
- Plank
- Side Plank
- Sit-ups

Task: Extract all the country names in the paragraph, list them separated by commas.
Example 1
Paragraph: Dr. No is the sixth novel by the English author Ian Fleming to feature his 
British Secret Service agent James Bond. Written at Fleming’s Goldeneye estate in 
Jamaica, it was first published in the United Kingdom by Jonathan Cape in 1958. In 
the novel Bond looks into the disappearance in Jamaica of two fellow MI6 operatives 
who had been investigating Doctor No. Bond travels to No’s Caribbean island and meets 
Honeychile Rider, who is there to collect shells. They are captured and taken to a 
luxurious facility carved into a mountain. The character of Doctor No, the son of a 
German missionary and a Chinese woman, was influenced by Sax Rohmer’s Fu Manchu 
stories. Dr. No was the first of Fleming’s novels to face widespread negative reviews 
in Britain, but it was received more favourably in the United States. 
Output: English, British,  Jamaica, the United Kingdom, German, Chinese, Britain, 
the United States.

Task: Converting 85 F to Celsius.
Output: 85°F = 29.44°C
```
Above:
- The challenge with generating responses *input-first* is that it tends to bias output responses toward one label.
	- For example, given the task of *grammar error detection*, it will typically generate grammatically correct input.
	- Thus, we propose the output-first approach *for the classification task!*

Overall, the self-instruct authors generated 52,000 instructions and 82,000 input-output pairs
The authors sampled 200 of these instructions and an expert annotator (the paper author 😜) labeled whether the input-output pair was "correct":
![[Pasted image 20240312191931.png|300]]
Above:
- ((Eugene and I are surprised that ==synthetic data that was only *half correct* was useful for finetuning==! The authors reasoned that even though the synthetic data contained errors, most were still in the correct format, or partially correct -- this still made the synthetic data useful for *instruction-tuning* (but perhaps not for base-knowledge pretraining)))

To evaluate the improvement in instruction-following ability, they used the evaluation set of "Super-NaturalInstructions" -- it has 119 tasks with 100 instances in each task -- ==The authors saw that their self-instructed GPT-3 outperformed vanilla GPT-3 by 33%.==


### Unnatural Instructions
- Generates synthetic data from an *external model* (GPT-3.5), in contrast to Self-Instruct.
- The synthetic data is then used to finetune `t5-lm`, a language model variant of `t5-11b`.
- The authors started with a seed set of 15 human-written examples, and then, to to generate new instructions and input (i.e. context), they prompted GPT-3.5 with three examples to generate a fourth synthetic sample.
	- The authors used 5 different seeds of three-shot examples to generate a distilled dataset of 68k examples.
	- To encourage creativity when using the same few-shot examples, the authors applied [[Top-P Sampling|Nucleus Sampling]] (top-p) with p=.99.

-----
Aside: Nucleus Sampling (Top-P)

Given an output distribution over tokens, we have a hyperparameter `topP`, which chooses from the smallest possible set of tokens whose summed probability exceeds topP during decoding. Given this set of tokens, we re-normalize the probability distribution based on each token's respective probability, and then sample! 

----

Here's some examples of Unnatural Instruction's data generation process:

```
Example 1
Instruction: You are given a science question (easy-level) and four answer options 
(associated with “A”, “B”, “C”, “D”). Your task is to find the correct answer based on 
scientific facts, knowledge, and reasoning. Do not generate anything else apart from one 
of the following characters: ‘A’, ‘B, ‘C’, ‘D’. There is only one correct answer for 
each question.
Input: Which part of a bicycle BEST moves in a circle? (A) Seat (B) Frame (C) Foot 
pedal (D) Kickstand
Constraints: The output should be one of the following characters: ‘A’, ‘B, ‘C’, ‘D’.

Example 2
Instruction: You are given a negative review and your task is to convert it to a positive 
review by one or more making minimal changes. Avoid changing the context of the review.
Input: we stood there in shock, because we never expected this.
Constraints: None.

Example 3
Instruction: In this task, you are given two sentences taken from a conversation, and 
your job is to classify whether these given sentences are sequential or not. We will 
mark the given sentence pair as ’True’ if it’s sequential, otherwise ’False’. The
two sentences are spoken by two different people.
Input: Noah: When and where are we meeting? :), Madison: I thought you were busy...?
Constraints: None.
```
Above: Prompt used to generate new instructions and inputs

They then removed the instruction-input pairs that:
1. Didn't contain expected fields
2. Were identical to the examples in the few-shot prompt
3. Were duplicates

To generate *responses*, they conditioned GPT-3.5 with the synthetic instruction-input examples! In this stage, they applied greedy decoding to prioritize correctness over creativity.

To evaluate the quality of synthetic data, they audited 200 samples.
Again only about 60% of the generated dat was good, but it was deemed as all useful for use as fine-tuning data.

They finetuned `t5-lm` on the synthetic instructions and found it to outperform the vanilla t5-lm.

---------

# Distillation Techniques
- Since unnatural instructions, several models have been finetuned on distilled synthetic data, usually from OpenAI APIs.
- These models explored ways to improve instruction-following on increasingly complex queries, with some focused on code.

[[Alpaca]] (Stanford, March 2023) finetuned llama-6b on 52k instruction-following samples that were generated from GPT-3.5.
- This cost less than $500
- They used the same 175 human-written instruction-response pairs from the Self-Instruct seed set and generated *more* instruction-response pairs from GPT-3.5 via few-shot prompting ((Recall that the Self-Instruct paper only had access to GPT-3)).

[[Vicuna]] (Stanford, March 2023) finetuned llama-7b and llama-13b on user conversations from ShareGPT.com
- They filtered out inappropriate and low-quality samples, resulting in 125k conversations which were used for instruction-tuning.

[[WizardLM]] (Microsoft, April 2023) generated how to generate *more complex* instructions and responses via GPT-3.5
- It distinguishes between *in-depth evolution* and *in-breadth evolution*.
	- ==In-depth evolution== makes instructions more complex via five types of prompts, such as:
		- Adding constraints
		- Deepening
		- Increased reasoning steps
		- ...
	- ==In-breadth evolution== increases topic coverage, skills coverage, and overall dataset diversity.

![[Pasted image 20240312210637.png|400]]
Above: examples of [[Evol-Instruct]], the technique that was used to generate data that was used to train WizardLM

[[Orca]] (Microsoft, June 2023)
- Explores how *smaller models* can imitate the reasoning processes of a larger, stronger model via explanation traces.




















