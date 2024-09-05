December 19, 2022 (8mo after [[Super-NaturalInstructions]], 11 months before ChatGPT)
[[Meta AI Research]] , Tel Aviv University
Paper: [Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor](https://arxiv.org/abs/2212.09689)
#zotero 
Takeaway: A dataset of 240k synthetic instruction-following examples created by prompting a language model (text-davinci-002, an instruction-tuned variant of GPT-3) with three seed examples of `(instruction, input, output_constraint)`, and eliciting a fourth, to create 64k examples. We then use a LM to generate outputs/responses for these 64k, then discard the output constraints. The set is then expanded by prompting the model to *rephrase* each instruction, creating a total of 240k examples of instructions, inputs, and outputs.

Relevance: I think this might be one of the earliest examples of creating synthetic instruction-tuning datasets. It's relevant because they were able to generate 240k instructions by starting from only 15 manually-constructed examples. ((I really wonder whether starting with more instructions would have helped results.))
- =="To the best of our knowledge, this is the first general-purpose NLP dataset that was automatically generated."==
- Interesting to me that the dataset is only considered ~60% "correct" (though the non-correct ones *do* have benefit to train on)

----

Notes:
- An automatically collected instruction dataset of 240k examples where [[InstructGPT]] (text-davinci-002) is prompted with three [[Super-NaturalInstructions]] examples (consisting of an instruction and input (but not an output)) and asked to generate a new example. 
	- We collect data in a fully automatic manner by prompting a pretrained LM with three examples from [[Super-NaturalInstructions]], and asking the model to generate a fourth. We repeat this process with 5 different seeds, i.e. the entire process requires only 15 instruction examples to automatically produce 64k diverse triplets of instructions, inputs, and outputs. We then further diversify the dataset by generating additional natural langauge paraphrases of each instruction, while preserving the contents of any input arguments and outputs, expanding the dataset to approximately 240k examples.
	- "Although more than ==50% of generated examples are indeed correct==, even incorrect examples typically contain valuable information for instruction-tuning." ((I don't love this!))
- Covers a ==more diverse set of tasks than Super-Natural Instructions==; while many examples reflect classical NLP tasks, it also includes other interesting tasks like Recipe Correction, Poem Generation, and more.
	- Authors note that LMs can generate creative and diverse data, which is hard to do with crowd workers, who lack intrinsic motivation to create novel examples, and typically collapse into predictable heuristics to form annotation artifacts.
- Authors note that for the core dataset generation, they use "stochastic decoding" to generate example inputs (to promote creativity), followed by "deterministic decoding" to generate their outputs (for accuracy).
- Each example in the core dataset contains four fields:
	1. An **instruction** describing the task: "Write whether the following review is positive or negative"
	2. An **input** argument instantiating the instruction, creating a specific example of the task.
	3. Output space **constraints**, which detail the restrictions on the task's output space (may be "None").
	4. A textual **output** reflecting correct execution of the instruction
	- The first three fields are the model's input, and the output acts as the reference for training/evaluation. The constraints are meant to guide the model during output generation, and is ==discarded== after generating the outputs.
- Input generation
	- The first step in the data generation pipeline is to generate examples of `(instruction-input-constraints)` triplets, which we do by prompting a model with three demonstrations wrapped in a meta-prompt incentivizing the model to create a fourth example. 
	- Authors use 5 different seeds of 3 demonstrations each to generate the entire core 64k dataset (requiring only 15 manually-constructed samples).
	- To obtain various examples using the same prompt, authors do decoding via [[Top-P Sampling|Nucleus Sampling]] (Top-P sampling) to promote creativity, with *p=0.99*
- Filtering
	- Authors then apply three filters to the generated examples to remove generations that don't include all three required fields (instruction, input, constraints), as well duplicates between eachother.
- Output Generation
	- Given a generated example *x*, we generate the corresponding output *y* by conditioning a pretrained LM with the instruction, input argument, and constraints. Here, we apply [[Greedy Decoding]] to prioritize correctness, over creativity.
		- ((Interesting. I think assistants' "creative" responses can be aligned with human preferences!))
		- ((Recall: I believe we can remove/drop the output constraints, at this point; so now we have (instruction, input, response) triplets.)))
- Template Expansion
	- Examples in the core dataset have a strict instruction-input-output format; to increase the format diversity, we collect alternative formulations that *preserve the content of the original instructions,* asking an LM to reformulate tasks in the core dataset, generating alternative formulations for each generated task. These are often shorter and less-formal than original instructions. We end with 240k examples.
		- The prompt contains two examples of instructions and their alternative formulations.
- Evaluation: Dataset Correctness
	- Authors analyzed 200 sampled examples
	- ==113 of the 200 analyzed examples (56.5%) are *correct*== ((This seems not good!))
	- Of the 87 incorrect examples:
		- 9 (4%) had incomprehensible instructions
		- 35 (17.5%) had an input that didn't mask the task description
		- 43 (21.5%) had incorrect outputs
- "Unnatural Instructions leads to stronger or equal performance for every dataset, except Super-NaturalInstruction itself" 🤔
- Regarding the meta-prompt provided with the 3-shot prompt used to generate the core dataset, they experiment with *minimal, enumeration, and verbose* styles of prompts, and observe that the simple enumeration elicits more informative examples than either the minimal or verbose approaches.

Abstract
> ==Instruction tuning== enables pretrained language models to perform new tasks from inference-time natural language descriptions. These approaches ==rely on vast amounts of human supervision in the form of crowdsourced datasets or user interactions==. In this work, we introduce ==Unnatural Instructions==: a large ==dataset of creative and diverse instructions, collected with virtually no human labor==. We collect ==64,000 examples== by *prompting a language model with three seed examples of instructions and eliciting a fourth*. This set is then *expanded by prompting the model to rephrase each instruction*, creating a total of approximately 240,000 examples of instructions, inputs, and outputs. Experiments show that despite containing a fair amount of noise, training on Unnatural Instructions rivals the effectiveness of training on open-source manually-curated datasets, surpassing the performance of models such as T0++ and Tk-Instruct across various benchmarks. These results demonstrate the potential of model-generated data as a cost-effective alternative to crowdsourcing for dataset expansion and diversification.

# Paper Figures
![[Pasted image 20240508170533.png|300]]
The data generation prompt, where we provide three (seed?) examples of *instructions and inputs*, and then ask the model to generate a fourth. This is the techniques that's used to generate 64k examples, before subsequently using the rephrasing technique to balloon it to 240k.
- Note that this is just generating `(instruction, input, constraint)` triplets , rather than `(instruction,input,constraint, output)` quartets
![[Pasted image 20240508173758.png]]
Above: Basically the same as the previous figure. Note that in this figure we also include constraints on the output.

![[Pasted image 20240508173831.png]]
Above: Note it seems that the "stochastic sampling" that they refer to in the paper is actually this [[Top-P Sampling|Nucleus Sampling]] that they show in this page (for creativity)... And then note that they use [[Greedy Decoding]] for the output generation (for accuracy).

![[Pasted image 20240508180106.png]]
Above: Some interesting examples of categories of instructions present in the Unnatural Instructions dataset as as result of instruction generation.

![[Pasted image 20240508182128.png]]
Above: On the "correctness" of the generated data (only just under 60% were correct, I believe)

# Non-Paper Figures


![[Pasted image 20240424004237.png]]