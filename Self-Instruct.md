Dec 20, 2022
Various universities, including [[Allen Institute]]
Paper: [Self-Instruct: Aligning Language Models with Self-Generated Instructions](https://arxiv.org/abs/2212.10560)
#zotero 
Takeaway: Self-Instruct is a method for generating instruction-following datasets with minimal human-labeled data by starting with a limited seed set of manually written tasks, sampling a task for the task pool, generating instructions for new tasks, creating input/output instances for the task, and filtering low-quality or repeated instructions.


Note: *Self-Instruct* refers to the technique/pipeline, but it's also often used as a descriptor for datasets that are created with this technique -- eg "XYZ, a Self-Instruct dataset". It confusingly *also* can refer to the 52k instruction dataset produced and released in this paper.

-----

Notes:
- Note that their process of Self-Instruct refers to the pipeline of generating tasks using a **vanilla pretrained language model**; I'm not sure that this is a requirement, though, and I'm sure that an instruction-tuned model to produce high-quality instructions might do better?
- Pipeline
	1. Instruction Generation: Authors initiate the task pool with 175 tasks (1 instructions and 1 instance for each task). For every step, they sample 8 task instructions from the pool as in-context examples, where 6 must be human-generated tasks, and 2 are from the model-generated tasks from previous steps, to promote diversity.
	2. Classification Task Identification: After generating a task, we need two different approaches fro classification and non-classification tasks. We prompt the LM in a few-shot way to determine this, using 12 classification instructions and 19 non-classification instructions from the seed tasks.
	3. Instance Generation: Given instructions and task type, generate instances for each instruction type independently. 
		- Authors found that the **input-first approach** (generating input based on instruction, then response) can generate inputs biased towards one label, especially for classification tasks. So authors use this input-first approach only for *non-classification tasks*.
		- As a result, authors use an **output-first approach** for the *classification-tasks*, where we first generate the possible class labels, and then condition the input generation on each class label.
			- ((It seems like this isn't necessarily output first, more like output-possibilities first? Do they generate N responses, one for each class?))
	4. Filtering and Postprocessing: To encourage diversity, ==a new instruction is added to the task pool only when its [[ROUGE]]-L similarity (testing longest-common substring) with any existing instruction is less than .7.==
		- "We also exclude instructions that contain some specific keywords that usually can't be processed by LMs (eg image, picture, graph)" ((? How did it generate them if it can't process them))
		- Invalid generations are identified and filtered based on heuristics (too long or too short, output being a repetition of the input, etc.)
- Although the generated instructions are mostly valid, the generated outputs are often noisy. Authors note that even when the outputs aren't factually correct, training on them still has a positive impact on instruction-following ability.
	- ((Though I don't really like this... surely you're still taking small steps in the direction of ignorance))
- Self-Instruct can be viewed as a form of [[Distillation]], though it differs (in this paper) in two ways:
	1. The distillee and the distiller are the same model
	2. The output of the distillee is instructions, rather than labels or logits






Paper Abstract:
> Large "instruction-tuned" language models (i.e., finetuned to respond to instructions) have demonstrated a remarkable ability to generalize zero-shot to new tasks. Nevertheless, they depend heavily on human-written instruction data that is often limited in quantity, diversity, and creativity, therefore hindering the generality of the tuned model. We introduce Self-Instruct, a ==framework for improving the instruction-following capabilities of pretrained language models by bootstrapping off their own generations==. Our ==pipeline generates instructions, input, and output samples from a language model, then filters invalid or similar ones before using them to finetune the original model==. Applying our method to the vanilla GPT3, we demonstrate a 33% absolute improvement over the original model on Super-NaturalInstructions, on par with the performance of InstructGPT-001, which was trained with private user data and human annotations. For further evaluation, we curate a set of expert-written instructions for novel tasks, and show through human evaluation that tuning ==GPT3 with Self-Instruct outperforms using existing public instruction datasets by a large margin==, leaving only a 5% absolute gap behind InstructGPT-001. Self-Instruct provides an almost annotation-free method for aligning pre-trained language models with instructions, and we release our large synthetic dataset to facilitate future studies on instruction tuning.

# Paper Figures
![[Pasted image 20240506204037.png|200]]

![[Pasted image 20240507122308.png]]
Above: The Self-Instruct process

![[Pasted image 20240507131428.png|300]]
Above: A description of the dataset produced in the paper using GPT-3. "Instances" refers to input/output pairs, given an instruction. I'm still not sure how they end up generating multiple instances for a given instruction.

![[Pasted image 20240507132654.png]]
Above: See that the Self-Instruct-Instruction-Tuned GPT3 model (via the GPT finetuning API) is almost as good as [[InstructGPT]], and better than GPT trained on the T0/SuperNI dataset, which required (?) large human labeling efforts. Note that InstructGPT was both SFT'd and RLHF'd, whereas this paper's model was just SFT'd.

![[Pasted image 20240507132853.png|300]]
Above: It's interesting that human evaluation seemed to roughly plateau at ~52k instructions. Note that InstructGPT was both instruction-fine-tuned (dataset size not released) as well as RLHF'd. "Overall, we see consistent improvement as we grow the dataset size, but the improvement almost plateaus after 16k."

![[Pasted image 20240507145000.png]]
Above: The prompt for generating new instructions

![[Pasted image 20240507145023.png|300]]
Above: The prompt for determining whether something is a classification task or not

![[Pasted image 20240507145048.png|300]]
Above: Prompt used for input-output "instance" generation (input, response) given a task using the ==input-first approach== (for non-classification tasks). Sometimes tasks won't have inputs (Null), and the instruction is mostly in the task.

![[Pasted image 20240507145229.png|300]]
Above: The input-output "instance" generation prompt, given a task description. This is the ==output-first generation==, which is what we use when something is a classification task, so as to not bias towards a specific label.


# Non-Paper Figures

![[Pasted image 20240415233101.png]]

![[Pasted image 20240418164722.png]]

![[Pasted image 20240424005031.png]]