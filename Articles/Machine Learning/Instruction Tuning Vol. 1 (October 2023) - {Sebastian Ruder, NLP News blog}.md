#article 
Link: https://newsletter.ruder.io/p/instruction-tuning-vol-1?utm_source=profile&utm_medium=reader2

Followed by: [[Instruction Tuning Vol. 2 (Nov 2023) {Sebastian Ruder, NLP News}]]
- Instead of the most important IFT datasets, this covers the *latest*, and some new models!
---------

Let's cover some of the most important instruction-datasets for [[Instruction-Tuning]]!

# Instruction Tuning
- The main difference between Instruction Tuning and Supervised Fine-Tuning generally lies in the *data* that the model is trained on!
	- Supervised fine-tuning trains models on ***input examples and their corresponding outputs.***
	- Instruction-tuning *augments* input-output examples with *instructions*

![[Pasted image 20240423171449.png|450]]
Above: Supervised-finetuning versus instruction-tuning


Methods differ for how the instruction-tuning data should be constructed.

Existing instruction-tuning datasets fall into one of two categories:
1. Instructions are added to existing NLP tasks
2. Data from (a) is used to *condition a model* to *generate* new instruction input-output tuples

Let's look at some of the most popular instruction datasets

## [[Natural Instructions]] (Mishra et al, 2022)
- 193k instruction-output examples sourced from 61 existing NLP tasks.
- Crowd-sourcing instructions from each dataset are aligned to a *common schema*, so the instructions are more structured compared to other datasets.
- Outputs are relatively short, however, which makes the data less useful for generating long-form content.
![[Pasted image 20240423173921.png]]


# Natural Instructions v2/ [[Super-NaturalInstructions]] (2022)
- A crowd-sourced collection of instruction-data based on existing NLP tasks and simple synthetic tasks..
- Includes 5M examples across 76 tasks in 55 languages.
- Compared to Natural Instructions, the *instructions are simplified* -- the consist of a task definition and positive/negative examples, with explanations.
![[Pasted image 20240424003200.png]]


# [[Unnatural Instructions]] (2023)
- An automatically collected instruction dataset of 240k examples where [[InstructGPT]] (text-davinci-002) is prompted with three [[Super-NaturalInstructions]] examples (consisting of an instruction, input, and possible output constraints) and asked to generate a new example.
- Covers a more diverse set of tasks than Super-Natural Instructions; while many examples reflect classical NLP tasks, it also includes other interesting tasks like Recipe Correction, Poem Generation, and more.

![[Pasted image 20240424004234.png]]


# [[Self-Instruct]] (Wang et al, 2023)
- Similar to [[Unnatural Instructions]], [[Self-Instruct]] consists of 82k examples automatically generated using InstructGPT conditioned on a set of *seed task examples* (175 tasks in total; one example per task; 8 examples are sampled for in-context learning).
- Self-Instruct decouples the example generation by ==first generating the instruction==, ==then the input (conditioned on instruction),== and ==then the output==.
- For classification tasks, authors choose to instead first generate the possible output labels, and then condition the input generation on each class label, to avoid biasing towards a specific label. 
- Although the generated instructions are mostly valid, the generated outputs are often noisy.

![[Pasted image 20240424004927.png]]


# P3: Public Pool of Prompts (2022)
- A crowd-sourced collection of *prompts* for 177 English NLP tasks.
- For each dataset, about 11 different prompts are available on average, which enables studying the impact of different data prompts.
- Compared to the instructions in the above instruction datasets, P3 prompts are often shorter and less elaborate.

![[Pasted image 20240424005703.png]]


# xP3, xP3mt ( 2023)
- An extension of P3 including 19 multilingual datasets and 11 code datasets, with English prompts.
- They also release a machine-translated version of the data (xP3mt), which contains prompts automatically translated into 20 languages.
- Fine-tuning on multilingual tasks with English prompts further improves performance beyond only fine-tuning on English instruction data.

![[Pasted image 20240424005919.png]]


# Flan 2021 (2022)
- Prompts for 62 English text datasets, with 10 prompt templates for each task.
- For classification tasks, an OPTIONS suffix is appended to the input in order to indicate output constraints.
![[Pasted image 20240424010101.png]]

# Flan 2022 (2022)
- A combination of Flan 2021, P3, Super-Natural Instructions, and additional reasoning, dialog, and program synthesis datasets. 
- The 9 new reasoning datasets are additionally annotated with [[Chain of Thought]] annotations.



# Opt-IML Bench (2022)
- A combination of [[Super-NaturalInstructions]], P3, and [[FLAN]] 2021.
- They additionally include dataset collections on cross-task transfer, knowledge grounding, dialogue, and a larger number of [[CoT]] datasets.
![[Pasted image 20240424010510.png]]


![[Pasted image 20240424010534.png]]


# Important Aspects of Instruction Data
- ==Mixing few-shot settings==: Training with mixed zero-shot and few-shot prompts significantly improve performance in both settings.
- ==Task diversity==: Large models benefit from continuously increasing the number of tasks.
- ==Data augmentation==: Augmenting the data such as by inverting inputs/outputs (eg turning a question answering task into a question generation task) is beneficial.
- ==Mixing weights==: When using a combination of instruction-tuning dataset, appropriately tuning the mixing weights is important.














