---
aliases:
  - Orca 3
---
July 3, 2024
[[Microsoft Research]]
[AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502v1)
#zotero 
Takeaway: ...


---

## Introduction
- Synthetic data is increasingly important to all stages of training, from pre-training to instruction-tuning and RLHF.
- Generating high-quality synthetic data is hard -- pre-training on it can lead to model collapse/gradual degeneration. Creating high-quality and diverse synthetic data is hard.
- In post-training synthetic data, most approaches include starting with a set of prompts and using a powerful model like GPT-4 to generate either responses to these prompts or an expanded set of prompts.
- ==We've seen the rise of Agentic (especially Multi-Agent) workflows that can generate high-quality data surpassing the capabilities of underlying LLMs by using workflows with reflection and iteration, where agents can look back at solutions, generate critiques, and improve solutions, while also using tools (search apis, calculators, code interpreters)==to address limitations of LLMs.
- ==Generative Teaching and Orca AgentInstruct==
	- Generating synthetic data for post-training often relies on an existing prompt set used as-is, or used as seeds for generating more instructions.
	- We refer to =="Generative Teaching"== as the setting of generating abundant amounts of diverse, challenging, and high-quality data to teach a *particular skill* to an AI model.
	- This paper's [[AgentInstruct]] ==focuses on creating demonstrations and feedback data==, and ==requires only raw documents as input==. With generic data used as seeds, AgentInstruct can be used to teach an LLM a general capability (eg Math, Reasoning, RAG, etc.). Domain-specific data (eg gaming, finance) can also be used as seeds to improve the model in a certain specialization.
		- AgentInstruct ==generates both prompts and responses==. It uses a large number of agents and a ==taxonomy of over 100 subcategories== to create diverse and high-quality prompts and responses.
	- Using raw data (unstructured text documents or source code) as seed has two benefits: it's abundant and can promote learning more general capabilities.
	- ==They create a synthetic post-training dataset of 25 million prompt-repose pairs,== covering a range of skills including writing, reasoning, math, RAG, tool use, etc.
		- They use it to finetune Mistral-7B into [[AgentInstruct|Orca 3]], and it seems to score significantly better (eg 20-50% better) on benchmarks like [[AGIEval]], [[MMLU]], [[GSM8K]], [[[BIG-Bench Hard|BBH]], and [[AlpacaEval]].

## Generative Teaching: AgentInstruct
- When creating synthetic data, we want to create a *large* amount of *high-quality, diverse, complex/nuanced/challenging* data.
- AgentInstruct uses a structured approach:
	- For every raw seed (eg textbook chapters, web articles, code snippets) in your collection:
		1. Transform the seed with the aid of a *content-transformation agent* (==Content Transformation Flow==)
		2. Route it trough a series of *instruction-creation agents* to create a diverse set of instructions (==Seed Instruction Creation Flow==)
		3. Utilize another group of *refinement agents* to iteratively refine the complexity and quality of the seed instructions (==Refinement Flow==)
- So we use agentic flows to automate the generation process, allowing us to create data at scale (from automated flows), with high diversity (based on diverse seed documents), and varying complexity (benefitting from iterative and refinement patterns supported by agentic flows).
- We define three flows:
	1. ==Content Transformation Flow==: Converts the raw seed (eg textbook chapter) into an *intermediate representation* that simplifies the creation of instructions tailored to *specific objectives*. 
		- It comprises multiple agents and is often instrumental in the generation of high-quality data, serving as an additional means to introduce diversity.
	2. ==Seed Instruction Generation Flow==: Comprises multiple agents and takes as input the transformed seeds from the Content Transformation Flow and *generates a set of diverse instructions*. 
		- The goal is to introduce diversity, for which it often *relies on a pre-defined, but extensible taxonomy*.
	3. ==Instruction Refinement Flow==: *Takes instructions* generated from the Seed Instruction Flow *and iteratively enhances their complexity and quality*. 
		- We use the concept of "==Suggester-Editor Agent==s," where the ==Suggester== agents propose various *approaches* to increase the intricacy of the initial instructions, and the ==Editor== modifies the instructions in accordance with these instructions.
- Authors implemented these flows for ==17 different skills== (eg reading comprehension, creative writing, tool use), *each having multiple subcategories*.
	- Full list of skills: Reading Comprehension, Open-Domain Question Answering, Text Modification, Web Agent, Brain Teaser, Analytical Reasoning, Multiple-Choice Questions, Dat to Text, Fermi, Coding, Text Extraction, Text Classification, Retrieval-Augmented Generation, Tool Use, Creative Content Generation, Few-Shot Reasoning, Conversation.
- Below I'll go through the flow for Reading Comprehension, Text Modification, and Tool Use
- Skill example: ==Reading Comprehension==
	- A critical skill involving processing and understanding text, encompassing decoding, fluency, and vocabulary knowledge. Questions range from explicit information/literal comprehension to requiring inferences, understanding vocabulary in context, analyzing text structure and argumentation, critically evaluating the content, and synthesizing information from disparate locations.
	1. ==Content Transformation Flow==: The objective is to translate/transform arbitrary articles into well-crafted pieces with the particular stylistic/logical framework conducive to the formulation of a wide array of reading comprehension question types.
		- We use a suite of nine content transformation agents for generating argument passages, debates and conversations, long passages, meeting transcripts, poems, satirical content, etc.
		- Given a seed article, we randomly pick one of the transformation agents to attempt to generate transformed text passages.
	2. ==Seed Instruction Generation Flow==: Authors have a collection of ***43 reading comprehension question types***, including literal comprehension questions, critical comprehension questions, reasoning, identifying assumptions, ordering events, etc.
		- We have multiple agents targeting these categories. Each of these agents takes a piece of text as input and generates a *list of questions* based on a predefined question type. The orchestration process engages *a subset of these agents,* determined by the content transformation agent in the preceding step (? How).
	3. ==Instruction Refinement Flow==
		- Contains multiple Suggestion-Editor agents that will go through each o the (passage, question) pairs and create *more* such pairs with the following goals:
			1. Modify the *passage* to make the question *unanswerable*
			2. Modify the *passage* to *alter the answer* -- if possible, in the opposite direction.
			3. Modify the *question or answer choices* (if applicable) to *make them more complex*.
		- So the suggester will suggest ways to complicate the (passage, question), and the editor will implement those ideas, generating new questions.
- Skill example: ==Text Modification==
	- Text modification involves editing written content to enhance its quality and effectiveness, or alter its attributes. Involves correcting spelling and grammar, clarifying ideas, reorganizing content, adjusting tone, ensuring style consistency, removing redundancies, etc.
	- (They skip describing the content transformation step here)
	2. Seed Instruction Genera
- Skill example: ==Tool Use==
	- 



Abstract
> Synthetic data is becoming increasingly important for accelerating the development of language models, both large and small. Despite several successful use cases, researchers also raised concerns around ==model collapse== and ==drawbacks of imitating other models==. This discrepancy can be attributed to the fact that synthetic data varies in quality and diversity. ==Effective use of synthetic data usually requires significant human effort in curating the data==. We focus on using synthetic data for post-training, specifically creating data by powerful models to teach a new skill or behavior to another model, we refer to this setting as ==Generative Teaching==. We introduce ==AgentInstruct==, an ==extensible agentic framework for automatically creating large amounts of diverse and high-quality synthetic data==. AgentInstruct can create both the prompts and responses, using only raw data sources like text documents and code files as seeds. We demonstrate the utility of AgentInstruct by creating a post training dataset of 25M pairs to teach language models different skills, such as text editing, creative writing, tool usage, coding, reading comprehension, etc. The dataset can be used for instruction tuning of any base model. We post-train Mistral-7b with the data. When comparing the resulting model Orca-3 to Mistral-7b-Instruct (which uses the same base model), we observe significant improvements across many benchmarks. For example, 40% improvement on AGIEval, 19% improvement on MMLU, 54% improvement on GSM8K, 38% improvement on BBH and 45% improvement on AlpacaEval. Additionally, it consistently outperforms other models such as LLAMA-8B-instruct and GPT-3.5-turbo.



# Paper Figures
![[Pasted image 20240712125513.png|500]]
This picture shows the improvement that the AgentInstruct formula made to the IFT process on the base Mistral-7B model, compared to Mistral's own IFT process that resulted in Mistral-7B-Instruct. It's not clear to me in this part of the paper how much of this is benchmark-maximizing behavior.

![[Pasted image 20240712133542.png|500]]
Note the use of:
1. *transformation agents* (==Content Transformation Flow==): Turn the seed document into an intermediate representation appropriate for the use case of teaching some specific skill.
2. *instruction-creation agents* (==Seed Instruction Creation Flow==): Generates an instruction from the provided intermediate representation. Often uses some sort of taxonomy to help instruction generation.
3. *refinement agents* (==Refinement Flow==): Complexifies the instruction using a Suggestion agent and an Editor agent.

![[Pasted image 20240712141157.png]]


![[Pasted image 20240712140007.png|500]]
Interesting examples to me here here are the Brain Teaser, Data To Text, and Fermi

![[Pasted image 20240712142243.png|400]]
Content Transformation Flow: Given a random seed describing Uric Acid and addressing some questions about it, and then the transformation agent turns it into a sort of dense passage including specific facts, etc. Seems like a high-quality transformation.

![[Pasted image 20240712142753.png|400]]
Seed Instruction Generation Flow: Given a passage (previous figure), this is an example of a generated question for the Reading Comprehension skill (specifically the *Strengthening* subtype).

![[Pasted image 20240712150655.png|400]]
Instruction Refinement Flow: Where we try to modify the passage to the make the question unanswerable, alter the answer in a different direction, or make the question/answer choices more complex. See that we use a ==Suggestion + Editor== flow, where a Suggester agent suggest ways to complicate a (passage, question) pair, and the Editor agent makes makes those modifications.

# Non-Paper Figures