July 3, 2024
[[Microsoft Research]] (Mitra et al.)
[AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502v1) "The Orca 3 Paper"
#zotero 
Takeaway: A agentic workflow for creating synthetic data tailored to helping a language model learn specific skills, given seed documents (eg textbook chapters, blog posts, code data). The workflow has three pieces:
1. ==Content Transformation Flow==: Transform the seed documents by making it more relevant for the skill we're trying to teach.
2. ==Seed Instruction Generation Flow==: Generate an instruction relevant to the skill and (transformed) seed document. Often uses a set of templates specific to the skill.
3. ==Instruction Refinement Flow==: Use a pair of Suggester and Editor agents to increase the complexity of the task by editing the document/task.
Authors use a 25M dataset created by this technique to finetune Mistral-7B into [[AgentInstruct|Orca 3]], which performs significantly better on benchmarks than the original Mistral-7B-Instruct.
Interestingly, the pipeline doesn't include information on if there's any subsequent filtering (eg via some LLM-as-a-Judge) after any of the steps. Nor do they include the actual prompts used for the above stages, just examples of the outputs at each stage.

They'll be releasing the Orca-3-7b model soon, but when asked about the dataset, they said "You never know, we were able to release the [[Orca-Math]] dataset." Teknium said that the paper made replicating the code kind of impossible, because there were a lot of missing pieces.
> Re: Teacher responses: "We used a mix of many GPT-$, GPT-4 Turbo versions, whatever we could get access to. Teacher response collection can be tricky, but in most cases GPT4 responses are taken as-is. But in some cases, like tool use, GPT4 might not always format correctly so we add checks." [twitter](https://x.com/420_gunna/status/1811927194979193131)... Regarding filtering steps to remove low-quality instructions, they reduced the number of unanswerable questions/instructions, but didn't remove them, so as to teach the model when not to answer (with the assumption here I think that the smart GPT-4 teacher model knows when to not answer, and would, via distillation, teach Orca 3.) But I don't think they did any other sort of quality filtering.

References:
- Podcast: [Microsoft Research Abstracts Podcast: AgentInstruct](https://www.microsoft.com/en-us/research/podcast/abstracts-july-18-2024/)
	- "Automated model repairing: Now that we have ability to generate data for a particular skill... we need basically error handling; something we can plug in and take the y and $\hat{y}$, figures out the error, and figures out how to give feedback (eg training data), so that's where we're working now."
- Blog Post: [The Rise of Agentic Data Generation by Maximme Labonne](https://huggingface.co/blog/mlabonne/agentic-datagen)

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
		- They use it to finetune Mistral-7B into [[Orca 3]], and it seems to score significantly better (eg 20-50% better) on benchmarks like [[AGIEval]], [[MMLU]], [[GSM8K]], [[BIG-Bench Hard|BBH]], and [[AlpacaEval]].

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
	2. ==Seed Instruction Generation Flow==
		- Given a random seed, the seed instruction creation flow randomly picks one of the 18 agents and uses that to obtain a seed (text, text modification instruction) pair.
		- Eg: "{Seed text} ... Task: Rewrite the event details (date, location, abstract deadline) in a more casual tone."
	3. ==Instruction Refinement Flow==
		- Suggestions might include: "Incorporate a fictional narrative," "Translate the event details into a poetic format," and "Frame the event details as a social media post." 
		- Corresponding modified instructions might be: "Rewrite the event details as if you're telling a funny store to a friend using casual and colloquial language, incorporating a fictional narrative," "Transform the event details into a light-hearted poem with rhyming couplets, ensuring that essential information is accurately conveyed in poetic format," and "Craft a social media post that includes event details using internet slang, emojis, and a casual tone"
- Skill example: ==Tool Use==
	- The task of tool use involves enabling models to interact with external tools or services via APIs.
	1. ==Content Transformation Flow==
		- We use source code snippets or API description as the random seed. If source code snippets have been used, we use a content transformation agent to synthesize an API description from the code snippet.
		- The goal of the Content Transformation Flow is to synthesize a list of APIs from the random seed.
		- We either use an API retrieval agent that iteratively searches for similar code to expand the API list, or the agent uses the LLM to *hypothesize* other APIs that might be present in the library.
	2.  ==Seed Instruction Creation Flow== (No example provided)
		- We consume the list of API documentation produced in the previous step and employ a variety of agents to create several tasks of the following types:
			1. Tasks that require the use of a single API:
				- Where the input supplies all necessary parameteres
				- Where the input includes superfluous parameters
				- Where the input is missing some required parameters
			2. Tasks that necessitate the use of *multiple* APIs
				- Where the input provides all necessary parameters
				- Where the input is missing some required parameters
			3. Tasks that require some single API that's not listed among the available APIs
			4. Tasks that require multiple APIs, but black some of the essential APIs in the provided list
	3. ==Instruction Refinement Flow==
		- Increase the complexity, eg by looking at the task and conversation to suggest refinements that increase the number of steps required to solve the task.

## Orca 3
- They implement AgentInstruct flows for 17 capabilities, and crated a collection of 22 million instructions, using unstructured code and text files from [[Knowledge Pile]] and AutoMathText. They also add some samples from [[Orca]], [[Orca 2]], [[Orca-Math]], and other sources like [[Tulu 2]], [[UltraChat]], MetaMath, resulting in ==25.8M paired instructions==.
- Finetuning details:
	- Mistral tokenizer
	- [[L2 Regularization|Weight Decay]] (L2 Regularization) at 0.1
	- 152 Nvidia A100 GPUs with bs=10, trained for 3 epochs, finishing after 200 hours
	- [[AdamW]] with initial LR of 8e-6 and a cosine learning rate schedule and a linear rate warm-up during the first 500 steps.

## Evaluation Results
- Used the [[Orca-Bench]] dataset as a held-out test set, consisting of 100 samples from each of the 17 skills for which data was curated using AgentInstruct (except Open Domain Question Answering (ODQA), where they created two test sets -- an easier and hard one). Authors compare Orca-3 and similar models to GPT-4, normalizing against GPT-4's score on each skill as if it were a "10".  Shows an ==average performance improvement of 33.94% over Orca 2.5 baseline, and an 14.92% over Mistral-Instruct-7B.==
- They evaluate it on a bunch of specific skills and it does well. Check figures.


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

![[Pasted image 20240712141157.png|500]]


![[Pasted image 20240712140007.png|500]]
Interesting examples to me here here are the Brain Teaser, Data To Text, and Fermi

![[Pasted image 20240712142243.png|400]]
Content Transformation Flow (for Reading Comprehension skill): Given a random seed describing Uric Acid and addressing some questions about it, and then the transformation agent turns it into a sort of dense passage including specific facts, etc. Seems like a high-quality transformation.

![[Pasted image 20240712142753.png|400]]
Seed Instruction Generation Flow (for Reading Comprehension skill): Given a passage (previous figure), this is an example of a generated question for the Reading Comprehension skill (specifically the *Strengthening* subtype).

![[Pasted image 20240712150655.png|400]]
Instruction Refinement Flow (for Reading Comprehension skill): Where we try to modify the passage to the make the question unanswerable, alter the answer in a different direction, or make the question/answer choices more complex. See that we use a ==Suggestion + Editor== flow, where a Suggester agent suggest ways to complicate a (passage, question) pair, and the Editor agent makes makes those modifications.

![[Pasted image 20240712152354.png|400]]
Seed Instruction Generation Flow (for Text Modification/rewriting skill):  Given a random seed, the seed instruction creation flow randomly picks one of the 18 agents and uses that to obtain a (text, text modification instruction) pair.
(It seems like for this skill they basically skill the first content transformation step)

![[Pasted image 20240712153059.png|400]]
Instruction Refinement Flow (Text Modification/rewriting skill): Given a (text, text modification instruction), try to complicate the task. The suggester agent proposes changes to the instruction, and the editor agent implements those changes.

![[Pasted image 20240712154659.png|400]]
Content Transformation Flow (for the tool use task): Given code that comes in the form of source code or API documentation, the goal is to create API documentation. In this case, it seems we're given a limited seed of a single function API documentation, and the content transformation flow either hypothesizes what additional examples from the library might look like, or uses retrieval to find similar examples.

![[Pasted image 20240712161728.png|400]]
![[Pasted image 20240712161724.png|400]]
Instruction Refinement Flow (For Tool Use skill): We increase the complexity, suggesting refinements to increase the number of tasks. This is an example of a multi-turn conversation created by the Agent-Instruct flow.

![[Pasted image 20240712170539.png|400]]
Performance on [[Orca-Bench]], a held-out test set, consisting of 100 samples from each of the 17 skills for which data was curated using AgentInstruct (except Open Domain Question Answering (ODQA), where they created two test sets -- an easier and hard one).

![[Pasted image 20240712171355.png|400]]
[[Orca-Bench]] comparison

![[Pasted image 20240712171502.png|400]]
Comparison of [[Orca 3]] against similar models on well-known benchmarks. The (+x%) scores show improvement vs Mistral-7B-Instruct, the most direct comparison.
Note the use of [[DROP]], [[FoFo]], and [[InFoBench]], all of which are new to me but look interesting.

# Non-Paper Figures





-----

# Appendix Figures


## Skill: Reading Comprehension
![[Pasted image 20240712174846.png|300]]
![[Pasted image 20240712174902.png|300]]

## Skill: Text Modification
![[Pasted image 20240712174934.png|300]]

![[Pasted image 20240712175030.png|400]]