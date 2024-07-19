June 14, 2024
University of Maryland (UMD) (Chen et al.)
[GenQA: Generating Millions of Instructions from a Handful of Prompts](https://arxiv.org/abs/2406.10323)
#zotero 
Takeaway: A synthetic dataset produced *directly* from language models, without context provided by human documents, etc. GenQA contains 11,082,134 questions (15,518,076 counting each conversation turn separately) broken into nine splits, each produced using different prompts.
- Splits include (Academic, MMLU, Multiple Choice, Writing, Task, Code, Math, Dialog, General)


---

## Introduction
- Datasets for language model finetuning are typically crafted by hand or built by prompting other LLMs to edit and augment existing human-written datasets.
- Associated costs of data collection and generation means that there's a split between academia and industry; academic datasets often comprise hundreds or thousands of samples, while industrial datasets comprise tens of millions or more.
- [[GenQA]] is an instruction dataset that's written autonomously by an LLM without being conditioned on human-written questions.
	- Authors use a single hand-written meta-prompt to extract millions of diverse questions from an LLM.
	- ((This seems like a red flag ⛳ to me; conditioning on human-written text is a good way to ensure diversity.))
	- To extract diverse questions from an LLM in an automated way, they propose ==generator prompts==, a prompting strategy that boosts the randomness of LLM outputs. Using a small number of these generator prompts, they create a large instruction dataset with many different question styles across a wide array of subjects.


## Related Work
- The [[FLAN]] collection is one of the earliest forms of modern data for language model finetuning... ()
- These datasets, while large and mostly factually correct, suffer from grammatical errors and other major text quality issues.
- Wang et al in [[Self-Instruct]] showed that existing finetuned models like [[InstructGPT]] could be used to produce instruction-output pairs that were in turn useful to finetune other foundation models in an output-based distillation process.
	- [[Alpaca]] turned the baes [[LLaMA]] model into a highly-capable instruction-following version using this technique.
	- In the [[WizardLM]] paper, Xu et al augmented the original Alpaca dataset with [[Evol-Instruct]], a pipeline that "evolves" existing instructions with five types of meta prompts that constrain, deepen, concretize, increase reasoning steps, and generally complicate the original input.
	- In [[Orca]], Mukherjee et al sampled the dataset from the [[FLAN-T5]] paper and rewrote the responses using a mix of [[GPT-3.5]] and [[GPT-4]].
	- In [[OpenHermes2.5 Dataset]]  and [[Tulu]] papers, authors methodically combined multiple sources of existing instruction-tuning datasets to create large singular datasets. Other did similar things for Mathematical reasoning datasets, supplemented by GPT-4.
	- In the Magpie paper (2024, Xu et al), authors extract instructions from LLaMA 3 models by prompting them with an empty string, which often results in a random instruction. 
		- ==We actually use this same technique to create the *general* split of GenQA using GPT-3.5.==
		- After making 2M queries to GPT-3.5, GenQA authors found that the empty string resulted in only 304K unique answers (15% uniqueness).


## GenQA Dataset
- ==Generator Prompts==
	- The quality and diversity of training data are crucial to instruction-tuning. Unfortunately, it can be difficult to induce an LLM to directly produce a large amount of diverse content.
	- A naive method for automating content generation is simply to choose a topic, then construct a *static prompt* asking for content on that topic. The use of static prompts yields low diversity.
		- ==(Asking for "State a random color" 1000 times produces only 33 outputs on Gemeni Pro 1.0)==
	- A ==generator prompt== boosts diversity by asking a model to produce a long list of possible choices, and then select one of the candidates at random.
		- ((Okay, so this is just really expensive and wastes a lot of tokens -- cool.))
```
First, print the heading "Colors:", folllowed by a numbered list of 100 colors. Then, print the heading "Chosen color:". Then print the color nmber {N} on a line by itself.
```
- The placeholder $N$ should be replaced with a random number each time the prompt is invoked. 
	- ==When ran 1000 times, this prompt yields 383 different colors==
- We can use two *nested generators* as follows:
```
First, print the heading "Colors:", folllowed by a numbered list of 100 colors. Then, print the heading "Chosen color:". Then print the color number {N} on a line by itself.
Then, print the heading "Color variants:". Then print a numbered list of 100 different color variants that look like color number {N1}, and don't appear on the original "Colors:" list.
Then, print the heading "Chosen variant:".
Then print the color variant number {N2} on the line itself.
```
- ==When ran 782 unique colors from 1000 runs.==

The output diversity produced when using a *generator prompt* comes from several sources:
- By explicitly creating a list, we guarantee that there will always be at least 100 candidates, and that these candidates will be unique (if the LLM follows directions)
- The process of creating the list requires many sequential samples to be drawn from the model; if the temperature is warm, this compounded randomness makes it unlikely that the same list will be produced more than once. ((?? wtf does this mean, lol))

A study of Generator Prompts for Dataset Curation
- ==Static==: `Write a random complex question and its long answer. Begin your question with "Question:" and your answer with "Answer:"`
	- Results in many repeated/identical outputs, motivating us to provide additional context.
- ==Static-Conditional==: `Write a complex question from the domain {random topic}. Then write the long answer. Your question should not contain the question {random_topic}`
- ==Generator-Conditional==: `List 40 subopics in the domain of {random_topic}. State subtopic {N}. Then write a question that's not about subtopic {N}, but can only be answered with expertise in subtopic {N}, and then write the answer. Both the question and answer should be long. The name of teh subtopic should not appear in the question. Begin your questions with "Question:" and your answer with "Answer:". Be creative.`
	- Conditioning on a topic prevents the model from collapsing into a small number of modes, but it also constrains the range of possible topics.
- ==Generator-Nested==: `List 60 topics that you can answer questions about. State topic {N1}. Then write 60 subtopics about topic {N1}. Then state subtopic {N2}. Then write a question that is not about subtopic {N2}, but can only be answered with expertise about subtopic {N2}. Then write the answer. Both the question and answer should be long. The name of the subtopic {N2} shouldn't appear in the question, and none of the words in subtopic {N2} should be reused in the question. Begin your questions with "Question:" and your answer with "Answer:". Be creative.`
	- ((It's interesting that you can really see from the prompting that they were struggling with the model just asking directly about eg Molecular Biology, instead of about *ideas* related to Molecular Biology.))
	- The downside of this is that the LLM sees the selected indices before writing the list, which might influence the order of the listed items. So we modify to:
- ==Generator Uniform==: `List 60 topics that you can answer questions about. Choose a topic uniformly from this list, and state it. Then write 60 subtopics about the chosen topic. Then choose a subtopic uniformly from the list, and state it. Then write a question that is not about the subtopic, but can only be answered wiht expertise in teh subtopic. Then write the answer. Both the question and answer should be long. The name of the subtopic should not appear in the question, and none of the words in subtopic should be reused in the question. Begin your questions with "Question:" and your answer with "Answer:". Be creative.`
	- In this construction, the random index isn't available to the LLM when the lists are being constructed, as the index is chosen via "sampling" ((Not accurate)) rather than appearing in the prompt.
- Note that the Gemeni model tends to interpret instructions quite literally -- If you ask for questions about Cultural Anthropology, we're likely to get a question about Cultural Anthropology *per se*, such as "Who is the father of Cultural Anthropology", or "What is the most famous textbook in cultural anthropology." We avoid this caveat by prompting for a question that's "not about the subtopic, but can only be answered with expertise in the subtopic."
- The ==generator-conditional== and ==generator-nested prompts,== which were used to create the final academic split, yield the highest diversity.
	- ((Wait, so the Generator Uniform was just... not as diverse? Even thought we made the modifications in the name/hopes of diversity?))

The phrases at the end of the prompts like "Be creative," "Be different," "Be smart," "Be weird," "Don't ask the first thing you think of" ... are referred to as "boosters" by the authors.
- They assess the effect of the booster by sampling some questions and answer pairs generated with and without without a booster for each split in the dataset (n=200 for each split)... and it seems like it helps.

The [[GenQA]] dataset is constructed by either the ==*topic-conditioned*== or ==*generator-nested*== generator prompts.
- For each split, the generator prompts were fed through Gemeni Pro 1.0 many times and outputs were parsed into questions and answers... final questions were deduplicated using an *exact match criteria* on the first two sentences of the question.
	- ((That deduplication using exact match is insane.))

## Finetuning 
- Authors performa an empirical evaluation against other strong finetuning datasets by finetuning [[LLaMA 3]]-8B
	- [[UltraChat]]: A synthetic instruction dataset focusing on multi-turn conversational abilities; we use a filtered version of UltraChat with a total of 200k multi-turn instructions.
	- [[WizardLM]]: WizardLM-Evol-Instruct-V2 (196k single-turn instructions)
- Authors consider tasks from the [[HuggingFace]] [[Open LLM Leaderboard]] and two instruction-following benchmarks.
- Evaluation on ARC, BoolQ, HellaSwag, MMLU, OpenBookQA, PIQA, TruthfulQA, and Winogrande.
- Authors find that they get best finetuning performance using an adjusted sampling ratio for GenQA that updates the smaller splits. They refer to the rebalanced version the dataset as "Full GenQA" and "Subset GenQA".

## Conclusion
- GenQ is an instruction dataset written autonomously by an LLM without conditioning on human questions or using complex multi-stage pipelines.
- We hope the methods in this paper can be a swiss army knife for creating datasets for other domains; experiments indicate that prompt engineering alone can yield millions of diverse training samples with quality as good (or in some cases as surpassing) high-cost human labeling.


## APPENDIX
- Each split was created by forming a "generator-type" "meta-prompt" which was fed to Gemeni 1.0 Pro many times to produce different outputs.
- Academic
	- ![[Pasted image 20240718230732.png]]
- MMLU
	- ![[Pasted image 20240718230742.png]]
- Multiple Choice
	- ![[Pasted image 20240718230758.png]]
- Writing
	- ![[Pasted image 20240718230809.png]]
- Task
	- ![[Pasted image 20240718230820.png]]
- Code
	- ![[Pasted image 20240718230830.png]]
- Math
	- ![[Pasted image 20240718230841.png]]
- Dialog
	- ![[Pasted image 20240718230849.png]]
- General
	- Created by handing GPT-3.5 an empty string, to which it typically responds with an answer to some (unknown) instruction. After generating the answer in this way, the question corresponding to the answer was written by Gemeni.


Abstract
> ==Most public instruction finetuning datasets are relatively small== compared to the closed source datasets used to train industry models. To study questions about finetuning at scale, such as curricula and learning rate cooldown schedules, ==there is a need for industrial-scale datasets==. However, this scale necessitates a data generation process that is almost entirely automated. In this work, ==we study methods for generating large instruction datasets from a single prompt==. With little human oversight, we get LLMs to write diverse sets of instruction examples ranging from simple completion tasks to complex multi-turn dialogs across a variety of subject areas. When finetuning a Llama-3 8B base model, ==our dataset meets or exceeds both [[WizardLM]] and [[UltraChat]] on both knowledge-intensive leaderboard tasks as well as conversational evaluations==. We release our dataset, the "generator" prompts that created it, and our finetuned model checkpoints.


# Paper Figures

![[Pasted image 20240718205857.png|600]]


![[Pasted image 20240718210055.png|600]]
GenQA's prompting strategy results in more diversity than a static prompt that simply asks an LLM for a question/answer pair.

![[Pasted image 20240718211352.png|500]]

![[Pasted image 20240718213020.png|600]]

![[Pasted image 20240718223011.png|500]]

![[Pasted image 20240718223444.png|600]]
Fuck these violin plots

![[Pasted image 20240718223804.png]]
I'm confused by the comparisons here. We know that GenQA has like 10M tokens or so, and then the paper says that the WizardLM and Ultrachat datsets use something like ~200k token subsets...